"""Run the Gemini deletion-audit agent with resumable per-case checkpoints."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ghost_audit.agent import DeletionAuditAgent, AgentError, AgentOutputError
from ghost_audit.checkpoint import CheckpointStore
from ghost_audit.gemini import GeminiBackend, DEFAULT_MODEL
from ghost_audit.groq import GroqBackend, DEFAULT_MODEL as DEFAULT_GROQ_MODEL
from ghost_audit.envfile import load_project_env
from ghost_audit.identity import IdentityResolver
from ghost_audit.index import EvidenceEngine
from ghost_audit.policy import DeletionPolicy
from ghost_audit.timeutil import parse_time


def _safe_error(error):
    text = " ".join(str(error).split())
    key = os.environ.get("GEMINI_API_KEY", "")
    if key:
        text = text.replace(key, "[REDACTED]")
    text = re.sub(r"(?i)bearer\s+\S+", "Bearer [REDACTED]", text)
    text = re.sub(r"(?i)(api[_ -]?key|access[_ -]?token|authorization)\s*[:=]\s*\S+", r"\1=[REDACTED]", text)
    text = re.sub(r"AIza[0-9A-Za-z_-]{20,}", "[REDACTED]", text)
    text = re.sub(r"gsk_[0-9A-Za-z]{20,}", "[REDACTED]", text)
    text = re.sub(r"org_[0-9A-Za-z]+", "[ORG_REDACTED]", text)
    return text[:400]


def _record_preflight_counts(state, backend):
    state["model_lookup_requests"] = state.get("model_lookup_requests", 0) + backend.model_lookup_attempts
    state["smoke_function_call_attempts"] = state.get("smoke_function_call_attempts", 0) + backend.smoke_function_call_attempts


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as target:
            json.dump(value, target, ensure_ascii=False, indent=2)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _load_state(path, has_checkpoint):
    if Path(path).exists():
        try:
            with Path(path).open(encoding="utf-8") as source:
                state = json.load(source)
            if not isinstance(state, dict) or state.get("schema_version") != 1:
                raise ValueError
            defaults = {"generation_attempts": 0, "transient_retries": 0, "agent_turns": 0,
                        "input_tokens": 0, "output_tokens": 0, "input_tokens_complete": False,
                        "output_tokens_complete": False, "errors": [], "completed_cases": [],
                        "model_lookup_requests": 0, "smoke_function_call_attempts": 0}
            for key, value in defaults.items():
                state.setdefault(key, value)
            if not isinstance(state["errors"], list) or not isinstance(state["completed_cases"], list):
                raise ValueError
            return state
        except (OSError, ValueError, json.JSONDecodeError):
            raise ValueError("run state is invalid; preserve it and inspect before resuming") from None
    return {"schema_version": 1, "generation_attempts": 0, "transient_retries": 0,
            "agent_turns": 0, "input_tokens": 0, "output_tokens": 0,
            "input_tokens_complete": not has_checkpoint, "output_tokens_complete": not has_checkpoint,
            "errors": [], "completed_cases": [], "model_lookup_requests": 0,
            "smoke_function_call_attempts": 0}


def _reconcile_checkpoint_state(state, store):
    checkpoint_ids = set(store.rows)
    usage_ids = {item.get("case_id") for item in state["completed_cases"] if isinstance(item, dict)}
    unaccounted = sorted(checkpoint_ids - usage_ids)
    dangling_usage = sorted(usage_ids - checkpoint_ids)
    if unaccounted or dangling_usage:
        state["checkpoint_rows_without_usage"] = unaccounted
        state["usage_rows_without_checkpoint"] = dangling_usage
        state["input_tokens_complete"] = False
        state["output_tokens_complete"] = False
        state["attempt_counts_complete"] = False
        return True
    return False



def dataset_fingerprint(dataset_dir):
    """Fingerprint the participant-visible tables and evidence used by predictions."""
    root = Path(dataset_dir).resolve()
    relevant = [root / name for name in ("user_identities.csv", "user_audit_log.csv",
                "system_activity.csv", "deletion_rules.csv", "evidence_files.csv")]
    evidence_root = root / "evidence"
    relevant.extend(path for path in evidence_root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in sorted(relevant, key=lambda item: item.relative_to(root).as_posix()):
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("dataset fingerprint input escapes dataset directory")
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()

def load_requests(path):
    required = ["case_id", "email", "requested_at", "reported_status", "reported_at", "audit_at", "request_text"]
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != required:
            raise ValueError(f"request columns must exactly match {required}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("request rows must match the exact input column count")
    ids = set()
    for row in rows:
        if any(not isinstance(row.get(field), str) or not row[field].strip() for field in required):
            raise ValueError("request rows must have every required field populated")
        if row["case_id"] in ids:
            raise ValueError("duplicate request case_id")
        ids.add(row["case_id"])
        for time_field in ("requested_at", "reported_at", "audit_at"):
            parse_time(row[time_field])
        if row["reported_status"] not in {"COMPLETE", "PARTIALLY_COMPLETE", "NOT_STARTED"}:
            raise ValueError("request row contains an unsupported reported_status")
    return sorted(rows, key=lambda row: row["case_id"])



def write_final_output(path, requests, rows, citations):
    """Atomically write a complete, unique, case-sorted challenge CSV."""
    path = Path(path)
    request_ids = [row["case_id"] for row in requests]
    output_rows = list(rows)
    output_ids = [row.get("case_id") for row in output_rows]
    if (len(request_ids) != len(set(request_ids)) or len(output_ids) != len(set(output_ids))
            or set(output_ids) != set(request_ids)):
        raise ValueError("final output requires exactly one row for every input case_id")
    if any(set(row) != {"case_id", "deletion_status", "steps_json"} for row in output_rows):
        raise ValueError("final output rows do not match output contract")
    request_by_id = {row["case_id"]: row for row in requests}
    from ghost_audit.validation import validate_steps
    for row in output_rows:
        try:
            steps = json.loads(row["steps_json"])
            cutoff = request_by_id[row["case_id"]]["audit_at"]
            validate_steps(row["deletion_status"], steps, available_citations=citations(cutoff))
        except (KeyError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid final output row for {row['case_id']}: {exc}") from None
    output_rows.sort(key=lambda row: row["case_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=["case_id", "deletion_status", "steps_json"],
                                    extrasaction="raise")
            writer.writeheader()
            writer.writerows(output_rows)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _validate_run_paths(dataset, input_path, output, checkpoint_path, state_path):
    dataset = Path(dataset).resolve()
    paths = {"input": Path(input_path).resolve(), "output": Path(output).resolve(),
             "checkpoint": Path(checkpoint_path).resolve(), "run-state": Path(state_path).resolve()}
    if paths["input"] == paths["output"] or paths["input"] == paths["checkpoint"] or paths["input"] == paths["run-state"]:
        raise ValueError("run artifacts must not overwrite the request input")
    for name in ("output", "checkpoint", "run-state"):
        if paths[name].is_relative_to(dataset):
            raise ValueError(f"{name} path must be outside the participant dataset")
    if len({paths["output"], paths["checkpoint"], paths["run-state"]}) != 3:
        raise ValueError("output, checkpoint, and run-state paths must be distinct")
    return paths


def process_batch(requests, agent, backend, store, state, state_path, limit=None):
    pending = [row for row in requests if row["case_id"] not in store.rows]
    if limit is not None:
        pending = pending[:limit]
    for request in pending:
        before_attempts = backend.generate_request_attempts
        before_retries = backend.transient_retries
        try:
            outcome = agent.run(request)
            row = {"case_id": outcome.case_id, "deletion_status": outcome.deletion_status,
                   "steps_json": json.dumps(outcome.steps_json, ensure_ascii=False, separators=(",", ":"))}
            store.commit(row, getattr(outcome, "retrieved_source_ids", ()))
        except Exception as exc:
            safe_exc = exc if isinstance(exc, AgentError) else AgentError(f"{type(exc).__name__}: request failed safely")
            state["generation_attempts"] += backend.generate_request_attempts - before_attempts
            state["transient_retries"] += backend.transient_retries - before_retries
            failed_usage = getattr(agent, "last_usage", {})
            state["agent_turns"] = state.get("agent_turns", 0) + failed_usage.get("agent_turns", 0)
            if failed_usage.get("input_tokens_complete") is False:
                state["input_tokens_complete"] = False
            else:
                state["input_tokens"] = state.get("input_tokens", 0) + failed_usage.get("input_tokens", 0)
            if failed_usage.get("output_tokens_complete") is False:
                state["output_tokens_complete"] = False
            else:
                state["output_tokens"] = state.get("output_tokens", 0) + failed_usage.get("output_tokens", 0)
            state["errors"].append({"case_id": request["case_id"],
                                    "generation_attempts": backend.generate_request_attempts - before_attempts,
                                    "error": _safe_error(safe_exc),
                                    "tool_trace": list(getattr(agent, "last_tool_trace", ())),
                                    "output_diagnostics": list(getattr(agent, "last_output_diagnostics", ())),
                                    "at_utc": datetime.now(timezone.utc).isoformat()})
            _atomic_json(state_path, state)
            if isinstance(exc, AgentError) and not isinstance(exc, AgentOutputError):
                raise
            if not isinstance(exc, AgentOutputError):
                raise AgentError(_safe_error(safe_exc)) from None
            print(f"Case {request['case_id']} failed validation; checkpoint retained and continuing.", flush=True)
            continue
        for error in state.get("errors", []):
            if error.get("case_id") == outcome.case_id and not error.get("resolved"):
                error["resolved"] = True
                error["resolved_at_utc"] = datetime.now(timezone.utc).isoformat()
        attempts = backend.generate_request_attempts - before_attempts
        retries = backend.transient_retries - before_retries
        state["generation_attempts"] += attempts
        state["transient_retries"] += retries
        state["agent_turns"] += outcome.api_calls
        if outcome.input_tokens is None:
            state["input_tokens_complete"] = False
        else:
            state["input_tokens"] += outcome.input_tokens
        if outcome.output_tokens is None:
            state["output_tokens_complete"] = False
        else:
            state["output_tokens"] += outcome.output_tokens
        state["completed_cases"].append({"case_id": outcome.case_id,
                                          "generation_attempts": attempts,
                                          "agent_turns": outcome.api_calls,
                                          "input_tokens": outcome.input_tokens,
                                          "output_tokens": outcome.output_tokens,
                                          "tool_calls": getattr(outcome, "tool_calls", 0),
                                          "tool_trace": list(getattr(outcome, "tool_trace", ()))})
        _atomic_json(state_path, state)
        print(f"Checkpointed {len(store.rows)}/{len(requests)}", flush=True)
    return store.ordered_rows()



def _write_usage_report(input_path, requests, state, model_name, provider_name="Google Gemini"):
    if input_path.name == "deletion_requests.csv" and len(requests) == 518:
        report_path = ROOT / "code/evaluation/usage_report.md"
        title = "# Full-dataset model usage report"
        scope = "Completed participant requests: 518/518."
    elif input_path.name == "sample_deletion_requests.csv" and len(requests) == 32:
        report_path = ROOT / "code/evaluation/sample_usage_report.md"
        title = "# Public sample model usage"
        scope = "Completed public sample requests: 32/32."
    else:
        return
    tokens_ok = state.get("input_tokens_complete") and state.get("output_tokens_complete")
    token_total = state.get("input_tokens", 0) + state.get("output_tokens", 0)
    lines = [title, "", scope, "Provider/model: " + provider_name + " / " + model_name,
             f"Agent model turns: {state.get('agent_turns', 0)}",
             f"Agent generation request attempts including retry: {state.get('generation_attempts', 0)}",
             f"Transient retries: {state.get('transient_retries', 0)}",
             f"Model availability lookups: {state.get('model_lookup_requests', 0)}",
             f"Function-call smoke attempts (excluded from agent turns): {state.get('smoke_function_call_attempts', 0)}"]
    if tokens_ok:
        lines += [f"Agent input tokens: {state['input_tokens']}", f"Agent output tokens: {state['output_tokens']}",
                  f"Total agent tokens: {token_total}", f"Average agent tokens per request: {token_total / len(requests):.2f}"]
    else:
        lines += ["Agent input tokens: unavailable (provider metadata or checkpoint usage is incomplete)",
                  "Agent output tokens: unavailable (provider metadata or checkpoint usage is incomplete)",
                  "Total and average tokens per request: unavailable"]
    try:
        price_prefix = "GROQ" if provider_name.casefold() == "groq" else "GEMINI"
        rate_in = float(os.environ[price_prefix + "_INPUT_PRICE_PER_MILLION_USD"])
        rate_out = float(os.environ[price_prefix + "_OUTPUT_PRICE_PER_MILLION_USD"])
        if tokens_ok and all(0 <= value < float("inf") for value in (rate_in, rate_out)):
            cost = (state["input_tokens"] * rate_in + state["output_tokens"] * rate_out) / 1_000_000
            lines += [f"Estimated cost (USD): {cost:.6f}", f"Estimated cost per request (USD): {cost / len(requests):.8f}"]
        else:
            lines.append("Estimated cost: unavailable (token metadata or finite non-negative configured rates missing)")
    except (KeyError, ValueError):
        lines.append("Estimated cost: unavailable (verified input/output price rates were not configured)")
    lines += ["No API keys or credentials are recorded.", ""]
    report_path.write_text("\n".join(lines), encoding="utf-8")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset/deletion_requests.csv", help="Request CSV relative to repository root")
    parser.add_argument("--output", default="output.csv", help="Prediction CSV relative to repository root")
    parser.add_argument("--checkpoint", help="Checkpoint CSV (defaults to OUTPUT.checkpoint.csv)")
    parser.add_argument("--run-state", help="Sanitized attempt/usage JSON (defaults to OUTPUT.run.json)")
    parser.add_argument("--limit", type=int, help="Process at most this many new requests")
    parser.add_argument("--provider", choices=("gemini", "groq"), help="Model backend (defaults to LLM_PROVIDER or gemini)")
    parser.add_argument("--smoke-function-call", action="store_true", help="Make one function-call smoke request before processing")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    load_project_env(ROOT)
    provider = (args.provider or os.environ.get("LLM_PROVIDER", "gemini")).strip().casefold()
    if provider not in {"gemini", "groq"}:
        parser.error("LLM_PROVIDER must be gemini or groq")
    dataset = ROOT / "dataset"
    input_path = (ROOT / args.input).resolve()
    output = (ROOT / args.output).resolve()
    checkpoint_path = (ROOT / args.checkpoint).resolve() if args.checkpoint else output.with_suffix(output.suffix + ".checkpoint.csv")
    state_path = (ROOT / args.run_state).resolve() if args.run_state else output.with_suffix(output.suffix + ".run.json")
    paths = _validate_run_paths(dataset, input_path, output, checkpoint_path, state_path)
    input_path, output, checkpoint_path, state_path = (paths["input"], paths["output"], paths["checkpoint"], paths["run-state"])
    requests = load_requests(input_path)
    if input_path.name == "deletion_requests.csv" and len(requests) != 518:
        raise ValueError("deletion_requests.csv must contain exactly 518 cases")
    engine = EvidenceEngine(dataset)
    citations = lambda audit_at: engine.valid_citations(audit_at, engine.citation_ids)
    context_fingerprint = dataset_fingerprint(dataset)
    store = CheckpointStore(checkpoint_path, requests, citations, context_fingerprint)
    state = _load_state(state_path, bool(store.rows))
    if _reconcile_checkpoint_state(state, store):
        _atomic_json(state_path, state)
    pending = [row for row in requests if row["case_id"] not in store.rows]
    if not pending:
        store.save()
        write_final_output(output, requests, store.ordered_rows(), citations)
        default_model = (os.environ.get("GROQ_MODEL", "").strip() or DEFAULT_GROQ_MODEL) if provider == "groq" else os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        model_name = state.get("model", default_model)
        provider_name = state.get("provider", "Groq" if provider == "groq" else "Google Gemini")
        _write_usage_report(input_path, requests, state, model_name, provider_name)
        print(f"All {len(requests)} requests already checkpointed; no model calls made.")
        return
    backend = GroqBackend.from_environment() if provider == "groq" else GeminiBackend.from_environment()
    state["provider"] = backend.provider_name
    state["model"] = backend.model_name
    try:
        backend.verify_model()
        if args.smoke_function_call:
            backend.smoke_function_call()
    except AgentError as exc:
        _record_preflight_counts(state, backend)
        state["errors"].append({"case_id": None,
                                "generation_attempts": backend.generate_request_attempts - backend.smoke_function_call_attempts,
                                "smoke_function_call_attempts": backend.smoke_function_call_attempts,
                                "latencies_ms": dict(backend.last_smoke_latencies_ms),
                                "phase": "preflight_or_smoke", "error": _safe_error(exc),
                                "at_utc": datetime.now(timezone.utc).isoformat()})
        _atomic_json(state_path, state)
        raise
    _record_preflight_counts(state, backend)
    if args.smoke_function_call:
        state["smoke_function_call_successes"] = state.get("smoke_function_call_successes", 0) + 1
    _atomic_json(state_path, state)
    agent = DeletionAuditAgent(engine, IdentityResolver(dataset), DeletionPolicy(dataset), backend)
    process_batch(requests, agent, backend, store, state, state_path, args.limit)
    if len(store.rows) == len(requests):
        write_final_output(output, requests, store.ordered_rows(), citations)
        _write_usage_report(input_path, requests, state, backend.model_name, backend.provider_name)
    else:
        print(f"Partial checkpoint only; final output not written ({len(store.rows)}/{len(requests)} complete).", flush=True)
    print(f"Agent generation turns: {state['agent_turns']}")
    print(f"Generation API attempts: {state['generation_attempts']}; transient retries: {state['transient_retries']}")
    if state.get("smoke_function_call_attempts"):
        print(f"Smoke function-call attempts: {state['smoke_function_call_attempts']}")
    unresolved_errors = sum(not error.get("resolved", False) for error in state["errors"])
    print(f"Completed predictions: {len(store.rows)}/{len(requests)}; unresolved sanitized errors: {unresolved_errors}")
    if state["input_tokens_complete"] and state["output_tokens_complete"]:
        print(f"input_tokens={state['input_tokens']} output_tokens={state['output_tokens']} total_tokens={state['input_tokens'] + state['output_tokens']}")
    else:
        print("token_usage=incomplete SDK metadata or pre-existing checkpoint")


if __name__ == "__main__":
    try:
        main()
    except AgentError as exc:
        print(f"Agent stopped safely: {_safe_error(exc)}", file=sys.stderr)
        raise SystemExit(2)
    except ValueError as exc:
        print(f"Run stopped safely: {_safe_error(exc)}", file=sys.stderr)
        raise SystemExit(2)
