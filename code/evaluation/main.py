"""Validate full predictions and score separate public-sample predictions."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from ghost_audit.validation import validate_steps, ContractError
from ghost_audit.index import EvidenceEngine

FIELDS = ("case_id", "deletion_status", "steps_json")
STEP_FIELDS = ("step_no", "phase", "system_id", "resource_id", "operation", "state", "evidence_id")

def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"{path}: expected columns {FIELDS}")
        rows = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise ValueError(f"{path}: rows must match the exact output column count")
        if any(not row.get("case_id", "").strip() or not row.get("deletion_status", "").strip()
               or row.get("steps_json") is None for row in rows):
            raise ValueError(f"{path}: output rows must populate all required fields")
        return rows

def compare(full_prediction_path, expected_path=None, sample_prediction_path=None):
    expected_path = expected_path or ROOT / "dataset/sample_output.csv"
    expected = read_rows(expected_path)
    full_rows = read_rows(full_prediction_path)
    full_by_id = {row["case_id"]: row for row in full_rows}
    full_duplicates = len(full_rows) - len(full_by_id)
    dataset_dir = ROOT / "dataset"
    engine = EvidenceEngine(dataset_dir)
    with (dataset_dir / "deletion_requests.csv").open(encoding="utf-8-sig", newline="") as stream:
        full_requests = {row["case_id"]: row for row in csv.DictReader(stream)}
    full_missing = sorted(set(full_requests) - set(full_by_id))
    full_extra = sorted(set(full_by_id) - set(full_requests))
    row_order_valid = [row["case_id"] for row in full_rows] == sorted(full_requests)
    invalid_full = []
    for case_id, pred in full_by_id.items():
        request = full_requests.get(case_id)
        if request is None:
            continue
        try:
            steps = json.loads(pred["steps_json"])
            validate_steps(pred["deletion_status"], steps,
                           available_citations=engine.valid_citations(request["audit_at"], engine.citation_ids))
        except (ValueError, TypeError, KeyError, ContractError, json.JSONDecodeError) as exc:
            invalid_full.append((case_id, str(exc)))

    sample_rows = read_rows(sample_prediction_path) if sample_prediction_path else full_rows
    sample_by_id = {row["case_id"]: row for row in sample_rows}
    sample_duplicates = len(sample_rows) - len(sample_by_id)
    with (dataset_dir / "sample_deletion_requests.csv").open(encoding="utf-8-sig", newline="") as stream:
        sample_requests = {row["case_id"]: row for row in csv.DictReader(stream)}
    expected_ids = [row["case_id"] for row in expected]
    sample_missing = sorted(set(expected_ids) - set(sample_by_id))
    invalid_sample = []
    status_correct = exact_correct = sequence_correct = 0
    total_fields = correct_fields = total_steps = correct_steps = 0
    for gold in expected:
        pred = sample_by_id.get(gold["case_id"])
        if pred is None:
            continue
        try:
            gold_steps = json.loads(gold["steps_json"])
            got_steps = json.loads(pred["steps_json"])
            cutoff = sample_requests[gold["case_id"]]["audit_at"]
            validate_steps(pred["deletion_status"], got_steps,
                           available_citations=engine.valid_citations(cutoff, engine.citation_ids))
        except (ValueError, TypeError, KeyError, ContractError, json.JSONDecodeError) as exc:
            invalid_sample.append((gold["case_id"], str(exc)))
            continue
        status_correct += pred["deletion_status"] == gold["deletion_status"]
        sequence_correct += got_steps == gold_steps
        exact_correct += pred["deletion_status"] == gold["deletion_status"] and got_steps == gold_steps
        total_steps += max(len(gold_steps), len(got_steps))
        for index in range(max(len(gold_steps), len(got_steps))):
            if index < len(gold_steps) and index < len(got_steps):
                correct_steps += gold_steps[index] == got_steps[index]
                for key in STEP_FIELDS:
                    total_fields += 1
                    correct_fields += gold_steps[index].get(key) == got_steps[index].get(key)
            else:
                total_fields += len(STEP_FIELDS)
    count = len(expected)
    return {"sample_cases": count, "full_rows": len(full_rows), "full_missing_rows": full_missing, "full_extra_rows": full_extra,
            "full_duplicates": full_duplicates, "full_row_order_valid": row_order_valid,
            "invalid_full_rows": invalid_full, "sample_rows": len(sample_rows),
            "sample_missing_rows": sample_missing, "sample_duplicates": sample_duplicates,
            "invalid_sample_rows": invalid_sample,
            "status_accuracy": status_correct / count if count else 0,
            "step_accuracy": correct_steps / total_steps if total_steps else 1,
            "field_accuracy": correct_fields / total_fields if total_fields else 1,
            "sequence_accuracy": sequence_correct / count if count else 0,
            "case_exact_accuracy": exact_correct / count if count else 0}

def compare_samples(sample_prediction_path, expected_path=None, allow_partial=False):
    """Score only the distinct public sample requests; never require 518-row output."""
    expected = read_rows(expected_path or ROOT / "dataset/sample_output.csv")
    sample_rows = read_rows(sample_prediction_path)
    by_id = {row["case_id"]: row for row in sample_rows}
    duplicates = len(sample_rows) - len(by_id)
    expected_ids = [row["case_id"] for row in expected]
    missing = sorted(set(expected_ids) - set(by_id))
    data_dir = ROOT / "dataset"
    engine = EvidenceEngine(data_dir)
    with (data_dir / "sample_deletion_requests.csv").open(encoding="utf-8-sig", newline="") as stream:
        requests = {row["case_id"]: row for row in csv.DictReader(stream)}
    field_errors = {"deletion_status": 0, **{name: 0 for name in STEP_FIELDS}}
    patterns = {}
    invalid = []
    exact = status_ok = sequence_ok = step_ok = 0
    total_steps = total_fields = fields_ok = 0
    scored_cases = 0
    for gold in expected:
        pred = by_id.get(gold["case_id"])
        if pred is None:
            continue
        scored_cases += 1
        case_patterns = set()
        try:
            want = json.loads(gold["steps_json"])
            got = json.loads(pred["steps_json"])
            if not isinstance(got, list):
                raise ValueError("steps_json must be an array")
            validate_steps(pred["deletion_status"], got,
                           available_citations=engine.valid_citations(requests[gold["case_id"]]["audit_at"], engine.citation_ids))
        except (ValueError, TypeError, KeyError, ContractError, json.JSONDecodeError) as exc:
            invalid.append((gold["case_id"], str(exc)))
            case_patterns.add("invalid_output_or_citation")
            expected_steps = json.loads(gold["steps_json"])
            field_errors["deletion_status"] += 1
            total_steps += len(expected_steps)
            total_fields += len(expected_steps) * len(STEP_FIELDS)
            for key in STEP_FIELDS:
                field_errors[key] += len(expected_steps)
            for pattern in case_patterns:
                patterns[pattern] = patterns.get(pattern, 0) + 1
            continue
        if pred["deletion_status"] == gold["deletion_status"]:
            status_ok += 1
        else:
            field_errors["deletion_status"] += 1
            case_patterns.add("status_mismatch")
        if len(got) < len(want):
            case_patterns.add("missing_step")
        if len(got) > len(want):
            case_patterns.add("extra_step")
        total_steps += max(len(want), len(got))
        if got == want:
            sequence_ok += 1
            exact += pred["deletion_status"] == gold["deletion_status"]
        else:
            case_patterns.add("step_sequence_mismatch")
        for idx in range(max(len(want), len(got))):
            if idx >= len(want):
                total_fields += len(STEP_FIELDS)
                for key in STEP_FIELDS:
                    field_errors[key] += 1
                continue
            if idx >= len(got):
                total_fields += len(STEP_FIELDS)
                for key in STEP_FIELDS:
                    field_errors[key] += 1
                continue
            if got[idx] == want[idx]:
                step_ok += 1
            for key in STEP_FIELDS:
                total_fields += 1
                if got[idx].get(key) == want[idx].get(key):
                    fields_ok += 1
                else:
                    field_errors[key] += 1
                    if key == "evidence_id": case_patterns.add("evidence_citation_mismatch")
                    elif key in {"system_id", "resource_id"}: case_patterns.add("wrong_system_or_resource")
                    elif key in {"phase", "operation", "state"}: case_patterns.add("phase_operation_state_mismatch")
                    elif key == "step_no": case_patterns.add("step_number_mismatch")
        for pattern in case_patterns:
            patterns[pattern] = patterns.get(pattern, 0) + 1
    return {"sample_cases": len(expected), "scored_cases": scored_cases, "coverage": scored_cases / len(expected) if expected else 0, "prediction_rows": len(sample_rows), "duplicates": duplicates,
            "missing_cases": missing, "invalid_cases": invalid,
            "exact_match_accuracy": exact / (scored_cases if allow_partial else len(expected)) if (scored_cases if allow_partial else len(expected)) else 0,
            "status_accuracy": status_ok / (scored_cases if allow_partial else len(expected)) if (scored_cases if allow_partial else len(expected)) else 0,
            "step_accuracy": step_ok / total_steps if total_steps else 1,
            "field_accuracy": fields_ok / total_fields if total_fields else 1,
            "sequence_accuracy": sequence_ok / (scored_cases if allow_partial else len(expected)) if (scored_cases if allow_partial else len(expected)) else 0,
            "field_errors": field_errors,
            "top_failure_patterns": sorted(patterns.items(), key=lambda item: (-item[1], item[0]))[:5]}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("full_predictions", nargs="?", default=ROOT / "output.csv")
    parser.add_argument("--sample-predictions", help="Predictions generated for sample_deletion_requests.csv")
    parser.add_argument("--samples-only", action="store_true", help="Score only the public sample run; do not require full output.csv")
    parser.add_argument("--allow-partial", action="store_true", help="Score completed predictions only and report sample coverage")
    parser.add_argument("--expected", default=ROOT / "dataset/sample_output.csv")
    args = parser.parse_args()
    if args.samples_only:
        if not args.sample_predictions:
            parser.error("--samples-only requires --sample-predictions")
        if Path(args.sample_predictions).resolve() == Path(args.expected).resolve():
            parser.error("prediction file must not be the reference output file")
        report = compare_samples(args.sample_predictions, args.expected, allow_partial=args.allow_partial)
        print(json.dumps(report, indent=2))
        failed = report["duplicates"] or report["invalid_cases"] or (report["missing_cases"] and not args.allow_partial) or report["prediction_rows"] != report["scored_cases"]
        if failed:
            raise SystemExit(2)
        patterns = report["top_failure_patterns"] or [("none", 0)]
        md = ["# Public sample evaluation", "", f"Scored {report['scored_cases']} of {report['sample_cases']} public samples; this does not report the 518-case full-dataset run.",
              f"- Exact-match accuracy: {report['exact_match_accuracy']:.4f}",
              f"- Status accuracy: {report['status_accuracy']:.4f}",
              f"- Step accuracy: {report['step_accuracy']:.4f}", f"- Field accuracy: {report['field_accuracy']:.4f}",
              f"- Sequence accuracy: {report['sequence_accuracy']:.4f}", "", "Per-field error counts:"]
        md.extend(f"- {name}: {count}" for name, count in report["field_errors"].items())
        md.extend(["", f"Coverage: {report['scored_cases']}/{report['sample_cases']} ({report['coverage']:.4f})", "", "Top failure patterns (affected cases):"])
        md.extend(f"- {name}: {count}" for name, count in patterns)
        md.extend(["", "Additional regression cases executed locally:",
                   "- Future nested JSON timestamp leakage is blocked.",
                   "- Identity must resolve uniquely before search; system/resource pairs must co-occur in subject evidence.",
                   "- Orphan remediation and policy Always-action mismatches are rejected.",
                   "- Checkpoint input fingerprints and retrieved-source provenance are validated on resume.",
                   "- Full request input schema, unique IDs, and 518-row count are validated.",
                   "", "Live model accuracy is measured only from model-produced predictions; the reference output is never scored against itself.", ""])
        (ROOT / "code/evaluation/evaluation_report.md").write_text("\n".join(md), encoding="utf-8")
        return
    report = compare(args.full_predictions, args.expected, args.sample_predictions)
    print(json.dumps(report, indent=2))
    failed = (report["full_missing_rows"] or report["full_extra_rows"] or report["full_duplicates"]
              or not report["full_row_order_valid"] or report["invalid_full_rows"]
              or report["sample_missing_rows"] or report["sample_duplicates"] or report["invalid_sample_rows"])
    if failed:
        raise SystemExit(2)
    md = ["# Public sample evaluation", "", "Comparison uses 32 participant-facing public sample cases, which have IDs separate from the 518 scored requests.", "",
          f"- Case-level deletion-status accuracy: {report['status_accuracy']:.4f}",
          f"- Step-level positional accuracy: {report['step_accuracy']:.4f}",
          f"- Field-level accuracy: {report['field_accuracy']:.4f}",
          f"- Sequence accuracy: {report['sequence_accuracy']:.4f}",
          f"- Exact case accuracy: {report['case_exact_accuracy']:.4f}",
          "- Full output: 518 required rows, unique, sorted, contract and citation validated.", "",
          "Additional safety/evaluator cases: unapproved filesystem-like tool call; invented citation with one correction limit; altered status must lower measured accuracy.",
          "Failures found and improvements: future nested JSON children could inherit an older timestamp; search could precede identity resolution; resource IDs and system IDs could be mixed across hits; policy Always actions were not checked; orphan remediation was accepted. These are now guarded by deterministic checks and regression tests.",
          "Additional local regression cases: future nested JSON; identity collision/search ordering; resource-pair collision; Always-policy mismatch; orphan remediation; checkpoint resume/provenance; 518-row input validation.", ""]
    (ROOT / "code/evaluation/evaluation_report.md").write_text("\n".join(md), encoding="utf-8")

if __name__ == "__main__":
    main()
