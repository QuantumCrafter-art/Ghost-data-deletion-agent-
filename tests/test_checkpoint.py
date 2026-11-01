import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from ghost_audit.agent import AgentError, AgentOutputError
from ghost_audit.checkpoint import CheckpointStore
runner_spec = importlib.util.spec_from_file_location("runner_main", ROOT / "code/main.py")
runner_module = importlib.util.module_from_spec(runner_spec)
runner_spec.loader.exec_module(runner_module)
process_batch, _safe_error = runner_module.process_batch, runner_module._safe_error
write_final_output = runner_module.write_final_output
_validate_run_paths = runner_module._validate_run_paths
load_requests = runner_module.load_requests
_record_preflight_counts = runner_module._record_preflight_counts

spec = importlib.util.spec_from_file_location("sample_eval", ROOT / "code/evaluation/main.py")
eval_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_module)


def not_applicable(case_id):
    return {"case_id": case_id, "deletion_status": "NOT_APPLICABLE", "steps_json": "[]"}


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT.parent)
        self.root = Path(self.tmp.name)
        self.requests = [{"case_id": f"case_{i}", "audit_at": "2025-01-01T00:00:00Z"} for i in range(3)]
        self.path = self.root / "pred.checkpoint.csv"
        self.state_path = self.root / "pred.run.json"
        self.citations = lambda audit_at: set()

    def tearDown(self):
        self.tmp.cleanup()

    def test_checkpoint_validates_contract_and_resume_does_not_reprocess(self):
        store = CheckpointStore(self.path, self.requests, self.citations)
        state = {"generation_attempts": 0, "transient_retries": 0, "agent_turns": 0,
                 "input_tokens": 0, "output_tokens": 0, "input_tokens_complete": True,
                 "output_tokens_complete": True, "errors": [], "completed_cases": []}
        class Backend:
            generate_request_attempts = 0
            transient_retries = 0
        backend = Backend()
        class Agent:
            def __init__(self): self.seen = []
            def run(self, request):
                self.seen.append(request["case_id"])
                backend.generate_request_attempts += 1
                return SimpleNamespace(case_id=request["case_id"], deletion_status="NOT_APPLICABLE",
                    steps_json=[], api_calls=1, input_tokens=2, output_tokens=1)
        agent = Agent()
        process_batch(self.requests, agent, backend, store, state, self.state_path, limit=1)
        self.assertEqual(agent.seen, ["case_0"])
        resumed = CheckpointStore(self.path, self.requests, self.citations)
        process_batch(self.requests, agent, backend, resumed, state, self.state_path, limit=1)
        self.assertEqual(agent.seen, ["case_0", "case_1"])
        self.assertEqual(set(resumed.rows), {"case_0", "case_1"})
        self.assertEqual(state["generation_attempts"], 2)

    def test_bad_or_unknown_checkpoint_rows_are_rejected(self):
        CheckpointStore(self.path, self.requests, self.citations).save()
        self.path.write_text("case_id,deletion_status,steps_json\ncase_outside,NOT_APPLICABLE,[]\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unknown case_id"):
            CheckpointStore(self.path, self.requests, self.citations)

    def test_checkpoint_requires_retrieval_provenance_and_input_fingerprint(self):
        citations = lambda audit_at: {"source_1"}
        store = CheckpointStore(self.path, self.requests, citations)
        row = {"case_id":"case_0", "deletion_status":"COMPLETE", "steps_json":json.dumps([{
            "step_no":1,"phase":"VERIFICATION","system_id":"s","resource_id":"r",
            "operation":"VERIFY_RECORD_ABSENT","state":"ABSENT","evidence_id":"source_1"}])}
        with self.assertRaisesRegex(ValueError, "not recorded as retrieved"):
            store.commit(row, [])
        store.commit(row, ["source_1"])
        changed = list(self.requests)
        changed[0] = {**changed[0], "audit_at":"2025-01-02T00:00:00Z"}
        with self.assertRaisesRegex(ValueError, "does not match this input"):
            CheckpointStore(self.path, changed, citations)
        with self.assertRaisesRegex(ValueError, "evidence context"):
            CheckpointStore(self.path, self.requests, citations, "different_dataset")

    def test_all_518_input_requests_load_with_unique_ids_and_valid_contract(self):
        requests = load_requests(ROOT / "dataset/deletion_requests.csv")
        self.assertEqual(len(requests), 518)
        self.assertEqual(len({row["case_id"] for row in requests}), 518)
        self.assertEqual(requests, sorted(requests, key=lambda row: row["case_id"]))

    def test_final_csv_writer_requires_all_rows_and_serializes_518_sorted(self):
        requests = [{"case_id": f"case_{index:03d}", "audit_at": "2025-01-01T00:00:00Z"} for index in range(518)]
        rows = [{"case_id": row["case_id"], "deletion_status": "NOT_APPLICABLE", "steps_json": "[]"}
                for row in reversed(requests)]
        output = self.root / "output.csv"
        write_final_output(output, requests, rows, lambda audit_at: set())
        with output.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, ["case_id", "deletion_status", "steps_json"])
            serialized = list(reader)
        self.assertEqual(len(serialized), 518)
        self.assertEqual([row["case_id"] for row in serialized], sorted(row["case_id"] for row in requests))
        previous = output.read_bytes()
        with self.assertRaisesRegex(ValueError, "exactly one row"):
            write_final_output(output, requests, rows[:-1], lambda audit_at: set())
        self.assertEqual(output.read_bytes(), previous)

    def test_run_paths_cannot_overwrite_dataset_or_collide(self):
        dataset = ROOT / "dataset"
        with self.assertRaisesRegex(ValueError, "outside the participant dataset"):
            _validate_run_paths(dataset, ROOT / "dataset/deletion_requests.csv", dataset / "output.csv",
                                ROOT / "pred.checkpoint.csv", ROOT / "pred.run.json")
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            _validate_run_paths(dataset, ROOT / "dataset/deletion_requests.csv", ROOT / "pred.csv",
                                ROOT / "pred.csv", ROOT / "pred.run.json")

    def test_request_csv_rejects_extra_cells(self):
        path = self.root / "extra.csv"
        path.write_text("case_id,email,requested_at,reported_status,reported_at,audit_at,request_text\n"
                        "c,e,2025-01-01T00:00:00Z,COMPLETE,2025-01-01T00:00:00Z,2025-01-01T00:00:00Z,text,extra\n",
                        encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exact input column count"):
            load_requests(path)

    def test_failed_case_records_attempts_and_redacted_error(self):
        class Backend:
            generate_request_attempts = 0
            transient_retries = 0
        backend = Backend()
        class Agent:
            def run(self, request):
                backend.generate_request_attempts += 2
                raise AgentError("request failed api_key=secret-value")
        store = CheckpointStore(self.path, self.requests, self.citations)
        state = {"generation_attempts": 0, "transient_retries": 0, "errors": []}
        with self.assertRaises(AgentError):
            process_batch(self.requests, Agent(), backend, store, state, self.state_path, limit=1)
        self.assertEqual(state["generation_attempts"], 2)
        saved = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.assertNotIn("secret-value", json.dumps(saved))
        self.assertEqual(saved["errors"][0]["generation_attempts"], 2)

    def test_validation_failure_is_checkpointed_and_batch_continues_to_next_cases(self):
        class Backend:
            generate_request_attempts = 0
            transient_retries = 0
        backend = Backend()
        class Agent:
            last_tool_trace = []
            last_output_diagnostics = []
            last_usage = {"agent_turns": 1, "input_tokens": 10, "output_tokens": 2,
                          "input_tokens_complete": True, "output_tokens_complete": True}
            def run(self, request):
                backend.generate_request_attempts += 1
                if request["case_id"] == "case_0":
                    raise AgentOutputError("bad decision")
                return SimpleNamespace(case_id=request["case_id"], deletion_status="NOT_APPLICABLE",
                    steps_json=[], retrieved_source_ids=(), api_calls=1, input_tokens=10,
                    output_tokens=2, tool_calls=0)
        store = CheckpointStore(self.path, self.requests, self.citations)
        state = {"generation_attempts": 0, "transient_retries": 0, "agent_turns": 0,
                 "input_tokens": 0, "output_tokens": 0, "input_tokens_complete": True,
                 "output_tokens_complete": True, "errors": [], "completed_cases": []}
        rows = process_batch(self.requests, Agent(), backend, store, state, self.state_path)
        self.assertEqual([row["case_id"] for row in rows], ["case_1", "case_2"])
        self.assertEqual([row["case_id"] for row in state["completed_cases"]], ["case_1", "case_2"])
        self.assertEqual(state["errors"][0]["case_id"], "case_0")
        self.assertEqual(state["errors"][0]["error"], "bad decision")
        self.assertEqual(state["generation_attempts"], 3)

    def test_partial_evaluation_reports_coverage_and_scored_case_accuracy(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as temp:
            path = Path(temp) / "partial.csv"
            with (ROOT / "dataset/sample_output.csv").open(encoding="utf-8-sig", newline="") as source:
                rows = list(csv.DictReader(source))[:2]
            with path.open("w", encoding="utf-8", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=["case_id", "deletion_status", "steps_json"])
                writer.writeheader(); writer.writerows(rows)
            metrics = eval_module.compare_samples(path, allow_partial=True)
            self.assertEqual(metrics["scored_cases"], 2)
            self.assertEqual(metrics["coverage"], 2 / 32)
            self.assertEqual(metrics["exact_match_accuracy"], 1)
            self.assertEqual(len(metrics["missing_cases"]), 30)
            full_denominator = eval_module.compare_samples(path)
            self.assertEqual(full_denominator["exact_match_accuracy"], 2 / 32)

    def test_preflight_counts_are_added_once_and_smoke_is_separate(self):
        state = {"model_lookup_requests": 1, "smoke_function_call_attempts": 0}
        backend = SimpleNamespace(model_lookup_attempts=1, smoke_function_call_attempts=1)
        _record_preflight_counts(state, backend)
        self.assertEqual(state["model_lookup_requests"], 2)
        self.assertEqual(state["smoke_function_call_attempts"], 1)
    def test_error_text_redacts_common_credential_forms(self):
        text = _safe_error("429 api_key=abcdef authorization: Bearer token-value")
        self.assertNotIn("abcdef", text)
        self.assertNotIn("token-value", text)

if __name__ == "__main__":
    unittest.main()
