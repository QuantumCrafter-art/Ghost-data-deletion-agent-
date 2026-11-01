import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from ghost_audit.agent import AgentOutputError, DeletionAuditAgent, ModelTurn, ToolCall
from ghost_audit.identity import IdentityResolver
from ghost_audit.index import EvidenceEngine
from ghost_audit.policy import DeletionPolicy
from test_evidence_engine import make_dataset

class FakeBackend:
    model_name = "mock"
    def __init__(self, turns): self.turns, self.calls = list(turns), []
    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if not self.turns: raise AssertionError("unexpected model call")
        return self.turns.pop(0)

class SnapshotRecoveryTests(unittest.TestCase):
    def test_case01_absence_is_discovered_through_report_activity_and_sql_snapshot(self):
        data = ROOT / "dataset"
        with (data / "sample_deletion_requests.csv").open(encoding="utf-8-sig", newline="") as f:
            request = next(row for row in csv.DictReader(f) if row["case_id"] == "case_01")
        with (data / "sample_output.csv").open(encoding="utf-8-sig", newline="") as f:
            expected = next(row for row in csv.DictReader(f) if row["case_id"] == "case_01")
        reference = json.loads(expected["steps_json"])
        turns = [
            ModelTurn(function_calls=(ToolCall("resolve_identity", {}, "t1"),)),
            ModelTurn(function_calls=(ToolCall("search_reported_activity", {}, "t2"),)),
            ModelTurn(function_calls=(ToolCall("list_system_snapshots", {"system_id":"account_service"}, "t3"),)),
            ModelTurn(function_calls=(ToolCall("inspect_sql_snapshot", {"evidence_id":"evidence_54", "system_id":"account_service", "resource_id":"account_001"}, "t4"),)),
            ModelTurn(text=json.dumps({"case_id":expected["case_id"], "deletion_status":expected["deletion_status"], "steps_json":reference})),
        ]
        engine = EvidenceEngine(data)
        agent = DeletionAuditAgent(engine, IdentityResolver(data), DeletionPolicy(data), FakeBackend(turns))
        result = agent.run(request)
        self.assertEqual(result.deletion_status, expected["deletion_status"])
        self.assertEqual(result.steps_json, reference)
        self.assertEqual(set(result.retrieved_source_ids), {"activity_3342", "evidence_54"})
        activity = engine.search_reported_activity(request["reported_at"], request["audit_at"])[0][1]
        state = agent._initialize_state(request)
        agent._observe_resource_ids(state, activity)
        self.assertIn(("account_service", "account_001"), state.observed_resource_pairs)
        self.assertNotIn(("deletion_worker", "account_001"), state.observed_resource_pairs)
        self.assertEqual([x["tool"] for x in result.tool_trace], ["resolve_identity", "search_reported_activity", "list_system_snapshots", "inspect_sql_snapshot"])
        self.assertEqual(result.tool_trace[-1]["source_ids"], ["evidence_54"])

    def test_reported_complete_final_is_gated_until_snapshot_inspection(self):
        data = ROOT / "dataset"
        with (data / "sample_deletion_requests.csv").open(encoding="utf-8-sig", newline="") as f:
            request = next(row for row in csv.DictReader(f) if row["case_id"] == "case_01")
        with (data / "sample_output.csv").open(encoding="utf-8-sig", newline="") as f:
            expected = next(row for row in csv.DictReader(f) if row["case_id"] == "case_01")
        correct = {"case_id":expected["case_id"], "deletion_status":expected["deletion_status"], "steps_json":json.loads(expected["steps_json"])}
        premature = json.dumps({"case_id":request["case_id"], "deletion_status":"NOT_APPLICABLE", "steps_json":[]})
        backend = FakeBackend([
            ModelTurn(function_calls=(ToolCall("resolve_identity", {}, "1"),)),
            ModelTurn(function_calls=(ToolCall("search_evidence", {"identifier":request["email"]}, "2"),)),
            ModelTurn(function_calls=(ToolCall("search_evidence", {"identifier":"user_01"}, "3"),)),
            ModelTurn(text=premature),
            ModelTurn(function_calls=(ToolCall("search_reported_activity", {}, "4"),)),
            ModelTurn(text=premature),
            ModelTurn(function_calls=(ToolCall("list_system_snapshots", {"system_id":"account_service"}, "5"),)),
            ModelTurn(text=premature),
            ModelTurn(function_calls=(ToolCall("inspect_sql_snapshot", {"evidence_id":"evidence_54", "system_id":"account_service", "resource_id":"account_001"}, "6"),)),
            ModelTurn(text=json.dumps(correct)),
        ])
        engine = EvidenceEngine(data)
        result = DeletionAuditAgent(engine, IdentityResolver(data), DeletionPolicy(data), backend).run(request)
        self.assertEqual(result.deletion_status, expected["deletion_status"])
        self.assertEqual(result.steps_json, json.loads(expected["steps_json"]))
        self.assertEqual(result.corrections, 0)
        self.assertEqual([t["tool"] for t in result.tool_trace], ["resolve_identity", "search_evidence", "search_evidence", "search_reported_activity", "list_system_snapshots", "inspect_sql_snapshot"])

    def test_sql_snapshot_state_blocks_unsupported_final_claim(self):
        data = ROOT / "dataset"
        with (data / "sample_deletion_requests.csv").open(encoding="utf-8-sig", newline="") as f:
            request = next(row for row in csv.DictReader(f) if row["case_id"] == "case_01")
        with (data / "sample_output.csv").open(encoding="utf-8-sig", newline="") as f:
            expected = next(row for row in csv.DictReader(f) if row["case_id"] == "case_01")
        correct = {"case_id":expected["case_id"], "deletion_status":expected["deletion_status"], "steps_json":json.loads(expected["steps_json"])}
        wrong = json.loads(json.dumps(correct))
        wrong["steps_json"][0].update(operation="VERIFY_RECORD_PRESENT", state="PRESENT")
        calls = [ToolCall("resolve_identity", {}, "1"), ToolCall("search_reported_activity", {}, "2"),
                 ToolCall("list_system_snapshots", {"system_id":"account_service"}, "3"),
                 ToolCall("inspect_sql_snapshot", {"evidence_id":"evidence_54", "system_id":"account_service", "resource_id":"account_001"}, "4")]
        turns = [ModelTurn(function_calls=(call,)) for call in calls]
        turns.extend([ModelTurn(text=json.dumps(wrong)), ModelTurn(text=json.dumps(correct))])
        backend = FakeBackend(turns)
        result = DeletionAuditAgent(EvidenceEngine(data), IdentityResolver(data), DeletionPolicy(data), backend).run(request)
        self.assertEqual(result.steps_json, correct["steps_json"])
        self.assertEqual(result.corrections, 1)
        self.assertIn("supports resource state ABSENT", backend.calls[5]["history"][-1].text)

    def test_partially_complete_case_finds_target_activity_five_minutes_late(self):
        data = ROOT / "dataset"
        with (data / "sample_deletion_requests.csv").open(encoding="utf-8-sig", newline="") as f:
            request = next(row for row in csv.DictReader(f) if row["case_id"] == "case_02")
        with (data / "sample_output.csv").open(encoding="utf-8-sig", newline="") as f:
            expected = next(row for row in csv.DictReader(f) if row["case_id"] == "case_02")
        activity = EvidenceEngine(data).search_reported_activity(request["reported_at"], request["audit_at"])
        self.assertEqual([(round(delta), hit.source_id) for delta, hit in activity], [(303, "activity_3376")])
        decision = {"case_id":expected["case_id"], "deletion_status":expected["deletion_status"], "steps_json":json.loads(expected["steps_json"])}
        backend = FakeBackend([
            ModelTurn(function_calls=(ToolCall("resolve_identity", {}, "1"),)),
            ModelTurn(function_calls=(ToolCall("search_evidence", {"identifier":request["email"]}, "2"),)),
            ModelTurn(text=json.dumps({"case_id":request["case_id"], "deletion_status":"NOT_APPLICABLE", "steps_json":[]})),
            ModelTurn(function_calls=(ToolCall("search_reported_activity", {}, "3"),)),
            ModelTurn(function_calls=(ToolCall("list_system_snapshots", {"system_id":"account_service"}, "4"),)),
            ModelTurn(function_calls=(ToolCall("inspect_sql_snapshot", {"evidence_id":"evidence_54", "system_id":"account_service", "resource_id":"account_002"}, "5"),)),
            ModelTurn(text=json.dumps(decision)),
        ])
        result = DeletionAuditAgent(EvidenceEngine(data), IdentityResolver(data), DeletionPolicy(data), backend).run(request)
        self.assertEqual(result.deletion_status, expected["deletion_status"])
        self.assertEqual(result.steps_json, json.loads(expected["steps_json"]))
        self.assertEqual(set(result.retrieved_source_ids), {"activity_3376", "evidence_54"})

    def test_case03_service_fields_and_older_activity_retrieve_subject_evidence_not_decoy(self):
        data = ROOT / "dataset"
        with (data / "sample_deletion_requests.csv").open(encoding="utf-8-sig", newline="") as f:
            request = next(row for row in csv.DictReader(f) if row["case_id"] == "case_03")
        engine = EvidenceEngine(data)
        identities = IdentityResolver(data)
        agent = DeletionAuditAgent(engine, identities, DeletionPolicy(data), FakeBackend([]))
        state = agent._initialize_state(request)
        agent._resolve_identity(state, {})
        self.assertEqual(state.identity_resolution_status, "RESOLVED")
        self.assertEqual(agent._search(state, {"identifier": request["email"], "limit": 12})["count"], 1)
        self.assertEqual(agent._search(state, {"identifier": "user_43", "limit": 12})["count"], 2)
        self.assertIn(("account_service", "account_7566"), state.subject_resource_pairs)
        self.assertIn(("security_service", "audit_record_2016"), state.subject_resource_pairs)
        self.assertIn("inspect_resource", agent._required_investigation(state))
        agent._inspect_resource(state, {"system_id": "account_service", "resource_id": "account_7566"})
        agent._inspect_resource(state, {"system_id": "security_service", "resource_id": "audit_record_2016"})

        activity = agent._dispatch(state, ToolCall("search_reported_activity", {}, "activity"))
        activity_ids = {item["source_id"] for item in activity["results"]}
        self.assertTrue({"activity_3851", "activity_3852", "activity_3856"}.issubset(activity_ids))
        self.assertEqual(next(item for item in activity["results"] if item["source_id"] == "activity_3852")["proximity_seconds"], 1724)

        analytics = agent._dispatch(state, ToolCall("inspect_resource", {
            "system_id": "analytics_platform", "resource_id": "analytics_9406"}, "analytics"))
        analytics_text = "\n".join(item["content"] for item in analytics["results"])
        self.assertIn("email=null", analytics_text)
        self.assertEqual(analytics["linkage"], "TEMPORAL_CANDIDATE_ONLY")
        self.assertIn("evidence_94", state.retrieved_source_ids)

        backup = agent._dispatch(state, ToolCall("inspect_resource", {
            "system_id": "backup_service", "resource_id": "backup_6266"}, "backup"))
        backup_text = "\n".join(item["content"] for item in backup["results"])
        self.assertIn("customer_reference=user_42", backup_text)
        self.assertEqual(backup["linkage"], "TEMPORAL_CANDIDATE_ONLY")
        self.assertIn("activity_3856", {item["source_id"] for item in backup["results"]})

    def test_sql_absence_inspector_supports_multiline_create_and_committed_transaction(self):
        engine = EvidenceEngine(ROOT / "dataset")
        assessment = engine.inspect_sql_resource_snapshot("evidence_54", "account_001", "2026-09-03T11:30:00Z")
        self.assertEqual({table["resource_state"] for table in assessment["tables"]}, {"ABSENT"})
        self.assertTrue(all(table["transaction_committed"] for table in assessment["tables"]))
        self.assertEqual(assessment["content_trust"], "UNTRUSTED_EVIDENCE_DATA")

    def test_unparseable_snapshot_write_prevents_absence_claim(self):
        tmp = tempfile.TemporaryDirectory(dir=ROOT.parent)
        try:
            data = make_dataset(Path(tmp.name), [("sql_evidence", "db/accounts.sql", "2025-01-01T00:00:00Z",
                "BEGIN;\nCREATE TABLE accounts (resource_id TEXT);\nINSERT INTO accounts SELECT resource_id FROM staging;\nCOMMIT;")])
            result = EvidenceEngine(data).inspect_sql_resource_snapshot("sql_evidence", "account_1", "2025-02-01T00:00:00Z")
            self.assertEqual(result["tables"][0]["resource_state"], "UNKNOWN")
        finally:
            tmp.cleanup()

    def test_snapshot_listing_and_inspection_obey_cutoff_and_observed_pair_gate(self):
        tmp = tempfile.TemporaryDirectory(dir=ROOT.parent)
        try:
            data = make_dataset(Path(tmp.name), [("sql_evidence", "db/accounts.sql", "2025-02-01T00:00:00Z", "BEGIN;\nCREATE TABLE account_table (\nresource_id TEXT\n);\nCOMMIT;")])
            engine = EvidenceEngine(data)
            self.assertEqual(engine.system_sql_snapshots("account_service", "2025-01-31T00:00:00Z"), [])
            with self.assertRaisesRegex(ValueError, "after audit_at"):
                engine.inspect_sql_resource_snapshot("sql_evidence", "account_1", "2025-01-31T00:00:00Z")
        finally:
            tmp.cleanup()
        data = ROOT / "dataset"
        with (data / "sample_deletion_requests.csv").open(encoding="utf-8-sig", newline="") as f:
            request = next(row for row in csv.DictReader(f) if row["case_id"] == "case_01")
        agent = DeletionAuditAgent(EvidenceEngine(data), IdentityResolver(data), DeletionPolicy(data), FakeBackend([]))
        state = agent._initialize_state(request)
        with self.assertRaisesRegex(ValueError, "system must be an activity target"):
            agent._dispatch(state, ToolCall("list_system_snapshots", {"system_id":"account_service"}, "t"))

    def test_not_applicable_requires_explicit_decline_not_retrieval_failure(self):
        tmp = tempfile.TemporaryDirectory(dir=ROOT.parent)
        try:
            data = make_dataset(Path(tmp.name), [], identities=[{"user_id":"u1", "name":"P", "email":"person@example.test", "phone":"", "email_hash":"", "updated_at":"2025-01-01T00:00:00Z"}])
            request={"case_id":"case_test", "email":"person@example.test", "audit_at":"2025-01-03T00:00:00Z", "request_text":"Please delete my account."}
            na = json.dumps({"case_id":"case_test", "deletion_status":"NOT_APPLICABLE", "steps_json":[]})
            agent=DeletionAuditAgent(EvidenceEngine(data), IdentityResolver(data), DeletionPolicy(data), FakeBackend([
                ModelTurn(function_calls=(ToolCall("resolve_identity", {}, "t1"),)), ModelTurn(text=na), ModelTurn(text=na)]))
            with self.assertRaisesRegex(AgentOutputError, "NOT_APPLICABLE requires affirmative request text"):
                agent.run(request)
        finally:
            tmp.cleanup()

    def test_explicit_cancellation_allows_empty_not_applicable_decision(self):
        tmp = tempfile.TemporaryDirectory(dir=ROOT.parent)
        try:
            data = make_dataset(Path(tmp.name), [], identities=[{"user_id":"u1", "name":"P", "email":"person@example.test", "phone":"", "email_hash":"", "updated_at":"2025-01-01T00:00:00Z"}])
            request={"case_id":"case_test", "email":"person@example.test", "audit_at":"2025-01-03T00:00:00Z", "request_text":"I cancel my deletion request."}
            na = json.dumps({"case_id":"case_test", "deletion_status":"NOT_APPLICABLE", "steps_json":[]})
            agent=DeletionAuditAgent(EvidenceEngine(data), IdentityResolver(data), DeletionPolicy(data), FakeBackend([
                ModelTurn(function_calls=(ToolCall("resolve_identity", {}, "t1"),)), ModelTurn(text=na)]))
            self.assertEqual(agent.run(request).deletion_status, "NOT_APPLICABLE")
        finally:
            tmp.cleanup()

if __name__ == "__main__":
    unittest.main()
