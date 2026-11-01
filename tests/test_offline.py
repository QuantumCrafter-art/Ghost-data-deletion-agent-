import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
DATASET = ROOT / "dataset"
from ghost_audit.offline import OfflinePredictor, _category, _system, _lookup_result_matches, _pii_present, _rules_for_category, _subject_resource_link
from ghost_audit.policy import DeletionPolicy
from ghost_audit.intent import explicitly_requests_other_action_only

class OfflineSafetyTests(unittest.TestCase):
    def setUp(self):
        self.policy = DeletionPolicy(DATASET)

    def test_service_and_system_names_map_to_grounded_categories(self):
        cases = [
            ({"service": "account_service"}, "USER_PROFILE"),
            ({"system": "support_platform"}, "SUPPORT_MESSAGE"),
            ({"service": "backup_service"}, "BACKUP_ARCHIVE"),
        ]
        for record, category in cases:
            with self.subTest(record=record):
                self.assertEqual(_category(record, "", self.policy), category)
                self.assertIsNotNone(_system(record, category))

    def test_marketing_opt_in_fields_override_host_service_category(self):
        self.assertEqual(_category({"service": "analytics_platform", "email_opt_in": "true"}, "", self.policy), "MARKETING_PROFILE")

    def test_corpus_collection_names_map_to_policy_categories(self):
        cases = {
            "customer_analytics_reports": "ANALYTICS_PROFILE",
            "support_conversations": "SUPPORT_MESSAGE",
            "ticket_attachments": "SUPPORT_ATTACHMENT",
            "connector_configurations": "ACTIVE_DATA_FLOW",
            "backup_catalog": "BACKUP_ARCHIVE",
            "customer_export_jobs": "USER_EXPORT",
            "retained_order_records": "LEGAL_HOLD_RECORD",
        }
        for collection, category in cases.items():
            with self.subTest(collection=collection):
                self.assertEqual(_category({"collection_name": collection}, "", self.policy), category)

    def test_contract_requires_active_flow_shutdown_even_without_csv_rule(self):
        rules = _rules_for_category(self.policy, "ACTIVE_DATA_FLOW")
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0].required_action, "DISABLE_DATA_FLOW")
        self.assertEqual(rules[0].condition.casefold(), "always")
        self.assertEqual(_rules_for_category(self.policy, "UNKNOWN_CATEGORY"), ())

    def test_explicit_activity_resource_link_is_only_a_candidate(self):
        aliases = [("email", "person@example.test"), ("user_id", "user_42")]
        linked = {"details": "Data catalog linked resource_9 to person@example.test and user_42"}
        unrelated = {"details": "Lookup started for person@example.test"}
        collision = {"details": "resource linked to xperson@example.test and user_420"}
        self.assertTrue(_subject_resource_link(linked, aliases))
        self.assertFalse(_subject_resource_link(unrelated, aliases))
        self.assertFalse(_subject_resource_link(collision, aliases))

    def test_completed_lookup_result_links_subject_but_query_key_is_not_pii(self):
        row = {"event": "profile_lookup_completed", "lookup_key": "old@example.test",
               "rows": "0", "http_status": "404", "email": "null"}
        self.assertTrue(_lookup_result_matches(row, [("email", "old@example.test")]))
        self.assertFalse(_pii_present(row))
        start_only = {"event": "profile_lookup_started", "lookup_key": "old@example.test"}
        self.assertFalse(_lookup_result_matches(start_only, [("email", "old@example.test")]))

    def test_evidence_gap_never_becomes_not_applicable(self):
        predictor = OfflinePredictor(DATASET)
        request = {
            "case_id": "offline-gap-test", "email": "missing.identity@example.invalid",
            "audit_at": "2026-09-03T11:30:00Z", "request_text": "Please delete my account and data."
        }
        with self.assertRaises(ValueError):
            predictor.predict(request)

    def test_explicit_non_deletion_only_requests_are_not_applicable(self):
        examples = [
            "I would like to download my account information. Please keep the account as is.",
            "Can you correct my surname without removing any of my records?",
            "Please remove me from marketing lists only.",
        ]
        for text in examples:
            with self.subTest(text=text):
                self.assertTrue(explicitly_requests_other_action_only(text))

    def test_explicit_refusal_is_the_only_no_evidence_not_applicable_path(self):
        predictor = OfflinePredictor(DATASET)
        request = {
            "case_id": "offline-refusal-test", "email": "missing.identity@example.invalid",
            "audit_at": "2026-09-03T11:30:00Z", "request_text": "Do not delete my account."
        }
        result, sources, _ = predictor.predict(request)
        self.assertEqual(result["deletion_status"], "NOT_APPLICABLE")
        self.assertEqual(result["steps_json"], [])
        self.assertEqual(sources, set())

if __name__ == "__main__": unittest.main()




