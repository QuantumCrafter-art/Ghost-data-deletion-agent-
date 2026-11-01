import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
from ghost_audit.agent import (AgentConfig, AgentError, AgentOutputError, AgentSafetyError, DeletionAuditAgent,
                               ModelTurn, ToolCall, ToolReply, UserMessage, _DECISION_SCHEMA)
from ghost_audit.identity import IdentityResolver
from ghost_audit.index import EvidenceEngine
from ghost_audit.policy import DeletionPolicy
from ghost_audit.gemini import GeminiBackend, _response_function_calls, _response_text
from test_evidence_engine import make_dataset

class FakeBackend:
    model_name = "mock-gemini-for-tests"
    def __init__(self, turns): self.turns, self.calls = list(turns), []
    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if not self.turns: raise AssertionError("unexpected model call")
        return self.turns.pop(0)

def fixture(root):
    email = "person@example.test"
    data = make_dataset(root, [("evidence_subject", "subject.json", "2025-01-02T00:00:00Z",
        json.dumps({"email": email, "user_id": "user_7", "system_id": "account_service", "resource_id": "account_7"}))],
        identities=[{"user_id":"user_7","name":"P","email":email,"phone":"5550107","email_hash":"hash7","updated_at":"2025-01-01T00:00:00Z"}],
        policies=[{"rule_id":"r1","data_category":"account","condition":"always","required_action":"DELETE"}])
    request={"case_id":"case_test","email":email,"audit_at":"2025-01-03T00:00:00Z","request_text":"Please delete my account."}
    return data, request

def complete(case="case_test", evidence="evidence_subject"):
    return json.dumps({"case_id":case,"deletion_status":"COMPLETE","steps_json":[{
        "step_no":1,"phase":"VERIFICATION","system_id":"account_service","resource_id":"account_7",
        "operation":"VERIFY_RECORD_ABSENT","state":"ABSENT","evidence_id":evidence}]})

class GeminiAgentTests(unittest.TestCase):
    def test_function_response_uses_sdk_model_with_call_id(self):
        from types import SimpleNamespace
        class FakeContent:
            def __init__(self, **kwargs): self.__dict__.update(kwargs)
        class FakePart:
            def __init__(self, **kwargs): self.__dict__.update(kwargs)
            @staticmethod
            def from_text(text): return FakePart(text=text)
        class FakeResponse:
            def __init__(self, **kwargs): self.__dict__.update(kwargs)
        fake_types=SimpleNamespace(Content=FakeContent,Part=FakePart,FunctionResponse=FakeResponse)
        backend=GeminiBackend(client=None,types=fake_types,model_name="mock")
        contents=backend._contents([ToolReply("search_evidence","call_9",{"ok":True})])
        self.assertEqual(contents[0].role,"user")
        self.assertEqual(contents[0].parts[0].function_response.id,"call_9")
    def test_image_function_response_preserves_media_type_filename_and_bytes(self):
        from types import SimpleNamespace
        from ghost_audit.index import ImageEvidence
        class Factory:
            def __init__(self, **kwargs): self.__dict__.update(kwargs)
            @staticmethod
            def from_text(text): return Factory(text=text)
        fake_types=SimpleNamespace(Content=Factory,Part=Factory,FunctionResponse=Factory,
            FunctionResponsePart=Factory,FunctionResponseBlob=Factory)
        backend=GeminiBackend(client=None,types=fake_types,model_name="mock")
        image=ImageEvidence("img_1","2025-01-01T00:00:00Z","image/jpeg",b"jpeg-bytes","support.jpg")
        contents=backend._contents([ToolReply("inspect_image","call_img",{"evidence_id":"img_1"},image)])
        response=contents[0].parts[0].function_response
        self.assertEqual(response.parts[0].inline_data.mime_type,"image/jpeg")
        self.assertEqual(response.parts[0].inline_data.display_name,"support.jpg")
        self.assertEqual(response.parts[0].inline_data.data,b"jpeg-bytes")

    def test_agent_generation_uses_bounded_output_budget(self):
        from types import SimpleNamespace
        class Factory:
            def __init__(self, **kwargs): self.__dict__.update(kwargs)
        class PartFactory(Factory):
            @staticmethod
            def from_text(text): return PartFactory(text=text)
        class Models:
            def __init__(self): self.kwargs = None
            def generate_content(self, **kwargs):
                self.kwargs = kwargs
                content = SimpleNamespace(parts=[])
                return SimpleNamespace(candidates=[SimpleNamespace(content=content)], text='{"case_id":"x"}')
        models = Models()
        fake_types = SimpleNamespace(GenerateContentConfig=Factory, FunctionDeclaration=Factory, Tool=Factory,
            ToolConfig=Factory, FunctionCallingConfig=Factory, Content=Factory, Part=PartFactory)
        backend = GeminiBackend(SimpleNamespace(models=models), fake_types, "mock")
        backend.generate(history=[UserMessage("case")], system_instruction="system", tools=[],
                         allowed_tool_names=set(), response_schema={})
        self.assertEqual(models.kwargs["config"].max_output_tokens, 2048)

    def test_text_smoke_uses_minimal_generation_config_and_measures_latency(self):
        from types import SimpleNamespace
        class Factory:
            def __init__(self, **kwargs): self.__dict__.update(kwargs)
        class Models:
            def __init__(self): self.kwargs = None
            def generate_content(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(text="OK")
        models = Models()
        fake_types = SimpleNamespace(GenerateContentConfig=Factory)
        backend = GeminiBackend(SimpleNamespace(models=models), fake_types, "mock")
        self.assertEqual(backend.smoke_text_generation(), "OK")
        self.assertEqual(models.kwargs["config"].max_output_tokens, 16)
        self.assertEqual(backend.generate_request_attempts, 1)
        self.assertIn("text", backend.last_smoke_latencies_ms)

    def test_function_smoke_measures_latency_and_counts_one_call(self):
        from types import SimpleNamespace
        class Factory:
            def __init__(self, **kwargs): self.__dict__.update(kwargs)
        class Models:
            def generate_content(self, **kwargs):
                return SimpleNamespace(function_calls=[SimpleNamespace(name="confirm_tool", args={"marker":"codex-smoke"})])
        fake_types = SimpleNamespace(FunctionDeclaration=Factory, Tool=Factory,
            GenerateContentConfig=Factory, ToolConfig=Factory, FunctionCallingConfig=Factory)
        backend = GeminiBackend(SimpleNamespace(models=Models()), fake_types, "mock")
        self.assertTrue(backend.smoke_function_call())
        self.assertEqual(backend.smoke_function_call_attempts, 1)
        self.assertEqual(backend.generate_request_attempts, 1)
        self.assertIn("function_call", backend.last_smoke_latencies_ms)
    def test_sdk_function_calls_are_preferred_for_mixed_response(self):
        from types import SimpleNamespace
        call=SimpleNamespace(name="resolve_identity",args={},id="call1")
        response=SimpleNamespace(function_calls=[call])
        content=SimpleNamespace(parts=[])
        self.assertEqual(_response_function_calls(response,content),[call])
    def test_function_call_turn_does_not_read_text_property(self):
        class FunctionResponse:
            @property
            def text(self): raise AssertionError("SDK text property must not be read for function-call turns")
        self.assertIsNone(_response_text(FunctionResponse(), True))
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=REPO.parent)
        self.data,self.request=fixture(Path(self.tmp.name))
        self.engine=EvidenceEngine(self.data)
        self.identity=IdentityResolver(self.data)
        self.policy=DeletionPolicy(self.data)
    def tearDown(self): self.tmp.cleanup()
    def agent(self, turns, config=AgentConfig()):
        self.backend=FakeBackend(turns)
        return DeletionAuditAgent(self.engine,self.identity,self.policy,self.backend,config)
    def test_tool_schemas_omit_fields_rejected_by_gemini_function_declarations(self):
        declarations=self.agent([])._tool_declarations(self.policy.categories())
        def walk(value):
            if isinstance(value,dict):
                self.assertNotIn("additionalProperties",value)
                for child in value.values(): walk(child)
            elif isinstance(value,list):
                for child in value: walk(child)
        walk(declarations)
        walk(_DECISION_SCHEMA)
    def test_search_requires_unique_identity_resolution_first(self):
        agent = self.agent([])
        state = agent._initialize_state(self.request)
        with self.assertRaisesRegex(ValueError, "resolve a unique identity"):
            agent._search(state, {"identifier": "person@example.test"})

    def test_resource_system_and_id_must_be_observed_as_same_pair(self):
        from types import SimpleNamespace
        agent = self.agent([])
        state = agent._initialize_state(self.request)
        agent._observe_resource_ids(state, SimpleNamespace(text=json.dumps({"system_id":"service_a", "resource_id":"resource_a"}), source_id="one"))
        agent._observe_resource_ids(state, SimpleNamespace(text=json.dumps({"system_id":"service_b", "resource_id":"resource_b"}), source_id="two"))
        with self.assertRaisesRegex(ValueError, "co-occur"):
            agent._inspect_resource(state, {"system_id":"service_a", "resource_id":"resource_b"})

    def test_always_policy_action_is_enforced_on_finding_resource(self):
        from types import SimpleNamespace
        agent = self.agent([])
        state = agent._initialize_state(self.request)
        hit = SimpleNamespace(text=json.dumps({"system_id":"account_service", "resource_id":"account_7", "data_category":"account"}), source_id="evidence_subject")
        agent._observe_resource_ids(state, hit)
        rule = agent.policy.lookup("account")[0]
        state.looked_up_policy_categories.add("account")
        state.policy_rules_by_category["account"] = (rule,)
        steps = [
            {"step_no":1,"phase":"FINDING","system_id":"account_service","resource_id":"account_7","operation":"REPORT_REMAINING_RECORD","state":"PRESENT","evidence_id":"evidence_subject"},
            {"step_no":2,"phase":"REMEDIATION","system_id":"account_service","resource_id":"account_7","operation":"REDACT_RECORD","state":"REDACTED","evidence_id":""},
            {"step_no":3,"phase":"VERIFICATION","system_id":"account_service","resource_id":"account_7","operation":"VERIFY_RECORD_REDACTED","state":"REDACTED","evidence_id":""},
        ]
        with self.assertRaisesRegex(Exception, "Always policy"):
            agent._validate_policy_actions(state, "INCOMPLETE", steps)
    def test_final_decision_requires_policy_lookup_for_observed_category(self):
        policy_data = make_dataset(Path(self.tmp.name) / "policy_case", [
            ("evidence_subject", "subject.json", "2025-01-02T00:00:00Z",
             json.dumps({"email":"person@example.test", "user_id":"user_7", "system_id":"account_service",
                         "resource_id":"account_7", "data_category":"account"}))
        ], identities=[{"user_id":"user_7","name":"P","email":"person@example.test","phone":"5550107",
                        "email_hash":"hash7","updated_at":"2025-01-01T00:00:00Z"}],
           policies=[{"rule_id":"r1","data_category":"account","condition":"always","required_action":"DELETE"}])
        self.engine = EvidenceEngine(policy_data)
        self.identity = IdentityResolver(policy_data)
        self.policy = DeletionPolicy(policy_data)
        agent = self.agent([
            ModelTurn(function_calls=(ToolCall("resolve_identity", {}, "1"),)),
            ModelTurn(function_calls=(ToolCall("search_evidence", {"identifier":"person@example.test"}, "2"),)),
            ModelTurn(function_calls=(ToolCall("inspect_resource", {"system_id":"account_service", "resource_id":"account_7"}, "2a"),)),
            ModelTurn(text=complete()),
            ModelTurn(function_calls=(ToolCall("lookup_policy", {"data_category":"account"}, "3"),)),
            ModelTurn(text=complete()),
        ])
        outcome = agent.run(self.request)
        self.assertEqual(outcome.corrections, 1)
        self.assertEqual(outcome.tool_calls, 4)
        self.assertIn("evidence_subject", outcome.retrieved_source_ids)

    def test_applicable_conditional_policy_action_is_enforced(self):
        from types import SimpleNamespace
        from ghost_audit.policy import PolicyAssessment, PolicyRule
        agent = self.agent([])
        state = agent._initialize_state(self.request)
        hit = SimpleNamespace(text=json.dumps({"system_id":"account_service", "resource_id":"account_7", "data_category":"account"}), source_id="evidence_subject")
        agent._observe_resource_ids(state, hit)
        rule = PolicyRule("conditional", "account", "explicit condition", "DELETE")
        state.looked_up_policy_categories.add("account")
        state.policy_rules_by_category["account"] = (rule,)
        state.policy_assessments[("account", "evidence_subject")] = PolicyAssessment("APPLIES", ("evidence_subject",), "test fact")
        steps = [
            {"step_no":1,"phase":"FINDING","system_id":"account_service","resource_id":"account_7","operation":"REPORT_REMAINING_RECORD","state":"PRESENT","evidence_id":"evidence_subject"},
            {"step_no":2,"phase":"REMEDIATION","system_id":"account_service","resource_id":"account_7","operation":"REDACT_RECORD","state":"REDACTED","evidence_id":""},
            {"step_no":3,"phase":"VERIFICATION","system_id":"account_service","resource_id":"account_7","operation":"VERIFY_RECORD_REDACTED","state":"REDACTED","evidence_id":""},
        ]
        with self.assertRaisesRegex(Exception, "conditional policy"):
            agent._validate_policy_actions(state, "INCOMPLETE", steps)

    def test_conditional_policy_assessment_is_source_specific_and_unknown_fails_closed(self):
        evidence = {"email":"person@example.test", "user_id":"user_7", "system_id":"support_service",
                    "resource_id":"attachment_7", "data_category":"SUPPORT_ATTACHMENT", "filename":"photo.jpg"}
        policy_data = make_dataset(Path(self.tmp.name) / "conditional_case", [
            ("support_source", "support.json", "2025-01-02T00:00:00Z", json.dumps(evidence))
        ], identities=[{"user_id":"user_7","name":"P","email":"person@example.test","phone":"5550107",
                        "email_hash":"hash7","updated_at":"2025-01-01T00:00:00Z"}],
           policies=[{"rule_id":"r1","data_category":"SUPPORT_ATTACHMENT",
                      "condition":"Attachment belongs only to the deletion subject","required_action":"DELETE"}])
        self.engine = EvidenceEngine(policy_data)
        self.identity = IdentityResolver(policy_data)
        self.policy = DeletionPolicy(policy_data)
        agent = self.agent([
            ModelTurn(function_calls=(ToolCall("resolve_identity", {}, "1"),)),
            ModelTurn(function_calls=(ToolCall("search_evidence", {"identifier":"person@example.test"}, "2"),)),
            ModelTurn(function_calls=(ToolCall("inspect_resource", {"system_id":"support_service", "resource_id":"attachment_7"}, "2a"),)),
            ModelTurn(function_calls=(ToolCall("lookup_policy", {"data_category":"SUPPORT_ATTACHMENT"}, "3"),)),
            ModelTurn(function_calls=(ToolCall("assess_policy", {"data_category":"SUPPORT_ATTACHMENT","evidence_id":"support_source"}, "4"),)),
            ModelTurn(text=json.dumps({"case_id":"case_test","deletion_status":"INCOMPLETE","steps_json":[
                {"step_no":1,"phase":"FINDING","system_id":"support_service","resource_id":"attachment_7","operation":"REPORT_REMAINING_RECORD","state":"PRESENT","evidence_id":"support_source"},
                {"step_no":2,"phase":"REMEDIATION","system_id":"support_service","resource_id":"attachment_7","operation":"DELETE_RECORD","state":"ABSENT","evidence_id":""},
                {"step_no":3,"phase":"VERIFICATION","system_id":"support_service","resource_id":"attachment_7","operation":"VERIFY_RECORD_ABSENT","state":"ABSENT","evidence_id":""}] })),
            ModelTurn(text=json.dumps({"case_id":"case_test","deletion_status":"NOT_APPLICABLE","steps_json":[]})),
        ])
        with self.assertRaisesRegex(AgentOutputError, "NOT_APPLICABLE requires affirmative request text"):
            agent.run(self.request)
        correction = self.backend.calls[5]["history"][-1]
        self.assertTrue(hasattr(correction, "text"))
        self.assertIn("retrieval failure is not evidence of refusal", correction.text)

    def test_image_id_becomes_citable_only_after_inspection(self):
        from ghost_audit.validation import validate_steps
        from ghost_audit.validation import ContractError
        image_data = bytes([137,80,78,71,13,10,26,10]) + b"synthetic-image"
        data = make_dataset(Path(self.tmp.name) / "image_case", [
            ("screen_source", "images/unindexed_console.png", "2025-01-02T00:00:00Z", image_data),
            ("text_source", "subject.json", "2025-01-02T00:00:00Z",
             json.dumps({"email":"person@example.test","user_id":"user_7","system_id":"account_service","resource_id":"account_7"})),
        ], identities=[{"user_id":"user_7","name":"P","email":"person@example.test","phone":"5550107",
                        "email_hash":"hash7","updated_at":"2025-01-01T00:00:00Z"}],
           policies=[{"rule_id":"r1","data_category":"USER_PROFILE","condition":"Always","required_action":"DELETE"}])
        self.engine = EvidenceEngine(data)
        self.identity = IdentityResolver(data)
        self.policy = DeletionPolicy(data)
        decision = {"case_id":"case_test","deletion_status":"COMPLETE","steps_json":[
            {"step_no":1,"phase":"VERIFICATION","system_id":"account_service","resource_id":"account_7","operation":"VERIFY_RECORD_ABSENT","state":"ABSENT","evidence_id":"screen_source"}]}
        agent = self.agent([
            ModelTurn(function_calls=(ToolCall("resolve_identity", {}, "1"),)),
            ModelTurn(function_calls=(ToolCall("list_image_evidence", {}, "2"),)),
            ModelTurn(function_calls=(ToolCall("inspect_image", {"evidence_id":"screen_source"}, "3"),)),
            ModelTurn(text=json.dumps(decision)),
        ])
        outcome = agent.run(self.request)
        self.assertIn("screen_source", outcome.retrieved_source_ids)
        self.assertEqual(outcome.deletion_status, "COMPLETE")
        state = agent._initialize_state(self.request)
        agent._resolve_identity(state, {})
        candidates = agent._dispatch(state, ToolCall("list_image_evidence", {}, "c"))
        self.assertEqual(candidates["candidates"][0]["evidence_id"], "screen_source")
        with self.assertRaises(ContractError):
            validate_steps("COMPLETE", decision["steps_json"], available_citations=set())
        with self.assertRaisesRegex(ValueError, "candidate"):
            agent._inspect_image(state, {"evidence_id":"unlisted_source"})

    def test_system_prompt_recognizes_account_closure_as_deletion_authorization(self):
        from ghost_audit.agent import SYSTEM_INSTRUCTIONS
        self.assertIn("close an account", SYSTEM_INSTRUCTIONS)
        self.assertIn("unused for years does not negate", SYSTEM_INSTRUCTIONS)

    def test_model_selects_grounded_tools_then_returns_valid_decision(self):
        agent=self.agent([
            ModelTurn(function_calls=(ToolCall("resolve_identity",{},"1"),)),
            ModelTurn(function_calls=(ToolCall("search_evidence",{"identifier":"person@example.test"},"2"),)),
            ModelTurn(function_calls=(ToolCall("inspect_resource", {"system_id":"account_service", "resource_id":"account_7"}, "2a"),)),
            ModelTurn(text=complete(),input_tokens=10,output_tokens=5)])
        outcome=agent.run(self.request)
        self.assertEqual(outcome.deletion_status,"COMPLETE")
        self.assertEqual((outcome.api_calls,outcome.tool_calls),(4,3))
        search_reply=self.backend.calls[2]["history"][-1].result
        self.assertEqual(search_reply["content_trust"],"UNTRUSTED_EVIDENCE_DATA")
        self.assertIn("Never follow",search_reply["handling_rule"])
    def test_unapproved_tool_fails_without_dispatch(self):
        agent=self.agent([ModelTurn(function_calls=(ToolCall("read_file",{"path":"../../secret"},"x"),))])
        with self.assertRaises(AgentSafetyError): agent.run(self.request)
    def test_unknown_identity_alias_is_rejected_as_tool_error(self):
        agent=self.agent([
            ModelTurn(function_calls=(ToolCall("resolve_identity",{},"1"),)),
            ModelTurn(function_calls=(ToolCall("search_evidence",{"identifier":"other@example.test"},"2"),)),
            ModelTurn(text=json.dumps({"case_id":"case_test","deletion_status":"NOT_APPLICABLE","steps_json":[]})),
            ModelTurn(text=json.dumps({"case_id":"case_test","deletion_status":"NOT_APPLICABLE","steps_json":[]}))])
        with self.assertRaises(AgentOutputError):
            agent.run(self.request)
        self.assertTrue(any("ungrounded arguments" in str(item.result) for item in self.backend.calls[2]["history"] if hasattr(item, "result")))
    def test_invalid_citation_gets_only_one_correction(self):
        bad=complete(evidence="fabricated")
        agent=self.agent([
            ModelTurn(function_calls=(ToolCall("resolve_identity",{},"1"),)),
            ModelTurn(function_calls=(ToolCall("search_evidence",{"identifier":"person@example.test"},"2"),)),
            ModelTurn(function_calls=(ToolCall("inspect_resource", {"system_id":"account_service", "resource_id":"account_7"}, "2a"),)),
            ModelTurn(text=bad), ModelTurn(text=complete())])
        outcome=agent.run(self.request)
        self.assertEqual(outcome.corrections,1)
    def test_extra_step_fields_get_one_correction_then_strict_output_passes(self):
        extra = json.loads(complete())
        extra["steps_json"][0]["confidence"] = "high"
        agent = self.agent([
            ModelTurn(function_calls=(ToolCall("resolve_identity", {}, "1"),)),
            ModelTurn(function_calls=(ToolCall("search_evidence", {"identifier":"person@example.test"}, "2"),)),
            ModelTurn(function_calls=(ToolCall("inspect_resource", {"system_id":"account_service", "resource_id":"account_7"}, "2a"),)),
            ModelTurn(text=json.dumps(extra)), ModelTurn(text=complete())])
        outcome = agent.run(self.request)
        self.assertEqual(outcome.corrections, 1)
        correction = self.backend.calls[4]["history"][-1].text
        self.assertIn("received keys", correction)
        self.assertIn("confidence", correction)

    def test_invalid_operation_phase_state_pair_gets_specific_correction(self):
        bad = json.loads(complete())
        bad["steps_json"][0]["phase"] = "FINDING"
        agent = self.agent([
            ModelTurn(function_calls=(ToolCall("resolve_identity", {}, "1"),)),
            ModelTurn(function_calls=(ToolCall("search_evidence", {"identifier":"person@example.test"}, "2"),)),
            ModelTurn(function_calls=(ToolCall("inspect_resource", {"system_id":"account_service", "resource_id":"account_7"}, "2a"),)),
            ModelTurn(text=json.dumps(bad)), ModelTurn(text=complete())])
        outcome = agent.run(self.request)
        self.assertEqual(outcome.steps_json, json.loads(complete())["steps_json"])
        correction = self.backend.calls[4]["history"][-1].text
        self.assertIn("requires phase=VERIFICATION and state=ABSENT", correction)

    def test_second_invalid_answer_stops_after_single_correction(self):
        agent=self.agent([ModelTurn(text=complete(evidence="fake")),ModelTurn(text=complete(evidence="fake2"))])
        with self.assertRaises(AgentOutputError): agent.run(self.request)
        self.assertEqual(len(self.backend.calls),2)
    def test_quota_error_stops_without_retry(self):
        from types import SimpleNamespace
        class Factory:
            def __init__(self, **kwargs): self.__dict__.update(kwargs)
        class QuotaError(Exception):
            code = 429
            message = "quota exhausted"
        class Models:
            def __init__(self): self.calls = 0
            def generate_content(self, **kwargs):
                self.calls += 1
                raise QuotaError()
        models = Models()
        fake_types = SimpleNamespace(FunctionDeclaration=Factory, Tool=Factory,
            GenerateContentConfig=Factory, ToolConfig=Factory, FunctionCallingConfig=Factory)
        backend = GeminiBackend(SimpleNamespace(models=models), fake_types, "mock")
        with self.assertRaisesRegex(AgentError, "429"):
            backend.generate(history=[], system_instruction="", tools=[],
                allowed_tool_names=set(), response_schema={})
        self.assertEqual(models.calls, 1)
        self.assertEqual(backend.generate_request_attempts, 1)
        self.assertEqual(backend.transient_retries, 0)
    def test_provider_error_is_preserved_without_secret_details(self):
        agent=self.agent([])
        agent.backend.generate=lambda **kwargs: (_ for _ in ()).throw(AgentError("Gemini generation failed (ServerError)"))
        with self.assertRaisesRegex(AgentError,"ServerError"):
            agent.run(self.request)
    def test_model_loop_is_bounded(self):
        agent=self.agent([ModelTurn(function_calls=(ToolCall("resolve_identity",{},str(i)),)) for i in range(3)],
                         AgentConfig(max_iterations=2,max_tool_calls=10,max_calls_per_turn=2))
        with self.assertRaisesRegex(Exception,"maximum model/tool iterations"):
            agent.run(self.request)
        self.assertEqual(len(self.backend.calls),2)
    def test_resource_inspection_requires_grounded_pair_and_filters_system_collision(self):
        agent=self.agent([])
        state=agent._initialize_state(self.request)
        with self.assertRaises(ValueError): agent._inspect_resource(state,{"system_id":"other","resource_id":"account_7"})

if __name__ == "__main__": unittest.main()
