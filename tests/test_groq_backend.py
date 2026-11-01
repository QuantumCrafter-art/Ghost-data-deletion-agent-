import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from ghost_audit.agent import AgentError, ModelTurn, ToolCall, ToolReply, UserMessage
from ghost_audit.groq import GroqBackend, DEFAULT_MODEL, _lower_schema

class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status_code = status
        self.is_error = status >= 400
    def json(self):
        return self.payload

class FakeClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
    def request(self, method, path, **kwargs):
        self.requests.append((method, path, kwargs))
        return self.responses.pop(0)

def completion(message, status=200, usage=None):
    return FakeResponse({"choices": [{"message": message}], "usage": usage or {"prompt_tokens": 11, "completion_tokens": 4}}, status)

class GroqBackendTests(unittest.TestCase):
    def backend(self, client):
        return GroqBackend(client, "configured-model", "synthetic-secret", timeout_seconds=15)

    def test_config_load_uses_groq_environment_without_exposing_key(self):
        fake = FakeClient()
        with patch.dict(os.environ, {"GROQ_API_KEY": "synthetic-secret", "GROQ_MODEL": "configured-model", "GROQ_REQUEST_TIMEOUT_SECONDS": "17"}, clear=False), \
             patch("ghost_audit.groq.httpx.Client", return_value=fake) as make_client:
            backend = GroqBackend.from_environment()
        self.assertEqual(backend.model_name, "configured-model")
        self.assertEqual(backend.timeout_seconds, 17)
        self.assertEqual(make_client.call_args.kwargs["headers"]["Authorization"], "Bearer synthetic-secret")

    def test_generate_translates_tools_and_parses_function_call_and_usage(self):
        response = completion({"role":"assistant", "content":None, "tool_calls":[
            {"id":"tc_1", "type":"function", "function":{"name":"resolve_identity", "arguments":"{}"}}
        ]})
        backend = self.backend(FakeClient(response))
        tool = {"name":"resolve_identity", "description":"resolve", "parameters":{"type":"OBJECT", "properties":{}}}
        turn = backend.generate(history=[UserMessage("case")], system_instruction="system", tools=[tool],
            allowed_tool_names={"resolve_identity"}, response_schema={"type":"OBJECT"})
        self.assertEqual(turn.function_calls, (ToolCall("resolve_identity", {}, "tc_1"),))
        self.assertIsNone(turn.text)
        self.assertEqual((turn.input_tokens, turn.output_tokens), (11, 4))
        body = backend.client.requests[0][2]["json"]
        self.assertEqual(body["tools"][0]["function"]["parameters"]["type"], "object")
        self.assertEqual(body["tool_choice"], "auto")
        self.assertNotIn("response_format", body)  # Groq rejects JSON mode together with tool calling.
        if backend.model_name.startswith("openai/gpt-oss-"):
            self.assertEqual(body["reasoning_effort"], "medium")
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertIn("Groq tool protocol", body["messages"][0]["content"])

    def test_tool_reply_history_preserves_call_id_and_serializes_result(self):
        backend = self.backend(FakeClient(completion({"role":"assistant", "content":"{}"})))
        prior = ModelTurn(provider_content={"role":"assistant", "tool_calls":[{"id":"tc_1"}]})
        messages = backend._messages([UserMessage("start"), prior, ToolReply("resolve_identity", "tc_1", {"ok":True})], "sys")
        self.assertEqual(messages[-1]["role"], "tool")
        self.assertEqual(messages[-1]["tool_call_id"], "tc_1")
        self.assertEqual(json.loads(messages[-1]["content"]), {"ok":True})

    def test_text_and_forced_function_smokes(self):
        client = FakeClient(completion({"role":"assistant", "content":"OK"}), completion({"role":"assistant", "tool_calls":[
            {"id":"tc_smoke", "type":"function", "function":{"name":"confirm_tool", "arguments":"{\"marker\":\"groq-smoke\"}"}}
        ]}))
        backend = self.backend(client)
        self.assertEqual(backend.smoke_text_generation(), "OK")
        self.assertTrue(backend.smoke_function_call())
        self.assertEqual(backend.smoke_text_generation_attempts, 1)
        self.assertEqual(backend.smoke_function_call_attempts, 1)
        self.assertEqual(backend.generate_request_attempts, 2)
        self.assertEqual(client.requests[1][2]["json"]["tool_choice"]["function"]["name"], "confirm_tool")

    def test_tpm_rate_limit_waits_for_window_before_single_retry(self):
        limited = FakeResponse({"error":{"message":"rate limit tokens per minute; try again in 2s"}}, 429)
        client = FakeClient(limited, completion({"role":"assistant", "content":"{}"}))
        backend = self.backend(client)
        with patch("ghost_audit.groq.time.sleep") as sleep:
            result = backend.generate(history=[UserMessage("x")], system_instruction="s", tools=[], allowed_tool_names=set(), response_schema={})
        self.assertEqual(result.text, "{}")
        self.assertEqual((backend.generate_request_attempts, backend.transient_retries), (2, 1))
        self.assertEqual(sleep.call_args.args[0], 61.0)

    def test_rate_limit_reset_header_is_parsed(self):
        from ghost_audit.groq import _retry_after_seconds
        self.assertEqual(_retry_after_seconds({"x-ratelimit-reset-tokens":"12.5s"}), 12.5)
        self.assertEqual(_retry_after_seconds({"retry-after":"4"}), 4.0)

    def test_transient_server_error_retries_once_but_not_quota(self):
        client = FakeClient(FakeResponse({"error":{"message":"temporary"}}, 503), completion({"role":"assistant", "content":"{}"}))
        backend = self.backend(client)
        turn = backend.generate(history=[UserMessage("x")], system_instruction="s", tools=[], allowed_tool_names=set(), response_schema={})
        self.assertEqual(turn.text, "{}")
        self.assertEqual((backend.generate_request_attempts, backend.transient_retries), (2, 1))

    def test_rate_limit_retry_honors_only_short_explicit_delay(self):
        client = FakeClient(FakeResponse({"error":{"message":"Please try again in 0.1s org_private123"}}, 429),
                            completion({"role":"assistant", "content":"{}"}))
        backend = self.backend(client)
        with patch("ghost_audit.groq.time.sleep") as sleep:
            result = backend.generate(history=[UserMessage("x")], system_instruction="s", tools=[], allowed_tool_names=set(), response_schema={})
        self.assertEqual(result.text, "{}")
        self.assertEqual((backend.generate_request_attempts, backend.transient_retries), (2, 1))
        self.assertAlmostEqual(sleep.call_args.args[0], 1.1)
        with self.assertRaises(AgentError) as caught:
            self.backend(FakeClient(FakeResponse({"error":{"message":"rate limit org_private123"}}, 429))).smoke_text_generation()
        self.assertNotIn("org_private123", str(caught.exception))

    def test_provider_errors_are_sanitized(self):
        client = FakeClient(FakeResponse({"error":{"message":"invalid key synthetic-secret"}}, 401))
        backend = self.backend(client)
        with self.assertRaises(AgentError) as caught:
            backend.smoke_text_generation()
        self.assertNotIn("synthetic-secret", str(caught.exception))
        self.assertIn("HTTP 401", str(caught.exception))

    def test_schema_types_are_normalized_recursively(self):
        self.assertEqual(_lower_schema({"type":"ARRAY", "maximum":12, "items":{"type":"STRING", "minimum":1}}),
                         {"type":"array", "items":{"type":"string"}})

if __name__ == "__main__":
    unittest.main()
