"""Google Gen AI SDK adapter with explicit, manually bounded function calling."""
import os
import time
from pathlib import Path
from .agent import AgentError, ModelTurn, ToolCall, ToolReply, UserMessage
from .envfile import load_project_env

DEFAULT_MODEL = "gemini-3.8-flash"

def _response_function_calls(response, content):
    calls = getattr(response, "function_calls", None)
    if calls:
        return list(calls)
    return [part.function_call for part in (getattr(content, "parts", None) or [])
            if getattr(part, "function_call", None) is not None]

def _response_text(response, has_function_calls):
    if has_function_calls:
        return None
    return getattr(response, "text", None)

def _provider_error_summary(exc):
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    message = getattr(exc, "message", None)
    if not isinstance(message, str) or not message:
        message = "provider request failed"
    key = os.environ.get("GEMINI_API_KEY", "")
    if key:
        message = message.replace(key, "[REDACTED]")
    message = " ".join(message.split())[:240]
    suffix = f" status={status}" if isinstance(status, (int, str)) else ""
    return f"{type(exc).__name__}{suffix}: {message}"


class GeminiBackend:
    provider_name = "Google Gemini"

    def __init__(self, client, types, model_name, timeout_seconds=60):
        self.client = client
        self.types = types
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.generate_request_attempts = 0
        self.transient_retries = 0
        self.model_lookup_attempts = 0
        self.smoke_function_call_attempts = 0
        self.smoke_text_generation_attempts = 0
        self.last_smoke_latencies_ms = {}

    @classmethod
    def from_environment(cls):
        project_root = Path(__file__).resolve().parents[2]
        load_project_env(project_root)
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise AgentError("GEMINI_API_KEY is not configured")
        model_name = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL).strip()
        if not model_name:
            raise AgentError("GEMINI_MODEL cannot be empty")
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise AgentError("Install the official google-genai SDK from code/requirements.txt") from None
        try:
            timeout_seconds = int(os.environ.get("GEMINI_REQUEST_TIMEOUT_SECONDS", "60"))
        except ValueError:
            raise AgentError("GEMINI_REQUEST_TIMEOUT_SECONDS must be an integer") from None
        if not 5 <= timeout_seconds <= 300:
            raise AgentError("GEMINI_REQUEST_TIMEOUT_SECONDS must be between 5 and 300")
        options = types.HttpOptions(
            timeout=timeout_seconds * 1000,
            retry_options=types.HttpRetryOptions(attempts=1),
        )
        try:
            client = genai.Client(api_key=api_key, http_options=options)
        except Exception as exc:
            raise AgentError(f"Could not initialize Gemini client ({type(exc).__name__})") from None
        return cls(client, types, model_name, timeout_seconds)

    def verify_model(self):
        """Use Models.get before a run/smoke test; no generation or token use."""
        self.model_lookup_attempts += 1
        try:
            model = self.client.models.get(model=self.model_name)
        except Exception as exc:
            raise AgentError(f"Configured Gemini model is unavailable ({_provider_error_summary(exc)})") from None
        model_id = getattr(model, "name", None)
        if model_id and model_id.rsplit("/", 1)[-1] != self.model_name.rsplit("/", 1)[-1]:
            raise AgentError("Gemini returned a different model than requested")
        actions = getattr(model, "supported_actions", None)
        if actions and not any("generateContent".casefold() in str(action).casefold() for action in actions):
            raise AgentError("Configured Gemini model does not expose generateContent")
        return model

    def _contents(self, history):
        types = self.types
        contents = []
        for message in history:
            if isinstance(message, UserMessage):
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=message.text)]))
            elif isinstance(message, ModelTurn):
                if message.provider_content is None:
                    raise AgentError("SDK conversation is missing the prior model function-call content")
                contents.append(message.provider_content)
            elif isinstance(message, ToolReply):
                parts = []
                if message.image is not None:
                    media = types.FunctionResponsePart(
                        inline_data=types.FunctionResponseBlob(
                            mime_type=message.image.mime_type,
                            display_name=message.image.display_name or message.image.source_id,
                            data=message.image.data,
                        )
                    )
                    parts.append(media)
                function_response = types.FunctionResponse(
                    name=message.name,
                    id=message.call_id,
                    response=message.result,
                    parts=parts[:1] if message.image is not None else None,
                )
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(function_response=function_response)],
                ))
            else:
                raise AgentError("Unsupported message type in Gemini conversation")
        return contents

    def generate(self, *, history, system_instruction, tools, allowed_tool_names, response_schema):
        declarations = [self.types.FunctionDeclaration(**tool) for tool in tools]
        tool = self.types.Tool(function_declarations=declarations)
        config = self.types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[tool],
            tool_config=self.types.ToolConfig(
                function_calling_config=self.types.FunctionCallingConfig(
                    mode="AUTO",
                )
            ),
            response_mime_type="application/json",
            response_schema=response_schema,
            max_output_tokens=2048,
        )
        contents = self._contents(history)
        response = None
        for attempt in range(2):
            self.generate_request_attempts += 1
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
                break
            except Exception as exc:
                status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                try:
                    status = int(status)
                except (TypeError, ValueError):
                    pass
                if attempt == 0 and status in {408, 500, 502, 503, 504}:
                    self.transient_retries += 1
                    time.sleep(1)
                    continue
                raise AgentError(f"Gemini generation failed ({_provider_error_summary(exc)})") from None

        stage = "candidate extraction"
        try:
            candidates = getattr(response, "candidates", None) or []
            if not candidates or getattr(candidates[0], "content", None) is None:
                raise AgentError("Gemini returned no candidate content")
            content = candidates[0].content
            stage = "function-call extraction"
            function_calls = []
            for function_call in _response_function_calls(response, content):
                call_id = getattr(function_call, "id", None)
                if not call_id:
                    raise AgentError("Gemini returned a function call without an ID")
                args = getattr(function_call, "args", None)
                if not isinstance(args, dict):
                    raise AgentError("Gemini returned invalid function arguments")
                function_calls.append(ToolCall(function_call.name, dict(args), str(call_id)))
            stage = "usage extraction"
            usage = getattr(response, "usage_metadata", None)
            input_tokens = getattr(usage, "prompt_token_count", None) if usage else None
            output_tokens = getattr(usage, "candidates_token_count", None) if usage else None
            stage = "text extraction"
            return ModelTurn(
                function_calls=tuple(function_calls),
                text=_response_text(response, bool(function_calls)),
                provider_content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        except AgentError:
            raise
        except Exception as exc:
            raise AgentError(f"Gemini response parsing failed at {stage} ({type(exc).__name__})") from None

    def smoke_text_generation(self):
        """One minimal text-only request for latency and endpoint diagnostics."""
        started = time.perf_counter()
        self.smoke_text_generation_attempts += 1
        self.generate_request_attempts += 1
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents="Return exactly the two letters OK.",
                config=self.types.GenerateContentConfig(temperature=0, max_output_tokens=16),
            )
            text = getattr(response, "text", None)
            if not isinstance(text, str) or not text.strip():
                raise AgentError("Gemini text-generation smoke test returned empty output")
            return text.strip()
        except AgentError:
            raise
        except Exception as exc:
            raise AgentError(f"Gemini text-generation smoke test failed ({_provider_error_summary(exc)})") from None
        finally:
            self.last_smoke_latencies_ms["text"] = round((time.perf_counter() - started) * 1000, 1)

    def smoke_function_call(self):
        """One minimal provider-compatibility call; caller must verify the model first."""
        schema = {
            "name": "confirm_tool",
            "description": "Return the supplied smoke-test marker.",
            "parameters": {"type": "OBJECT", "properties": {"marker": {"type": "STRING"}}, "required": ["marker"]},
        }
        declaration = self.types.FunctionDeclaration(**schema)
        tool = self.types.Tool(function_declarations=[declaration])
        config = self.types.GenerateContentConfig(
            tools=[tool],
            tool_config=self.types.ToolConfig(function_calling_config=self.types.FunctionCallingConfig(
                mode="ANY", allowed_function_names=["confirm_tool"])),
            max_output_tokens=64,
        )
        started = time.perf_counter()
        self.smoke_function_call_attempts += 1
        self.generate_request_attempts += 1
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents='Call confirm_tool once with marker exactly "codex-smoke".',
                config=config,
            )
        except Exception as exc:
            raise AgentError(f"Gemini function-calling smoke test failed ({_provider_error_summary(exc)})") from None
        finally:
            self.last_smoke_latencies_ms["function_call"] = round((time.perf_counter() - started) * 1000, 1)
        calls = getattr(response, "function_calls", None) or []
        if not calls or calls[0].name != "confirm_tool" or calls[0].args.get("marker") != "codex-smoke":
            raise AgentError("Gemini smoke test did not return the expected tool call")
        return True


