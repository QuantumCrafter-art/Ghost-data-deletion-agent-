# Public-sample model usage

## Groq integration test

The configured `.env` model was `llama-3.3-70b-versatile`; this was preserved. For live testing, a process-only `GROQ_MODEL=openai/gpt-oss-120b` override was used because Groq currently lists the configured Llama model as deprecated for developer-tier use. The override was not written into `.env`.

- Model availability lookup: passed.
- Text generation smoke: passed in 300.8 ms; response `OK`.
- Forced function-call smoke: passed in 336.9 ms.
- One real case (`case_01`): 6 agent turns, 7 request attempts including 1 bounded rate-limit retry; 9,936 input and 871 output tokens. The case passed the local output validator but was semantically wrong and retrieved zero evidence IDs.
- Earlier end-to-end integration attempts failed with API 400 errors for JSON-mode/tool incompatibility, a model-supplied limit outside the schema, and an unlisted `json` tool call. Those were fixed with a tool-compatible request, local bounds validation, and an explicit tool protocol instruction. The first run then hit one 429 TPM limit; a single bounded wait/retry succeeded. The failed attempts are retained as sanitized diagnostics, not completed decisions.

Token counts above are the one-case agent run only; smoke-test token metadata is not collected by the current smoke helpers. Cost is not estimated because account billing rates were not configured. No keys or credentials are recorded. No other public samples or the 518-case dataset were processed.
