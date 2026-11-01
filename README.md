# HackerRank Orchestrate - October 2026

## Ghost Data Still Exists

Build an AI agent that audits whether a user's personal data was actually deleted.

A deletion workflow may report **Complete** while data still exists in databases, logs, support attachments, exports, search indexes, archives, or backups. Sync and restore processes may also bring deleted data back.

For each request, the agent must first determine whether deletion was actually requested. It then returns `COMPLETE`, `INCOMPLETE`, or `NOT_APPLICABLE`, reports any remaining data or recreation risk, and produces the ordered remediation and verification steps required to finish an authorized deletion safely.

Read [`problem_statement.md`](./problem_statement.md) for the complete challenge contract.

---

## Quick Start

Clone the repository and move into the project directory:

```bash
git clone https://github.com/interviewstreet/hackerrank-orchestrate-october26.git
cd hackerrank-orchestrate-october26
```

Start your AI coding tool from the **repository root** so it automatically reads [`AGENTS.md`](./AGENTS.md) and logs your session turns to `log.txt` in the root directory.

Build your solution in `code/main.py`, or use another language and document its setup and run command clearly in your README.

Your solution must:

- Read the input files from `dataset/`
- Generate one prediction row for every request in `dataset/deletion_requests.csv`
- Write the final predictions to `output.csv` in the repository root

Run the starter Python entry point with:

```bash
python3 code/main.py
```

The blank template at `dataset/output.csv` is provided as a reference. Your final generated file for submission must be the root-level `output.csv`.


## Dataset

All participant-facing files are in `dataset/`:

| File | Contents |
| --- | --- |
| `deletion_requests.csv` | 518 requests |
| `sample_deletion_requests.csv` | 32 sample requests |
| `sample_output.csv` | Correct sample answers |
| `user_identities.csv` | Current user identities |
| `user_audit_log.csv` | Historical identity changes |
| `system_activity.csv` | Activity across services and data processes |
| `deletion_rules.csv` | Required treatment for each data category |
| `evidence_files.csv` | Manifest of supplied evidence files |
| `output.csv` | Blank output template |

Evidence is stored under `dataset/evidence/` and may include PDFs, images, logs, JSON, plain text, or SQL snapshot exports. SQL captures may contain a primary database, regional replicas, a reporting replica, or a failover replica so replication drift can be investigated directly. The `.sql` files are readable as text and executable with SQLite.

Only `deletion_requests.csv` requires answers. The other files provide evidence and context.

---

## Output

Create `output.csv` with these columns in this exact order:

```text
case_id,deletion_status,steps_json
```

Produce exactly one row per request. `deletion_status` is `COMPLETE`, `INCOMPLETE`, or `NOT_APPLICABLE`. `steps_json` is an ordered JSON array containing findings, remediation actions, and verification checks. Use an empty array (`[]`) for `NOT_APPLICABLE`. Evidence references may be a `file_id`, `activity_id`, or `audit_id`. Determine the resource state at `audit_at` from the available evidence; reported workflow statuses and operation results are claims. Use `dataset/sample_output.csv` as the format reference.

The final file must contain 518 data rows sorted by `case_id` (plus the header row). The grader parses `steps_json` as JSON, so object-key order and whitespace do not matter; array order does. Order findings, remediation, and verification in that phase order. Within a phase, follow the operation order in the problem statement, then sort by `system_id` and `resource_id`.

---

## Evaluation

`output.csv` will be compared with hidden canonical answers. Scoring considers:

- deletion assessment accuracy;
- current and historical identity resolution;
- discovery of remaining data and recreation risks;
- correct policy application and evidence references; and
- completeness and ordering of remediation and verification steps.

Your `code.zip` must include:

```text
code/evaluation/evaluation_report.md
code/evaluation/usage_report.md
```

- `evaluation_report.md` must cover case, row, field, and sequence accuracy, at least three additional evaluation cases created by you, failures found, and improvements made.
- `usage_report.md` must cover the final full-dataset run, including model providers and names, model calls, input and output tokens, total and average tokens per request, and estimated total and per-request cost. Include per-model and overall totals when using multiple models.

**Important:** Place `evaluation_report.md` and `usage_report.md` inside `code/evaluation/` (matching `code/evaluation/main.py` in this starter layout), write your complete evaluation metrics, custom evaluation test cases, and token/cost breakdowns directly inside those Markdown files, and keep prompts in source code or Markdown files. Automated code evaluation ignores raw data files (`.csv`, `.tsv`, `.json`, `.jsonl`, `.sql`, `.log`), so content stored only in those formats will not be visible to the code judge or AI interviewer.

---

## Chat Transcript

This repository includes [`AGENTS.md`](./AGENTS.md), which instructs compatible AI coding tools to append conversation summaries to `log.txt` in the repository root:

| Platform | Path |
| --- | --- |
| macOS / Linux | `<repo root>/log.txt` |
| Windows | `<repo root>\log.txt` |

- Always launch your AI coding agent from the repository root (or explicitly ask it to read `./AGENTS.md` first) so every turn is logged to `log.txt` with the required `User Prompt`, `Agent Response Summary`, `Actions`, and `Context` sections.
- Upload `log.txt` directly as your `chat_transcript` submission on HackerRank.
- Do not include API keys, tokens, credentials, or secrets in prompts, code, configuration, or `log.txt`.

---

## Packaging `code.zip` and Submission

| Item | Description |
| --- | --- |
| `code.zip` | Runnable solution, prompts, configuration, `README.md`, and `evaluation/` reports |
| `output.csv` | One answer row for each of the 518 requests in `dataset/deletion_requests.csv` |
| `chat_transcript` | The root `log.txt` file generated by your coding agent |
| AI interview | Live explanation of your architecture, approach, and evaluation |

You may use any programming language, model, framework, or retrieval approach. Include clear setup and execution instructions in `README.md` inside `code.zip`.

To package `code.zip` from the repository root, run:

```powershell
python code/package.py
```

The packager includes only runnable source and evaluation reports. It excludes `.env`, datasets, caches, virtual environments, and nested archives, then verifies required members. Partial agent runs remain in checkpoint files and do not create or replace the final `output.csv`; that file is emitted only after every input request has a validated prediction.

Before submitting, confirm that:

- `output.csv` has the exact header `case_id,deletion_status,steps_json` and contains all 518 data rows sorted by `case_id`;
- every `steps_json` value is a valid JSON array with consecutive step numbers starting at `1`;
- findings cite valid evidence IDs, and `COMPLETE` verification steps cite evidence proving the current state;
- proposed remediation and future verification steps use an empty `evidence_id` (`""`);
- `code.zip` includes your runnable code, `README.md`, `code/evaluation/evaluation_report.md`, and `code/evaluation/usage_report.md`;
- `log.txt` follows the `AGENTS.md` turn format and is uploaded to the `chat_transcript` slot; and
- no API keys, credentials, real personal data, or sensitive configuration values are included.


## Conditional policy evidence mapping

The deterministic assessor resolves only predicates supported by participant evidence fields. It recognizes analytics `IDENTIFIABLE`/`ANONYMIZED` record states and null linkage columns; security audit `retention_until` fields in logs and the `security_audit_records` SQL table; legal-service `legal_hold_*` rows with `retained_until`; blocked backup state or `backup_restore_entries.restore_allowed=0`; and deleted-customer-reference SQL rows with a deleted account and stable `subject_reference`. `SUPPORT_MESSAGE` shared context and `SUPPORT_ATTACHMENT` exclusive ownership require explicit fields; a status such as `IDENTIFIABLE` or `FOUND` is not enough. `PRODUCT_TELEMETRY` and `EMAIL_DELIVERY_LOG` have no matching textual corpus records, so only explicit predicate fields can resolve them. All other conditional interpretations return `UNKNOWN` and fail closed.

## Model backends

Install `code/requirements.txt`. The controller supports Gemini through the official Google Gen AI SDK and Groq through Groq's OpenAI-compatible Chat Completions API using HTTPX. Select `--provider gemini` or `--provider groq`; `LLM_PROVIDER` selects the default (Gemini). Provide the matching `GEMINI_API_KEY` or `GROQ_API_KEY` in the process environment or a `.env` at the repository root or `code/.env`. Existing process variables win; root `.env` is read before `code/.env`. Keys and `.env` files are excluded from `code.zip`.

Models are configured with `GEMINI_MODEL` (default `gemini-3.8-flash`) or `GROQ_MODEL` (default `openai/gpt-oss-120b`). Check current provider availability before runs; the unsubmitted `code/.env` in this working checkout has an older Groq model setting. Timeouts are set by `GEMINI_REQUEST_TIMEOUT_SECONDS` or `GROQ_REQUEST_TIMEOUT_SECONDS` (5–300 seconds, default 60). The Groq model selected for this task must support tool use; configure a multimodal Groq model if image inspection is needed. Groq function arguments are locally validated by the same allowlisted controller; numeric bounds are enforced locally because Groq may reject invalid bounded arguments before the controller can return a correction. Tool calls are sequential and bounded. Groq generation retries at most once for transient server failures or an explicit rate-limit reset. TPM throttles wait at least 60 seconds before the single retry; repeated throttling stops the run.

## Local evidence-only recovery run

When no runtime model is available, `python code/offline_batch.py` runs the deterministic evidence and policy pass without API calls. Each supported prediction is validated and checkpointed; unresolved cases are recorded in `offline_full.errors.json`, and the script never writes a partial root `output.csv`. Resume by running the same command again. This fallback is intentionally conservative and does not constitute a complete submission unless it resolves every request. The latest checked run resolved 389 of 518 cases; the remaining 130 cases lacked sufficiently specific subject-linked state evidence.

From the repository root, run `python code/main.py` for Gemini or `python code/main.py --provider groq` for Groq. The controller resolves identity before search, applies each request's `audit_at` cutoff, treats evidence as untrusted data, validates citations and policy, and checkpoints each completed case. Partial runs never create or replace final `output.csv`; it is written only when every input case has a validated prediction.

Run local tests with `python -m unittest discover -s tests -v`. To process one public sample with Groq, use `python code/main.py --provider groq --input dataset/sample_deletion_requests.csv --output groq_samples.csv --limit 1`. The sample evaluator can score checkpoints with `python code/evaluation/main.py --samples-only --sample-predictions groq_samples.csv.checkpoint.csv --allow-partial`; coverage and accuracy are reported separately. Do not describe a partial sample result as full-set accuracy. Usage reports record provider/model, attempts, and tokens when available; cost estimates are written only when provider-specific price-rate variables are explicitly configured.
