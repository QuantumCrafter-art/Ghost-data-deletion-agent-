"""Atomic, contract-validated per-case prediction checkpoints."""
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

from .validation import ContractError, validate_steps

FIELDS = ["case_id", "deletion_status", "steps_json"]


class CheckpointStore:
    def __init__(self, path, requests, citations, context_fingerprint=""):
        self.path = Path(path)
        request_rows = list(requests)
        request_ids = [row.get("case_id") for row in request_rows]
        if any(not isinstance(case_id, str) or not case_id for case_id in request_ids) or len(request_ids) != len(set(request_ids)):
            raise ValueError("input requests must have unique non-empty case_id values")
        self.requests = {row["case_id"]: row for row in request_rows}
        request_payload = json.dumps(sorted(request_rows, key=lambda row: row["case_id"]),
                                     ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        self.request_fingerprint = hashlib.sha256(request_payload.encode("utf-8")).hexdigest()
        self.context_fingerprint = context_fingerprint
        self.meta_path = self.path.with_suffix(self.path.suffix + ".meta.json")
        self.citations = citations
        self.rows = {}
        self.retrieved_sources = {}
        if self.meta_path.exists():
            try:
                metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raise ValueError("checkpoint metadata is invalid") from None
            if (not isinstance(metadata, dict) or metadata.get("schema_version") != 2
                    or metadata.get("request_fingerprint") != self.request_fingerprint
                    or metadata.get("context_fingerprint") != self.context_fingerprint):
                raise ValueError("checkpoint metadata does not match this input or evidence context")
            self.retrieved_sources = metadata.get("retrieved_sources", {})
            if not isinstance(self.retrieved_sources, dict):
                raise ValueError("checkpoint citation metadata is invalid")
        elif self.path.exists():
            raise ValueError("checkpoint is missing citation/provenance metadata; refusing unsafe resume")
        if self.path.exists():
            with self.path.open(encoding="utf-8-sig", newline="") as source:
                reader = csv.DictReader(source)
                if reader.fieldnames != FIELDS:
                    raise ValueError("checkpoint columns do not match output contract")
                for row in reader:
                    case_id = row["case_id"]
                    if case_id not in self.requests:
                        raise ValueError("checkpoint contains an unknown case_id")
                    if case_id in self.rows:
                        raise ValueError("checkpoint contains duplicate case_id")
                    retrieved = self.retrieved_sources.get(case_id)
                    if not isinstance(retrieved, list) or any(not isinstance(value, str) for value in retrieved):
                        raise ValueError("checkpoint is missing retrieved-source provenance")
                    self._validate(row, retrieved)
                    self.rows[case_id] = row

    def _validate(self, row, retrieved_sources):
        request = self.requests.get(row["case_id"])
        if request is None:
            raise ValueError("prediction case_id is not in the input")
        try:
            steps = json.loads(row["steps_json"])
            cited = {step.get("evidence_id") for step in steps if isinstance(step, dict) and step.get("evidence_id")}
            retrieved = set(retrieved_sources)
            visible = set(self.citations(request["audit_at"]))
            if not retrieved.issubset(visible):
                raise ContractError("checkpoint provenance contains a source unavailable at audit_at")
            if not cited.issubset(retrieved):
                raise ContractError("checkpoint cites a source not recorded as retrieved")
            validate_steps(row["deletion_status"], steps,
                           available_citations=retrieved & visible)
        except (json.JSONDecodeError, ContractError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid checkpoint row for {row['case_id']}: {exc}") from None

    def commit(self, row, retrieved_sources=()):
        row = {key: row[key] for key in FIELDS}
        retrieved_sources = sorted(set(retrieved_sources))
        self._validate(row, retrieved_sources)
        self.rows[row["case_id"]] = row
        self.retrieved_sources[row["case_id"]] = retrieved_sources
        self.save()

    def _atomic_write(self, path, writer):
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as target:
                writer(target)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temp_name, path)
        except Exception:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise

    def save(self):
        # Write provenance first: a crash can leave extra metadata but never a CSV row without citations.
        metadata = {"schema_version": 2, "request_fingerprint": self.request_fingerprint,
                    "context_fingerprint": self.context_fingerprint,
                    "retrieved_sources": {key: self.retrieved_sources[key] for key in sorted(self.rows)}}
        self._atomic_write(self.meta_path, lambda target: (json.dump(metadata, target, ensure_ascii=False, indent=2), target.write("\n")))
        def write_csv(target):
            writer = csv.DictWriter(target, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(self.rows[key] for key in sorted(self.rows))
        self._atomic_write(self.path, write_csv)

    def ordered_rows(self):
        return [self.rows[key] for key in sorted(self.rows)]
