"""Deterministic lookup over deletion_rules.csv."""
import csv
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from .timeutil import parse_time
@dataclass(frozen=True)
class PolicyRule:
    rule_id: str
    data_category: str
    condition: str
    required_action: str

@dataclass(frozen=True)
class PolicyAssessment:
    applicability: str
    evidence_ids: tuple[str, ...]
    reason: str

class DeletionPolicy:
    ALLOWED_ACTIONS = {"DELETE", "REDACT", "ANONYMIZE", "RETAIN"}
    def __init__(self, dataset_dir):
        with (Path(dataset_dir) / "deletion_rules.csv").open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            expected = ["rule_id", "data_category", "condition", "required_action"]
            if reader.fieldnames != expected:
                raise ValueError("deletion_rules.csv columns do not match contract")
            rules = []
            rule_ids = set()
            for row in reader:
                if any(not (row.get(key) or "").strip() for key in expected):
                    raise ValueError("deletion rule fields must be non-empty")
                rule = PolicyRule(**row)
                if rule.rule_id in rule_ids:
                    raise ValueError("duplicate deletion rule_id")
                if rule.required_action.strip().upper() not in self.ALLOWED_ACTIONS:
                    raise ValueError(f"unsupported required_action for {rule.rule_id}")
                rule_ids.add(rule.rule_id)
                rules.append(rule)
        self.rules = tuple(rules)
    def lookup(self, data_category):
        if not isinstance(data_category, str) or not data_category.strip():
            return ()
        key = data_category.strip().casefold()
        return tuple(rule for rule in self.rules if rule.data_category.casefold() == key)
    def categories(self):
        return tuple(sorted({rule.data_category for rule in self.rules}))

    @staticmethod
    def _explicit_bool(text, keys):
        names = "|".join(re.escape(key) for key in keys)
        match = re.search(r"(?i)\b(?:" + names + r")\b\s*[\"']?\s*[:=]\s*[\"']?\s*(true|false|yes|no|1|0)\b", text)
        if not match:
            return None
        return match.group(1).casefold() in {"true", "yes", "1"}

    @staticmethod
    def _contains_identifier(text, identifiers):
        folded = text.casefold()
        for kind, value in identifiers:
            if not value:
                continue
            candidate = str(value).casefold()
            if kind == "phone":
                matches = re.findall(r"(?i)\bphone(?:_number)?\b\s*[\"']?\s*[:=]\s*[\"']?([+()0-9 .-]+)", text)
                if candidate.isdigit() and any(re.sub(r"\D", "", value) == candidate for value in matches):
                    return True
            elif kind == "email":
                if re.search(r"(?<![A-Za-z0-9_.+%\-])" + re.escape(candidate) + r"(?![A-Za-z0-9_.\-])", text, re.IGNORECASE):
                    return True
            elif re.search(r"(?<![A-Za-z0-9_])" + re.escape(candidate) + r"(?![A-Za-z0-9_])", text, re.IGNORECASE):
                return True
        return False

    @staticmethod
    def _sql_values(text, table):
        match = re.search(r"(?i)\bINSERT\s+INTO\s+[\"`\[]?" + re.escape(table) + r"[\"`\]]?\s+VALUES\s*\((.*)\)\s*;?\s*$", text.strip())
        if not match:
            return None
        try:
            return next(csv.reader(io.StringIO(match.group(1)), delimiter=",", quotechar="'", skipinitialspace=True))
        except (csv.Error, StopIteration):
            return None

    def evidence_categories(self, text):
        """Recognize rule labels and the participant corpus's named SQL tables."""
        categories = set()
        for category in self.categories():
            if re.search(r"(?<![A-Za-z0-9_])" + re.escape(category) + r"(?![A-Za-z0-9_])", text, re.IGNORECASE):
                categories.add(category)
        table_map = {
            "support_records": "SUPPORT_MESSAGE",
            "customer_analytics_profiles": "ANALYTICS_PROFILE",
            "security_audit_records": "SECURITY_AUDIT_LOG",
            "backup_restore_entries": "BACKUP_ARCHIVE",
        }
        for line in text.splitlines() or [text]:
            for table, category in table_map.items():
                if self._sql_values(line, table) is not None:
                    categories.add(category)
            values = (self._sql_values(line, "customer_data_primary") or self._sql_values(line, "customer_data_read_replica") or
                      self._sql_values(line, "customer_data_reporting_replica") or self._sql_values(line, "customer_data_failover_replica"))
            if values and len(values) >= 4:
                if values[1].casefold() == "identity_service" and values[2].casefold() == "deleted_customer_references":
                    categories.add("ACCOUNT_TOMBSTONE")
                elif values[1].casefold() == "legal_service" and values[0].casefold().startswith("legal_hold_"):
                    categories.add("LEGAL_HOLD_RECORD")
            if re.search(r"(?i)\bresource_id\s*[=:\s]+[\"']?legal_hold_[A-Za-z0-9_-]+", line) and re.search(r"(?i)\b(?:system|system_id)\s*[=:\s]+[\"']?legal_service\b", line):
                categories.add("LEGAL_HOLD_RECORD")
        return categories

    @staticmethod
    def _pipe_record_state(text, category):
        match = re.search(r"\|\s*" + re.escape(category) + r"\s*\|\s*([A-Z_]+)\s*\|", text, re.IGNORECASE)
        return match.group(1).upper() if match else None

    @staticmethod
    def _json_record(text):
        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else None
        except (json.JSONDecodeError, TypeError):
            return None

    def assess(self, rule, evidence, identifiers, audit_at):
        """Resolve only explicit policy facts; ambiguous predicates stay UNKNOWN."""
        rows = [(str(source_id), str(text)) for source_id, text in evidence]
        text = "\n".join(content for _, content in rows)
        condition = rule.condition.casefold()
        source_ids = tuple(dict.fromkeys(source for source, _ in rows))
        if condition.strip() == "always":
            return PolicyAssessment("APPLIES", (), "unconditional rule")

        def resolved(value, reason):
            return PolicyAssessment("APPLIES" if value else "DOES_NOT_APPLY", source_ids, reason)

        if "shared" in condition:
            value = self._explicit_bool(text, ("shared_context", "contains_shared_context", "shared_customer_agent_order_context"))
            if value is None:
                exclusive = self._explicit_bool(text, ("subject_only", "belongs_only_to_subject", "exclusive_to_subject"))
                value = not exclusive if exclusive is not None else None
            if value is not None:
                return resolved(value, "explicit shared-context or subject-only field")
        if "belongs only" in condition or "only to the deletion subject" in condition:
            value = self._explicit_bool(text, ("subject_only", "belongs_only_to_subject", "exclusive_to_subject"))
            if value is None:
                shared = self._explicit_bool(text, ("shared_context", "contains_shared_context", "shared"))
                value = not shared if shared is not None else None
            if value is not None:
                return resolved(value, "explicit subject-ownership or shared-context field")
        if "attributable" in condition:
            if self._contains_identifier(text, identifiers):
                return resolved(True, "a resolved subject identifier is present")
            value = self._explicit_bool(text, ("attributable_to_user", "is_attributable", "is_anonymous"))
            if value is not None:
                if self._explicit_bool(text, ("is_anonymous",)) is True:
                    value = False
                return resolved(value, "explicit attribution field")
            record = self._json_record(text)
            status = (record or {}).get("result") or self._pipe_record_state(text, rule.data_category)
            if status == "IDENTIFIABLE":
                return resolved(True, "participant record state explicitly says IDENTIFIABLE")
            if status == "ANONYMIZED":
                return resolved(False, "participant record state explicitly says ANONYMIZED")
            values = self._sql_values(text, "customer_analytics_profiles")
            if values and len(values) >= 7:
                # Schema order: resource_id, customer_reference, email, phone, email_hash, cohort, updated_at.
                identifier_columns = [value.strip().strip("\"").casefold() for value in values[1:5]]
                if all(value in {"null", ""} for value in identifier_columns):
                    return resolved(False, "participant analytics row has all linkage columns NULL")
        if "user email address" in condition:
            email_identifiers = [(kind, value) for kind, value in identifiers if kind == "email"]
            if self._contains_identifier(text, email_identifiers):
                return resolved(True, "resolved subject email is present")
            value = self._explicit_bool(text, ("contains_user_email", "contains_email_address"))
            if value is not None:
                return resolved(value, "explicit email-presence field")
        if "retention period" in condition:
            value = self._explicit_bool(text, ("within_retention_period", "retention_active"))
            if value is not None:
                return resolved(value, "explicit retention-period field")
            deadline = None
            match = re.search(r"(?i)\b(?:retention_until|retention_expires_at)\b\s*[\"']?\s*[:=]\s*[\"']?([^\s,}\"']+)", text)
            if match:
                deadline = match.group(1)
            sql_values = self._sql_values(text, "security_audit_records")
            if sql_values and len(sql_values) >= 5:
                # Participant table schema puts retention_until before updated_at.
                deadline = sql_values[3].strip().strip("\"")
            if deadline:
                try:
                    return resolved(parse_time(audit_at) <= parse_time(deadline), "explicit retention_until compared with audit_at")
                except (TypeError, ValueError):
                    pass
        if "legal hold" in condition:
            value = self._explicit_bool(text, ("legal_hold_active", "active_legal_hold", "legal_hold"))
            if value is not None:
                return resolved(value, "explicit legal-hold field")
            if "LEGAL_HOLD_RECORD" in self.evidence_categories(text):
                until = re.search(r"(?i)\bretained_until\b\s*[\"']?\s*[:=]\s*[\"']?([^\s,}\"']+)", text)
                if until:
                    try:
                        return resolved(parse_time(audit_at) < parse_time(until.group(1)), "legal-service hold retention deadline compared with audit_at")
                    except (TypeError, ValueError):
                        pass
        if "restore access can be blocked" in condition:
            value = self._explicit_bool(text, ("restore_access_blockable", "restore_can_be_blocked", "can_block_restore"))
            if value is not None:
                return resolved(value, "explicit restore-blockability field")
            record = self._json_record(text)
            status = (record or {}).get("result") or self._pipe_record_state(text, rule.data_category)
            if status == "BLOCKED":
                return resolved(True, "participant backup record explicitly says restore state BLOCKED")
            sql_values = self._sql_values(text, "backup_restore_entries")
            if sql_values and len(sql_values) >= 6 and sql_values[4].strip().strip("\"") == "0":
                return resolved(True, "participant backup_restore_entries row has restore_allowed=0")
        if "account is deleted" in condition and "stable internal user_id" in condition:
            deleted = self._explicit_bool(text, ("account_deleted", "is_account_deleted"))
            stable_only = self._explicit_bool(text, ("only_stable_user_id_needed", "stable_user_id_only"))
            values = (self._sql_values(text, "customer_data_primary") or self._sql_values(text, "customer_data_read_replica") or
                      self._sql_values(text, "customer_data_reporting_replica") or self._sql_values(text, "customer_data_failover_replica"))
            payload = None
            if values and len(values) >= 4 and values[1].casefold() == "identity_service" and values[2].casefold() == "deleted_customer_references":
                try:
                    payload = json.loads(values[3])
                except (json.JSONDecodeError, TypeError):
                    payload = None
                if isinstance(payload, dict):
                    deleted = str(payload.get("account_status", "")).casefold() == "deleted"
                    stable_only = bool(re.fullmatch(r"user_[A-Za-z0-9_-]+", str(payload.get("subject_reference", ""))))
            if deleted is not None and stable_only is not None:
                return resolved(deleted and stable_only, "explicit deleted_customer_references record and stable subject_reference")
        return PolicyAssessment("UNKNOWN", source_ids, "condition is not explicit in the cited evidence")
