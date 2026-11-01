"""As-of identity resolution that preserves collisions as ambiguity."""
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from .timeutil import parse_time, time_is_at_or_before

def normalize_identifier(field_name, value):
    value = value.strip()
    field = field_name.casefold()
    if field == "email":
        return value.casefold()
    if field == "phone":
        return "".join(ch for ch in value if ch.isdigit())
    return value.casefold()

@dataclass(frozen=True)
class IdentityCandidate:
    user_id: str
    identifiers: dict

@dataclass(frozen=True)
class IdentityResolution:
    status: str
    candidates: tuple
    matched_identifiers: tuple
    @property
    def user_id(self):
        return self.candidates[0].user_id if self.status == "RESOLVED" else None

class IdentityResolver:
    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        self.identities_by_user = defaultdict(list)
        self.audits_by_user = defaultdict(list)
        with (self.dataset_dir / "user_identities.csv").open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                self.identities_by_user[row.get("user_id", "")].append(row)
        with (self.dataset_dir / "user_audit_log.csv").open(encoding="utf-8-sig", newline="") as f:
            for event in csv.DictReader(f):
                self.audits_by_user[event.get("user_id", "")].append(event)

    def _identifiers_for(self, user_id, cutoff):
        values = defaultdict(set)
        for row in self.identities_by_user.get(user_id, ()):
            if not time_is_at_or_before(row.get("updated_at"), cutoff):
                continue
            for field in ("email", "phone", "email_hash"):
                value = row.get(field, "").strip()
                if value:
                    values[field].add(value)
        for event in self.audits_by_user.get(user_id, ()):
            if not time_is_at_or_before(event.get("event_time"), cutoff):
                continue
            field = event.get("field_name", "").casefold()
            if field not in {"email", "phone", "email_hash"}:
                continue
            for key in ("old_value", "new_value"):
                value = event.get(key, "").strip()
                if value:
                    values[field].add(value)
        return values

    def resolve_email(self, email, audit_at):
        cutoff = parse_time(audit_at)
        target = normalize_identifier("email", email)
        matches = {}
        for user_id in self.identities_by_user:
            fields = self._identifiers_for(user_id, cutoff)
            if any(normalize_identifier("email", alias) == target for alias in fields.get("email", set())):
                matches[user_id] = fields
        candidates = tuple(
            IdentityCandidate(user_id, {field: tuple(sorted(vals)) for field, vals in sorted(fields.items())})
            for user_id, fields in sorted(matches.items())
        )
        status = "NOT_FOUND" if not candidates else "RESOLVED" if len(candidates) == 1 else "AMBIGUOUS"
        return IdentityResolution(status, candidates, (email.strip(),) if candidates else ())

    def identifiers_for(self, user_id, audit_at):
        cutoff = parse_time(audit_at)
        if user_id not in self.identities_by_user:
            return None
        values = self._identifiers_for(user_id, cutoff)
        return IdentityCandidate(user_id, {field: tuple(sorted(vals)) for field, vals in sorted(values.items())})

