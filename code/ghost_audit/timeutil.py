"""Strict UTC timestamp helpers."""
from datetime import datetime, timezone
def parse_time(value):
    if isinstance(value, datetime):
        parsed = value
    else:
        raw = str(value).strip()
        if not raw:
            raise ValueError("timestamp is empty")
        parsed = datetime.fromisoformat(raw[:-1] + "+00:00" if raw.endswith("Z") else raw)
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp must include a timezone: {value!r}")
    return parsed.astimezone(timezone.utc)
def time_is_at_or_before(value, cutoff):
    if value in (None, ""):
        return False
    try:
        return parse_time(value) <= cutoff
    except (TypeError, ValueError):
        return False
