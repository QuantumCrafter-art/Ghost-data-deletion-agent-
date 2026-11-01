"""Load simple dotenv assignments without overriding the process environment."""
import os
import re
import shlex
from pathlib import Path

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def _parse_value(raw):
    value = raw.strip()
    if not value:
        return ""
    if value[0] in "\"'":
        try:
            parts = shlex.split(value, comments=True, posix=True)
        except ValueError:
            return None
        return parts[0] if parts else ""
    return re.split(r"\s+#", value, maxsplit=1)[0].rstrip()

def load_project_env(project_root):
    """Load root .env then code/.env; existing process variables take precedence.

    Returns only variable names loaded, never their values.
    """
    root = Path(project_root)
    loaded = []
    for path in (root / ".env", root / "code" / ".env"):
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8-sig").splitlines()
        except OSError:
            continue
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                stripped = stripped[7:].lstrip()
            name, separator, raw = stripped.partition("=")
            name = name.strip()
            if not separator or not _NAME.fullmatch(name) or name in os.environ:
                continue
            value = _parse_value(raw)
            if value is None:
                continue
            os.environ[name] = value
            loaded.append(name)
    return tuple(loaded)
