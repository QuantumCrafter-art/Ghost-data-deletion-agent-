"""Deterministic extraction; all returned text is untrusted evidence."""
import json
import re
from dataclasses import dataclass
from pathlib import Path
from .timeutil import parse_time
try:
    import pdfplumber
except ImportError:
    pdfplumber = None
try:
    from PIL import Image
except ImportError:
    Image = None
@dataclass(frozen=True)
class ExtractedChunk:
    text: str
    location: str
    event_at: str = None
    image_path: str = None
    extraction_note: str = ""
_TIMESTAMP = re.compile(r"\b\d{4}-\d\d-\d\d[T ]\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:\d\d)?\b")
_TIME_KEYS = {"event_time", "recorded_at", "timestamp", "created_at", "updated_at", "captured_at", "occurred_at", "exported_at"}

def _latest_timestamp(values):
    candidates = [str(value) for value in values if isinstance(value, str) and value]
    if not candidates:
        return None
    try:
        return max(candidates, key=parse_time)
    except (TypeError, ValueError):
        return "INVALID_TIMESTAMP"

def _json_chunks(data):
    """Split JSON into scalar-only nodes so parent timestamps cannot expose future children."""
    chunks = []
    def walk(node, path, inherited_time=None):
        if isinstance(node, dict):
            local_values = [v for k, v in node.items() if k.casefold() in _TIME_KEYS]
            local_time = _latest_timestamp(local_values) or inherited_time
            if inherited_time and local_time:
                local_time = _latest_timestamp([inherited_time, local_time])
            scalar_node = {key: value for key, value in node.items()
                           if not isinstance(value, (dict, list))}
            if path and scalar_node:
                chunks.append(ExtractedChunk(json.dumps(scalar_node, ensure_ascii=False, sort_keys=True), path, local_time))
            for key, value in node.items():
                if isinstance(value, (dict, list)):
                    walk(value, f"{path}.{key}" if path else str(key), local_time)
        elif isinstance(node, list):
            if path and node and all(not isinstance(value, (dict, list)) for value in node):
                chunks.append(ExtractedChunk(json.dumps(node, ensure_ascii=False), path, inherited_time))
            for index, value in enumerate(node):
                if isinstance(value, (dict, list)):
                    walk(value, f"{path}[{index}]", inherited_time)
    walk(data, "")
    if not chunks:
        root_time = _latest_timestamp([value for key, value in data.items() if key.casefold() in _TIME_KEYS]) if isinstance(data, dict) else None
        chunks.append(ExtractedChunk(json.dumps(data, ensure_ascii=False, sort_keys=True), "$", root_time))
    return chunks

def extract_file(path):
    path = Path(path)
    suffix = path.suffix.casefold()
    if suffix == ".json":
        return _json_chunks(json.loads(path.read_text(encoding="utf-8-sig")))
    if suffix == ".pdf":
        if pdfplumber is None:
            return [ExtractedChunk(path.name, "metadata", extraction_note="PDF_TEXT_EXTRACTOR_UNAVAILABLE")]
        chunks = []
        with pdfplumber.open(path) as pdf:
            for number, page in enumerate(pdf.pages, 1):
                chunks.append(ExtractedChunk(page.extract_text() or "", f"page:{number}"))
        return chunks
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        dims = ""
        note = "IMAGE_TEXT_REQUIRES_VISION_OR_OCR"
        if Image is not None:
            try:
                with Image.open(path) as image:
                    dims = f" {image.width}x{image.height}"
            except (OSError, ValueError):
                note = "IMAGE_DECODE_FAILED"
        return [ExtractedChunk(f"{path.name}{dims}", "image", image_path=str(path), extraction_note=note)]
    chunks = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig", errors="replace").splitlines(), 1):
        text = line.strip()
        if text:
            timestamps = _TIMESTAMP.findall(text)
            if not timestamps:
                event_time = None
            elif suffix == ".sql":
                # In participant SQL rows, the final timestamp column is the row's updated_at;
                # earlier timestamps may be retention deadlines embedded in the row.
                event_time = timestamps[-1]
            else:
                leading_time = _TIMESTAMP.match(text)
                event_time = leading_time.group(0) if leading_time else _latest_timestamp(timestamps)
            chunks.append(ExtractedChunk(text, f"line:{number}", event_time))
    return chunks or [ExtractedChunk("", "empty")]

