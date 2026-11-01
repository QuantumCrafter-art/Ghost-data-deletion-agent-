"""Build a clean code.zip submission from the repository root."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "code.zip"
REQUIRED = [ROOT / "README.md", ROOT / "code/main.py", ROOT / "code/offline_batch.py", ROOT / "code/package.py", ROOT / "code/requirements.txt"]
REQUIRED += sorted((ROOT / "code/ghost_audit").glob("*.py"))
REQUIRED += sorted((ROOT / "code/evaluation").glob("*.py"))
REQUIRED += sorted((ROOT / "code/evaluation").glob("*.md"))
REQUIRED += sorted((ROOT / "tests").glob("test_*.py"))

def main():
    missing = [path for path in REQUIRED if not path.is_file()]
    if missing:
        raise SystemExit("missing submission file(s): " + ", ".join(str(p.relative_to(ROOT)) for p in missing))
    with ZipFile(OUTPUT, "w", ZIP_DEFLATED) as archive:
        for path in REQUIRED:
            name = path.relative_to(ROOT).as_posix()
            if path.name.lower() == ".env" or path.suffix.lower() in {".zip", ".tar", ".gz", ".tgz", ".7z", ".rar"}:
                continue
            archive.write(path, name)
    with ZipFile(OUTPUT) as archive:
        names = set(archive.namelist())
        forbidden = [name for name in names if ".env" in name.lower() or "__pycache__" in name
                     or name.startswith("dataset/") or Path(name).suffix.lower() in {".zip", ".tar", ".gz", ".tgz", ".7z", ".rar"}]
        if forbidden:
            OUTPUT.unlink(missing_ok=True)
            raise SystemExit("unsafe archive member(s): " + ", ".join(forbidden))
        for required in ("README.md", "code/main.py", "code/evaluation/evaluation_report.md", "code/evaluation/usage_report.md"):
            if required not in names:
                OUTPUT.unlink(missing_ok=True)
                raise SystemExit("submission archive missing " + required)
    print(f"Created {OUTPUT} ({len(names)} files)")

if __name__ == "__main__":
    main()
