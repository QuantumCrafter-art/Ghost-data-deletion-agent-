"""Run the conservative evidence-only predictor without a model provider."""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from ghost_audit.checkpoint import CheckpointStore
from ghost_audit.identity import IdentityResolver
from ghost_audit.index import EvidenceEngine
from ghost_audit.offline import OfflinePredictor
from ghost_audit.policy import DeletionPolicy
from main import dataset_fingerprint, load_requests, write_final_output

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="dataset/deletion_requests.csv")
    parser.add_argument("--checkpoint", default="offline_full.checkpoint.csv")
    parser.add_argument("--errors", default="offline_full.errors.json")
    args = parser.parse_args()
    dataset = ROOT / "dataset"
    input_path = ROOT / args.input
    requests = load_requests(input_path)
    if input_path.name == "deletion_requests.csv" and len(requests) != 518:
        raise ValueError("full request input must contain 518 rows")
    engine = EvidenceEngine(dataset)
    citations = lambda cutoff: engine.valid_citations(cutoff, engine.citation_ids)
    store = CheckpointStore(ROOT / args.checkpoint, requests, citations, dataset_fingerprint(dataset))
    predictor = OfflinePredictor(dataset, engine, IdentityResolver(dataset), DeletionPolicy(dataset))
    errors_path = ROOT / args.errors
    errors = []
    if errors_path.exists():
        try:
            saved = json.loads(errors_path.read_text(encoding="utf-8"))
            if saved.get("request_count") == len(requests): errors = saved.get("errors", [])
        except (OSError, json.JSONDecodeError, AttributeError): pass
    failed_ids = {entry.get("case_id") for entry in errors if entry.get("case_id")}
    visited = 0
    for request in requests:
        case_id = request["case_id"]
        if case_id in store.rows: continue
        visited += 1
        try:
            row, sources, _ = predictor.predict(request)
            row["steps_json"] = json.dumps(row["steps_json"], ensure_ascii=False, separators=(",", ":"))
            store.commit(row, sources)
            if case_id in failed_ids:
                for item in errors:
                    if item.get("case_id") == case_id: item["resolved"] = True
                failed_ids.remove(case_id)
        except Exception as exc:
            if case_id not in failed_ids:
                errors.append({"case_id": case_id, "error": f"{type(exc).__name__}: {str(exc)[:240]}",
                               "at_utc": datetime.now(timezone.utc).isoformat(), "resolved": False})
                failed_ids.add(case_id)
        errors_path.write_text(json.dumps({"request_count": len(requests), "supported_checkpoint_rows": len(store.rows), "errors": errors}, indent=2), encoding="utf-8")
        if visited % 25 == 0:
            errors_path.write_text(json.dumps({"request_count": len(requests), "supported_checkpoint_rows": len(store.rows), "errors": errors}, indent=2), encoding="utf-8")
            print(f"Visited {visited}/{len(requests)} new requests; supported checkpoints {len(store.rows)}", flush=True)
    errors_path.write_text(json.dumps({"request_count": len(requests), "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                                       "supported_checkpoint_rows": len(store.rows), "errors": errors}, indent=2), encoding="utf-8")
    if len(store.rows) == len(requests):
        write_final_output(ROOT / "output.csv", requests, store.ordered_rows(), citations)
        print(f"Wrote validated output.csv with {len(store.rows)} rows")
    else:
        print(f"No final output written: only {len(store.rows)}/{len(requests)} requests have defensible, contract-valid predictions")
    print(f"Unresolved cases: {sum(not e.get('resolved') for e in errors)}; checkpoint: {store.path}; errors: {errors_path}")

if __name__ == "__main__": main()


