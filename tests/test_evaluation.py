import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "code/evaluation"))
from main import compare, compare_samples

class EvaluatorTests(unittest.TestCase):
    def test_exact_public_sample_comparison_and_full_step_fields(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
            prediction=Path(tmp)/"pred.csv"
            prediction.write_bytes((ROOT/"dataset/sample_output.csv").read_bytes())
            metrics=compare(prediction, sample_prediction_path=prediction)
            self.assertEqual(metrics["sample_cases"],32)
            self.assertEqual(metrics["case_exact_accuracy"],1.0)
            self.assertEqual(metrics["field_accuracy"],1.0)
            self.assertEqual(metrics["sequence_accuracy"],1.0)
            self.assertEqual(len(metrics["full_missing_rows"]),518)
            self.assertFalse(metrics["full_row_order_valid"])
    def test_sample_only_score_returns_field_errors_and_pattern_counts(self):
        metrics=compare_samples(ROOT/"dataset/sample_output.csv")
        self.assertEqual(metrics["sample_cases"],32)
        self.assertEqual(metrics["exact_match_accuracy"],1.0)
        self.assertTrue(all(value == 0 for value in metrics["field_errors"].values()))
        self.assertEqual(metrics["top_failure_patterns"],[])
    def test_prediction_csv_rejects_extra_cells(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
            prediction=Path(tmp)/"malformed.csv"
            prediction.write_text("case_id,deletion_status,steps_json\nc,NOT_APPLICABLE,[],extra\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact output column count"):
                compare_samples(prediction, allow_partial=True)

    def test_detects_modified_status(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
            prediction=Path(tmp)/"pred.csv"
            with (ROOT/"dataset/sample_output.csv").open(encoding="utf-8-sig", newline="") as source:
                rows=list(csv.DictReader(source))
            rows[0]["deletion_status"]="INCOMPLETE"
            with prediction.open("w",encoding="utf-8",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=["case_id","deletion_status","steps_json"]);writer.writeheader();writer.writerows(rows)
            metrics=compare(prediction, sample_prediction_path=prediction)
            self.assertLess(metrics["status_accuracy"],1.0)
            self.assertLess(metrics["case_exact_accuracy"],1.0)

if __name__ == "__main__": unittest.main()

