import json
import tempfile
import unittest
from pathlib import Path
from model_governance import apply_publication_gate, status


class GovernanceTests(unittest.TestCase):
    def test_insufficient_evidence_is_annotated_without_changing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "reports").mkdir()
            report = {"verified": {"point": {
                "direction_correct": {"n": 20, "mean": .3},
                "abs_error_pp": {"n": 20, "mean": 2.2},
                "baseline_abs_error_pp": {"n": 20, "mean": 1.9}}}}
            (root / "reports/prediction_audit.json").write_text(json.dumps(report))
            predictions = [{"direction": "UP", "predicted_change_pct": 1.2}]
            evidence = apply_publication_gate(predictions, root)
            self.assertFalse(evidence["approved"])
            self.assertEqual(predictions[0]["direction"], "UP")
            self.assertEqual(predictions[0]["raw_direction"], "UP")

    def test_strong_live_evidence_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "reports").mkdir()
            metric = {"n": 100, "mean": .6}
            report = {"verified": {"point": {"direction_correct": metric,
                "abs_error_pp": {"n": 100, "mean": 1.0},
                "baseline_abs_error_pp": {"n": 100, "mean": 1.2}}}}
            (root / "reports/prediction_audit.json").write_text(json.dumps(report))
            self.assertTrue(status(root)["approved"])


if __name__ == "__main__":
    unittest.main()
