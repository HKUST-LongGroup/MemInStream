import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import evaluate as ev


class ScoringTests(unittest.TestCase):
    def test_options_and_ambiguous_prose(self):
        row = dict(question_type="multiple-choice", options=["A. Cat", "B. Dog"], correct_option="B")
        for response in ("B", "Answer: B", "B. Dog", "Dog"):
            self.assertTrue(ev.score(row, response, "skip")["correct"])
        for response in ("A or B", "I cannot tell", "", "C"):
            self.assertFalse(ev.score(row, response, "skip")["correct"])

    def test_multiselect_and_count(self):
        row = dict(question_type="multiple-choice", options=["A. Cat", "B. Dog"], correct_option=["A", "B"])
        result = ev.score(row, "A, B", "skip")
        self.assertTrue(result["correct"])
        self.assertTrue(result["multiselect_prompt_mismatch"])
        self.assertFalse(ev.score(row, "A", "skip")["correct"])
        row = dict(question_type="open-ended", original_task="Repetition Event Count", answer="2")
        self.assertTrue(ev.score(row, "2.", "skip")["correct"])
        self.assertFalse(ev.score(row, "12", "skip")["correct"])

    def test_binary_and_open(self):
        self.assertTrue(ev.score(dict(question_type="binary", answer="No"), "no.", "skip")["correct"])
        self.assertFalse(ev.score(dict(question_type="binary", answer="Yes"), "unknown", "skip")["correct"])
        row = dict(question_type="open-ended", answer="A person runs")
        self.assertIsNone(ev.score(row, "Someone is running", "skip")["correct"])
        self.assertFalse(ev.score(row, "Someone is running", "exact")["correct"])
        self.assertTrue(ev.score(row, "Someone is running", "skip", True)["correct"])

    def test_cli_denominators_and_reject_wrong_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [dict(source_dataset="test", source_id=str(i), question_type=kind,
                         question="Q", answer="Yes") for i, kind in enumerate(["binary", "binary", "open-ended"])]
            predictions = [dict(rows[i], fluxmem_prediction=dict(input_index=i, status="completed", response="Yes")) for i in (0, 2)]
            (root / "input.json").write_text(json.dumps(rows))
            pred_path = root / "pred.jsonl"
            pred_path.write_text("\n".join(json.dumps(r) for r in predictions))
            cmd = [sys.executable, str(Path(ev.__file__)), "--input", str(root / "input.json"),
                   "--predictions", str(pred_path), "--output-dir", str(root / "report")]
            subprocess.run(cmd, check=True, capture_output=True)
            report = json.loads((root / "report/report.json").read_text())
            self.assertEqual(report["overall"]["completed"], 2)
            self.assertEqual(report["overall"]["scored"], 1)
            self.assertEqual(report["objective"]["accuracy_scored"], 1)
            self.assertEqual(report["objective"]["correct_over_total_lower_bound"], .5)
            predictions[0]["source_id"] = "mismatch"
            pred_path.write_text("\n".join(json.dumps(r) for r in predictions))
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("mismatch", result.stderr)


if __name__ == "__main__":
    unittest.main()
