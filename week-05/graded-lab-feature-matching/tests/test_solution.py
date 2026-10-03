from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE


class SolutionWorkflowTest(unittest.TestCase):
    def test_solution_script_generates_required_outputs(self):
        subprocess.run([sys.executable, str(ROOT / "your_solution.py")], check=True, cwd=str(ROOT))

        required_files = [
            ROOT / "outputs" / "keypoints_image_a.png",
            ROOT / "outputs" / "keypoints_image_b.png",
            ROOT / "outputs" / "matches_baseline.png",
            ROOT / "outputs" / "matches_filtered.png",
            ROOT / "outputs" / "matching_summary.json",
            ROOT / "PART-A-REPORT.md",
            ROOT / "PART-B-CHANGE-LOG.md",
            ROOT / "LIMITATIONS.md",
            ROOT / "AI-DECLARATION.md",
        ]

        for file_path in required_files:
            self.assertTrue(file_path.exists(), msg=f"Missing required file: {file_path.name}")

    def test_summary_file_has_expected_structure(self):
        summary_path = ROOT / "outputs" / "matching_summary.json"
        self.assertTrue(summary_path.exists())
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertIn("image_a", payload)
        self.assertIn("image_b", payload)
        self.assertIn("keypoints", payload)
        self.assertIn("distance_summary", payload)
        self.assertGreater(payload["keypoints"]["image_a"], 0)
        self.assertGreater(payload["keypoints"]["image_b"], 0)


if __name__ == "__main__":
    unittest.main()
