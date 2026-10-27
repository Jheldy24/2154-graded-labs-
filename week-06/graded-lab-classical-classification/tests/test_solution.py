import csv
import json
import unittest
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parents[1]
DATA_DIR = HERE / "data"
OUTPUT_DIR = HERE / "outputs"


class Week6Tests(unittest.TestCase):
    def test_manifest_has_24_rows_and_split_counts(self):
        with (DATA_DIR / "manifest.csv").open("r", encoding="utf-8", newline="") as manifest:
            rows = list(csv.DictReader(manifest))
        self.assertEqual(len(rows), 24)
        counts = {"train": 0, "validation": 0, "test": 0}
        for row in rows:
            counts[row["split"]] += 1
        self.assertEqual(counts, {"train": 14, "validation": 5, "test": 5})

    def test_all_images_are_96x96_rgb(self):
        for image_path in sorted(DATA_DIR.glob("inspection_*.png")):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            self.assertIsNotNone(image, image_path.name)
            self.assertEqual(image.shape, (96, 96, 3), image_path.name)

    def test_required_output_files_exist(self):
        self.assertTrue((OUTPUT_DIR / "data_summary.json").exists())
        self.assertTrue((OUTPUT_DIR / "metrics.json").exists())
        self.assertTrue((OUTPUT_DIR / "confusion_matrix.json").exists())
        self.assertTrue((OUTPUT_DIR / "error_review.json").exists())
        self.assertTrue((OUTPUT_DIR / "part_b_comparison.json").exists())
        self.assertTrue((OUTPUT_DIR / "representative_test_review.png").exists())
        self.assertTrue((HERE / "PART-A-REPORT.md").exists())
        self.assertTrue((HERE / "PART-B-CHANGE-LOG.md").exists())
        self.assertTrue((HERE / "LIMITATIONS.md").exists())
        self.assertTrue((HERE / "AI-DECLARATION.md").exists())

    def test_confusion_matrix_shape(self):
        with (OUTPUT_DIR / "confusion_matrix.json").open("r", encoding="utf-8") as f:
            payload = json.load(f)
        self.assertEqual(len(payload["matrix"]), 2)
        self.assertEqual(len(payload["matrix"][0]), 2)


if __name__ == "__main__":
    unittest.main()
