import csv
from pathlib import Path
import unittest

import cv2
import numpy as np


HERE = Path(__file__).resolve().parents[1]


class SegmentationTest(unittest.TestCase):
    def test_otsu_masks_are_binary_and_match_image_dimensions(self):
        with (HERE / "data" / "manifest.csv").open(
            encoding="utf-8", newline=""
        ) as file:
            rows = list(csv.DictReader(file))

        self.assertTrue(rows)
        for row in rows:
            image_path = HERE / "data" / row["image"]
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            self.assertIsNotNone(image, row["image"])

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            self.assertEqual(mask.shape, gray.shape, row["image"])
            self.assertEqual(mask.dtype, np.uint8, row["image"])
            self.assertTrue(
                set(np.unique(mask)).issubset({0, 255}), row["image"]
            )

    def test_hsv_mask_finds_one_red_component_in_validation_and_test(self):
        with (HERE / "data" / "manifest.csv").open(
            encoding="utf-8", newline=""
        ) as file:
            rows = list(csv.DictReader(file))

        evaluation_rows = [
            row for row in rows if row["split"] in {"validation", "test"}
        ]
        self.assertEqual(len(evaluation_rows), 10)

        for row in evaluation_rows:
            image_path = HERE / "data" / row["image"]
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            self.assertIsNotNone(image, row["image"])

            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            low_hue_red = cv2.inRange(hsv, (0, 80, 40), (10, 255, 255))
            high_hue_red = cv2.inRange(hsv, (170, 80, 40), (180, 255, 255))
            red_mask = cv2.bitwise_or(low_hue_red, high_hue_red)
            component_count = cv2.connectedComponents(
                red_mask, connectivity=8
            )[0] - 1

            self.assertEqual(component_count, 1, row["image"])


if __name__ == "__main__":
    unittest.main()
