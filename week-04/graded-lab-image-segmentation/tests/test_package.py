from pathlib import Path
import unittest

import cv2


HERE = Path(__file__).resolve().parents[1]


class PackageTest(unittest.TestCase):
    def test_manifest_has_24_images(self):
        rows = (HERE / "data" / "manifest.csv").read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(rows), 25)

    def test_supplied_images_load(self):
        images = sorted((HERE / "data").glob("inspection_*.png"))
        self.assertEqual(len(images), 24)
        for path in images:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            self.assertIsNotNone(image, path.name)
            self.assertEqual(image.shape, (96, 96, 3))


if __name__ == "__main__":
    unittest.main()
