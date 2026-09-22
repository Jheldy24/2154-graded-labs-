from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent


def main() -> None:
    rows = list(csv.DictReader((HERE / "data" / "manifest.csv").open(encoding="utf-8", newline="")))
    counts: dict[str, int] = {}
    for row in rows:
        path = HERE / "data" / row["image"]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read {row['image']}")
        expected = (int(row["height"]), int(row["width"]), 3)
        if image.shape != expected:
            raise ValueError(f"Image contract failed for {row['image']}: {image.shape} != {expected}")
        counts[row["split"]] = counts.get(row["split"], 0) + 1

    output = {
        "status": "starter_ready",
        "records": len(rows),
        "split_counts": counts,
        "note": "This verifies supplied data only. Implement and defend the assessed method.",
    }
    (HERE / "outputs").mkdir(exist_ok=True)
    (HERE / "outputs" / "starter_evidence.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
