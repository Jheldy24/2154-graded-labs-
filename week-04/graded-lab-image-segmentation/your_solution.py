import csv
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
manifest_path = HERE / "data" / "manifest.csv"
output_path = HERE / "outputs" / "image_descriptions.csv"

groups = {"train": [], "validation": [], "test": []}

with manifest_path.open(encoding="utf-8", newline="") as file:
    for row in csv.DictReader(file):
        split = row["split"]
        if split not in groups:
            raise ValueError(f"Grupo desconocido: {split}")

        image_path = HERE / "data" / row["image"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"No se pudo leer: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        groups[split].append({
            "image": row["image"],
            "split": split,
            "width": width,
            "height": height,
            "grayscale_dtype": str(gray.dtype),
        })

output_path.parent.mkdir(exist_ok=True)
with output_path.open("w", encoding="utf-8", newline="") as file:
    fields = ["image", "split", "width", "height", "grayscale_dtype"]
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    for split in ("train", "validation", "test"):
        writer.writerows(groups[split])
        print(f"{split}: {len(groups[split])} imágenes")

print(f"Medidas guardadas en: {output_path}")

sample_path = HERE / "data" / "inspection_001.png"
image = cv2.imread(str(sample_path), cv2.IMREAD_COLOR)
if image is None:
    raise FileNotFoundError(f"No se pudo leer: {sample_path}")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
threshold, mask = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

mask_path = HERE / "outputs" / "baseline_inspection_001.png"
if not cv2.imwrite(str(mask_path), mask):
    raise OSError(f"No se pudo guardar: {mask_path}")

print(f"Umbral Otsu: {threshold}")
print(f"Máscara guardada en: {mask_path}")

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

cleaned_path = HERE / "outputs" / "cleaned_inspection_001.png"
if not cv2.imwrite(str(cleaned_path), cleaned):
    raise OSError(f"No se pudo guardar: {cleaned_path}")

print(f"Píxeles blancos antes: {cv2.countNonZero(mask)}")
print(f"Píxeles blancos después: {cv2.countNonZero(cleaned)}")
print(f"Máscara limpia guardada en: {cleaned_path}")

# A4: medir y guardar evidencia, manteniendo separados validation y test.
evaluation_path = HERE / "outputs" / "validation_test_summary.csv"
results = []
totals = {
    "validation": {"before": 0, "after": 0},
    "test": {"before": 0, "after": 0},
}

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

with manifest_path.open(encoding="utf-8", newline="") as file:
    for row in csv.DictReader(file):
        split = row["split"]
        if split not in ("validation", "test"):
            continue

        image_path = HERE / "data" / row["image"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"No se pudo leer: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, baseline = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        cleaned = cv2.morphologyEx(baseline, cv2.MORPH_CLOSE, kernel)

        pixels_before = cv2.countNonZero(baseline)
        pixels_after = cv2.countNonZero(cleaned)
        components_before = cv2.connectedComponents(
            baseline, connectivity=8
        )[0] - 1
        components_after = cv2.connectedComponents(
            cleaned, connectivity=8
        )[0] - 1

        results.append({
            "image": row["image"],
            "split": split,
            "pixels_before": pixels_before,
            "pixels_after": pixels_after,
            "components_before": components_before,
            "components_after": components_after,
        })
        totals[split]["before"] += pixels_before
        totals[split]["after"] += pixels_after

        panels = [
            image.copy(),
            cv2.cvtColor(baseline, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR),
        ]
        for panel, label in zip(panels, ("Original", "Base", "Cierre")):
            cv2.putText(
                panel, label, (4, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1
            )

        comparison_path = HERE / "outputs" / f"comparison_{row['image']}"
        if not cv2.imwrite(str(comparison_path), cv2.hconcat(panels)):
            raise OSError(f"No se pudo guardar: {comparison_path}")

with evaluation_path.open("w", encoding="utf-8", newline="") as file:
    fields = [
        "image", "split", "pixels_before", "pixels_after",
        "components_before", "components_after",
    ]
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)

for split in ("validation", "test"):
    print(
        f"{split}: {sum(r['split'] == split for r in results)} imágenes; "
        f"píxeles blancos base={totals[split]['before']}, "
        f"después del cierre={totals[split]['after']}"
    )

print(f"Resumen guardado en: {evaluation_path}")

# Part B: isolate the red indicator with HSV color thresholds.
part_b_summary_path = HERE / "outputs" / "part_b_summary.csv"
part_b_results = []

with manifest_path.open(encoding="utf-8", newline="") as file:
    for row in csv.DictReader(file):
        split = row["split"]
        if split not in ("validation", "test"):
            continue

        image_path = HERE / "data" / row["image"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"No se pudo leer: {image_path}")

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red_low_hue = cv2.inRange(hsv, (0, 80, 40), (10, 255, 255))
        red_high_hue = cv2.inRange(hsv, (170, 80, 40), (180, 255, 255))
        red_mask = cv2.bitwise_or(red_low_hue, red_high_hue)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, baseline = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        part_b_results.append({
            "image": row["image"],
            "split": split,
            "condition": row["condition"],
            "label": row["label"],
            "otsu_foreground_pixels": cv2.countNonZero(baseline),
            "otsu_components": cv2.connectedComponents(
                baseline, connectivity=8
            )[0] - 1,
            "hsv_red_pixels": cv2.countNonZero(red_mask),
            "hsv_red_components": cv2.connectedComponents(
                red_mask, connectivity=8
            )[0] - 1,
        })

        panels = [
            image.copy(),
            cv2.cvtColor(baseline, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR),
        ]
        for panel, label in zip(panels, ("Original", "Otsu", "HSV red")):
            cv2.putText(
                panel, label, (4, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1
            )

        comparison_path = HERE / "outputs" / f"part_b_comparison_{row['image']}"
        if not cv2.imwrite(str(comparison_path), cv2.hconcat(panels)):
            raise OSError(f"No se pudo guardar: {comparison_path}")

with part_b_summary_path.open("w", encoding="utf-8", newline="") as file:
    fields = [
        "image", "split", "condition", "label",
        "otsu_foreground_pixels", "otsu_components",
        "hsv_red_pixels", "hsv_red_components",
    ]
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(part_b_results)

for split in ("validation", "test"):
    split_results = [r for r in part_b_results if r["split"] == split]
    print(
        f"Part B {split}: {len(split_results)} images; "
        f"HSV red pixels={sum(r['hsv_red_pixels'] for r in split_results)}, "
        f"HSV red components={sum(r['hsv_red_components'] for r in split_results)}"
    )

print(f"Part B summary saved to: {part_b_summary_path}")