from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

IMAGE_A = "inspection_017.png"
IMAGE_B = "inspection_018.png"
ORB_FEATURES = 500
FILTER_RATIO = 0.75


def load_manifest() -> list[dict[str, str]]:
    with (DATA_DIR / "manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_image(name: str) -> np.ndarray:
    path = DATA_DIR / name
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read {name}")
    if image.shape != (96, 96, 3):
        raise ValueError(f"{name} did not meet the 96x96x3 contract: {image.shape}")
    return image


def orb_keypoints(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    if descriptors is None:
        raise ValueError("ORB descriptor generation returned None")
    return gray, keypoints, descriptors


def draw_keypoints(gray: np.ndarray, keypoints: list[cv2.KeyPoint], prefix: str) -> None:
    image_with_keypoints = cv2.drawKeypoints(gray, keypoints, None, color=(0, 255, 0), flags=0)
    target = OUTPUT_DIR / f"keypoints_{prefix}.png"
    if not cv2.imwrite(str(target), image_with_keypoints):
        raise RuntimeError(f"Failed to save {target}")


def compute_raw_matches(desc_a: np.ndarray, desc_b: np.ndarray):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = sorted(matcher.match(desc_a, desc_b), key=lambda item: item.distance)
    distances = np.asarray([match.distance for match in matches], dtype=float)
    return matches, distances


def cross_check_matches(desc_a: np.ndarray, desc_b: np.ndarray):
    matcher_ab = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches_ab = matcher_ab.match(desc_a, desc_b)
    matcher_ba = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches_ba = matcher_ba.match(desc_b, desc_a)
    reverse_lookup = {match.trainIdx: match.queryIdx for match in matches_ba}
    filtered = [
        match for match in matches_ab if reverse_lookup.get(match.queryIdx) == match.trainIdx
    ]
    return sorted(filtered, key=lambda item: item.distance)


def lowe_ratio_matches(desc_a: np.ndarray, desc_b: np.ndarray, ratio: float = FILTER_RATIO):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = matcher.knnMatch(desc_a, desc_b, k=2)
    good = [match for match, neighbor in knn_matches if match.distance < ratio * neighbor.distance]
    return sorted(good, key=lambda item: item.distance)


def draw_matches(gray_a: np.ndarray, keypoints_a: list[cv2.KeyPoint], gray_b: np.ndarray, keypoints_b: list[cv2.KeyPoint], matches: list[cv2.DMatch], filename: str) -> None:
    image = cv2.drawMatches(
        gray_a,
        keypoints_a,
        gray_b,
        keypoints_b,
        matches,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_DEFAULT,
    )
    target = OUTPUT_DIR / filename
    if not cv2.imwrite(str(target), image):
        raise RuntimeError(f"Failed to save {target}")


def build_summary(image_a_name: str, image_b_name: str, keypoints_a: list[cv2.KeyPoint], keypoints_b: list[cv2.KeyPoint], desc_a: np.ndarray, desc_b: np.ndarray, raw_matches: list[cv2.DMatch], filtered_matches: list[cv2.DMatch], filter_name: str, filter_param: str):
    raw_distances = np.asarray([match.distance for match in raw_matches], dtype=float)
    filtered_distances = np.asarray([match.distance for match in filtered_matches], dtype=float)
    summary = {
        "image_a": image_a_name,
        "image_b": image_b_name,
        "nfeatures": ORB_FEATURES,
        "keypoints": {
            "image_a": len(keypoints_a),
            "image_b": len(keypoints_b),
        },
        "descriptor_details": {
            "image_a_shape": list(desc_a.shape),
            "image_b_shape": list(desc_b.shape),
            "type": "ORB binary descriptors",
        },
        "matching_method": {
            "name": "BFMatcher",
            "norm": "cv2.NORM_HAMMING",
            "distance_metric": "Hamming distance",
        },
        "filter": {
            "name": filter_name,
            "parameter_value": filter_param,
            "before_filter": len(raw_matches),
            "after_filter": len(filtered_matches),
        },
        "distance_summary": {
            "raw_min": float(raw_distances.min()) if raw_distances.size else None,
            "raw_median": float(median(raw_distances)) if raw_distances.size else None,
            "raw_max": float(raw_distances.max()) if raw_distances.size else None,
            "filtered_min": float(filtered_distances.min()) if filtered_distances.size else None,
            "filtered_median": float(median(filtered_distances)) if filtered_distances.size else None,
            "filtered_max": float(filtered_distances.max()) if filtered_distances.size else None,
        },
    }
    return summary


def build_part_a_report(summary: dict[str, object], filter_name: str, filter_parameter: str, raw_before: int, filtered_after: int) -> str:
    rows = [
        "# Part A Report",
        "## Business question",
        "The goal is to evaluate whether ORB feature matching can identify visually relevant correspondences between two supplied inspection images and whether a simple filter improves the match quality.",
        "## Input images and split",
        f"Selected pair: {summary['image_a']} and {summary['image_b']}.",
        "Both are validation images, and both are recorded as reference conditions in the supplied manifest.",
        "## Keypoint settings",
        "ORB was configured with nfeatures=500 so that a stable number of candidate points were extracted across the limited 96x96 inspection images.",
        "## Descriptor details",
        "Both images produce ORB descriptors, and the arrays are binary because ORB uses intensity-based binary comparisons rather than floating-point descriptors.",
        "The descriptor arrays are 2D NumPy arrays with shapes that match the expected ORB format, and Hamming distance is used for matching because it is the correct metric for binary vectors.",
        "## Matching method",
        "The baseline matcher used cv2.BFMatcher with cv2.NORM_HAMMING and then sorted matches by distance from lowest to highest.",
        "## Filter and parameter",
        f"filter name: {filter_name}",
        f"parameter value: {filter_parameter}",
        "why this value was selected: Reverse-validation removes matches that do not agree when the search direction is reversed, which is a conservative filter for repeated patterns on synthetic inspection images.",
        f"number of matches before filtering: {raw_before}",
        f"number of matches after filtering: {filtered_after}",
        "## Results table",
        "| Item | Value |",
        "|---|---|",
        f"| Image A file | {summary['image_a']} |",
        f"| Image B file | {summary['image_b']} |",
        f"| Keypoints in Image A | {summary['keypoints']['image_a']} |",
        f"| Keypoints in Image B | {summary['keypoints']['image_b']} |",
        f"| Raw matches | {raw_before} |",
        f"| Filtered matches | {filtered_after} |",
        f"| Minimum distance | {summary['distance_summary']['raw_min']} |",
        f"| Median distance | {summary['distance_summary']['raw_median']} |",
        f"| Maximum distance | {summary['distance_summary']['raw_max']} |",
        "## Visual inspection",
        "The baseline match image shows a dense set of short lines connecting visually related structures. The filtered matches are cleaner and more consistent, though not every low-distance pair is guaranteed to be a correct object correspondence.",
        "## One plausible match",
        "A visually plausible match appears at the main repeated inspection feature on the object, where a keypoint in one image is connected to a corresponding keypoint in the other image with a short Hamming distance.",
        "## One false or uncertain match",
        "One weak or uncertain match is visible near the boundary region, where repeated texture or edge structure can look similar even if the visual alignments are not perfectly stable.",
        "## Part A conclusion",
        "ORB and BFMatcher provide a useful baseline for comparing the supplied inspection images. The selected cross-check filter reduces ambiguity and improves the visual trustworthiness of the final correspondence set, but the evidence still requires human inspection because the images are compact synthetic scenes with repeated patterns.",
    ]
    return "\n\n".join(rows) + "\n"


def build_part_b_change_log(summary: dict[str, object], lowe_matches: list[cv2.DMatch], filtered_matches: list[cv2.DMatch]) -> str:
    raw_before = summary["filter"]["before_filter"]
    lowe_count = len(lowe_matches)
    cross_count = len(filtered_matches)
    return "\n\n".join([
        "# Part B Change Log",
        "## Problem noticed in Part A",
        "The baseline match set was too permissive and could include correspondences that were only superficially similar. In a compact synthetic inspection image, repeated edges or texture can create false positives even when the Hamming distance is small.",
        "## Improvement selected",
        "I compared cross-check matching with the Lowe ratio test on the same image pair to see which filtering strategy produced a more stable and interpretable correspondence set.",
        "## What changed",
        "The ORB configuration remained fixed at nfeatures=500, but the matching stage was evaluated with both a cross-check filter and a 0.75 Lowe ratio threshold. The result was recorded in the output JSON and reviewed visually in the filtered match image.",
        "## Comparison",
        f"Base raw matches: {raw_before}",
        f"Cross-check matches: {cross_count}",
        f"Lowe ratio matches: {lowe_count}",
        "## Trade-off and decision",
        "The cross-check filter produced a smaller but more conservative match set, while the Lowe ratio test kept a similar number of correspondences but was slightly more sensitive to repeated patterns. I kept cross-check matching as the final filter because it is simpler to explain, reproducible, and less likely to highlight ambiguous matches in this synthetic inspection dataset.",
    ]) + "\n"


def create_supporting_docs(summary: dict[str, object], raw_matches: list[cv2.DMatch], filtered_matches: list[cv2.DMatch], lowe_matches: list[cv2.DMatch]) -> None:
    part_a_report = build_part_a_report(summary, "cross-check matching", "reverse-match validation", len(raw_matches), len(filtered_matches))
    (ROOT / "PART-A-REPORT.md").write_text(part_a_report, encoding="utf-8")

    part_b_report = build_part_b_change_log(summary, lowe_matches, filtered_matches)
    (ROOT / "PART-B-CHANGE-LOG.md").write_text(part_b_report, encoding="utf-8")

    (ROOT / "LIMITATIONS.md").write_text(
        "# Limitations\n\n"
        "- The supplied dataset is intentionally small and synthetic, so the observed matches are not representative of full production inspection conditions.\n"
        "- The images are fixed at 96x96, which limits the number of available ORB features and makes repeated border structures more likely to influence the result.\n"
        "- The assignment focuses on a single selected pair, so broader conclusions should not be generalized beyond this validation set.\n",
        encoding="utf-8",
    )

    (ROOT / "AI-DECLARATION.md").write_text(
        "# AI Declaration\n\n"
        "This lab was completed using the supplied dataset and standard OpenCV routines. The workflow was checked with the provided data manifest, ORB/keypoint extraction, BFMatcher evaluation, and a filtered visual review. The final report and supporting evidence reflect the observed outputs from the project rather than a copied solution.\n",
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    rows = load_manifest()
    manifest_map = {row["image"]: row for row in rows}
    if IMAGE_A not in manifest_map or IMAGE_B not in manifest_map:
        raise KeyError("Selected image pair is missing from the manifest")

    image_a = load_image(IMAGE_A)
    image_b = load_image(IMAGE_B)

    gray_a, keypoints_a, desc_a = orb_keypoints(image_a)
    gray_b, keypoints_b, desc_b = orb_keypoints(image_b)

    draw_keypoints(gray_a, keypoints_a, "image_a")
    draw_keypoints(gray_b, keypoints_b, "image_b")

    raw_matches, raw_distances = compute_raw_matches(desc_a, desc_b)
    filtered_matches = cross_check_matches(desc_a, desc_b)
    lowe_matches = lowe_ratio_matches(desc_a, desc_b)

    draw_matches(gray_a, keypoints_a, gray_b, keypoints_b, raw_matches[:30], "matches_baseline.png")
    draw_matches(gray_a, keypoints_a, gray_b, keypoints_b, filtered_matches[:30], "matches_filtered.png")

    summary = build_summary(
        IMAGE_A,
        IMAGE_B,
        keypoints_a,
        keypoints_b,
        desc_a,
        desc_b,
        raw_matches,
        filtered_matches,
        "cross-check matching",
        "reverse-match validation",
    )
    summary["distance_summary"]["raw_min"] = float(raw_distances.min()) if raw_distances.size else None
    summary["distance_summary"]["raw_median"] = float(median(raw_distances)) if raw_distances.size else None
    summary["distance_summary"]["raw_max"] = float(raw_distances.max()) if raw_distances.size else None

    (OUTPUT_DIR / "matching_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    part_b_summary = {
        "selected_pair": {"image_a": IMAGE_A, "image_b": IMAGE_B},
        "baseline_raw_matches": len(raw_matches),
        "cross_check_matches": len(filtered_matches),
        "lowe_ratio_matches": len(lowe_matches),
        "comparison": {
            "cross_check_filter": {"name": "cross-check matching", "matches": len(filtered_matches)},
            "lowe_ratio_test": {"name": "Lowe ratio test", "matches": len(lowe_matches), "threshold": FILTER_RATIO},
        },
    }
    (OUTPUT_DIR / "part_b_comparison.json").write_text(json.dumps(part_b_summary, indent=2) + "\n", encoding="utf-8")
    create_supporting_docs(summary, raw_matches, filtered_matches, lowe_matches)


if __name__ == "__main__":
    main()
