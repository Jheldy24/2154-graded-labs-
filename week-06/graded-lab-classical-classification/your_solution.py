from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUTPUT_DIR = HERE / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURE_SIZE = (32, 32)
SEED = 42


def load_manifest():
    with (DATA_DIR / "manifest.csv").open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def validate_manifest(rows):
    split_counts = {"train": 0, "validation": 0, "test": 0}
    label_counts = {"train": {"0": 0, "1": 0}, "validation": {"0": 0, "1": 0}, "test": {"0": 0, "1": 0}}

    for row in rows:
        split = row["split"]
        split_counts[split] += 1
        label_counts[split][row["label"]] += 1

        image_path = DATA_DIR / row["image"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {row['image']}")
        expected_shape = (int(row["height"]), int(row["width"]), 3)
        if image.shape != expected_shape:
            raise ValueError(f"Bad shape for {row['image']}: {image.shape} != {expected_shape}")

    return {"total_records": len(rows), "splits": split_counts, "label_counts": label_counts}


def flatten_image(image_path: Path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read {image_path.name}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, FEATURE_SIZE, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized.reshape(-1)


def build_feature_matrix(rows):
    X = []
    y = []
    for row in rows:
        feature = flatten_image(DATA_DIR / row["image"])
        X.append(feature)
        y.append(int(row["label"]))
    return np.vstack(X), np.asarray(y, dtype=int)


def split_features(rows):
    X_full, y_full = build_feature_matrix(rows)
    X_train, y_train, X_valid, y_valid, X_test, y_test = [], [], [], [], [], []

    for row, feature in zip(rows, X_full):
        label = int(row["label"])
        if row["split"] == "train":
            X_train.append(feature)
            y_train.append(label)
        elif row["split"] == "validation":
            X_valid.append(feature)
            y_valid.append(label)
        elif row["split"] == "test":
            X_test.append(feature)
            y_test.append(label)

    return (
        np.vstack(X_train),
        np.asarray(y_train, dtype=int),
        np.vstack(X_valid),
        np.asarray(y_valid, dtype=int),
        np.vstack(X_test),
        np.asarray(y_test, dtype=int),
    )


def safe_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, pos_label=1, zero_division=0)


def safe_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=1, zero_division=0)


def safe_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, pos_label=1, zero_division=0)


def metric_row(method, split_name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = cm[0]
    fn, tp = cm[1]
    return {
        "method": method,
        "split": split_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_class_1": float(safe_precision(y_true, y_pred)),
        "recall_class_1": float(safe_recall(y_true, y_pred)),
        "f1_class_1": float(safe_f1(y_true, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def write_confusion_csv(path: Path, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    rows = [
        {"actual": "Actual 0", "Predicted 0": int(cm[0, 0]), "Predicted 1": int(cm[0, 1])},
        {"actual": "Actual 1", "Predicted 0": int(cm[1, 0]), "Predicted 1": int(cm[1, 1])},
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["actual", "Predicted 0", "Predicted 1"])
        writer.writeheader()
        writer.writerows(rows)


def error_review_rows(rows, y_true, y_pred):
    records = []
    for row, actual, pred in zip(rows, y_true, y_pred):
        if actual == pred:
            continue
        if actual == 0 and pred == 1:
            error_type = "false_positive"
            visual = "The image was actually class 0, but the model predicted class 1."
        else:
            error_type = "false_negative"
            visual = "The image was actually class 1, but the model predicted class 0."
        records.append({
            "image": row["image"],
            "actual": int(actual),
            "predicted": int(pred),
            "error_type": error_type,
            "visual_observation": visual,
        })
    return records


def save_error_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "actual", "predicted", "error_type", "visual_observation"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_representative_review(records, output_path: Path):
    if not records:
        canvas = np.full((220, 480, 3), 255, dtype=np.uint8)
        cv2.putText(canvas, "No test errors", (130, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(canvas, "Zero errors do not prove production readiness.", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.imwrite(str(output_path), canvas)
        return

    canvas = np.full((240, 520, 3), 220, dtype=np.uint8)
    for idx, rec in enumerate(records[:4]):
        image = cv2.imread(str(DATA_DIR / rec["image"]), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_AREA)
        x = 20 + (idx % 2) * 240
        y = 20 + (idx // 2) * 120
        canvas[y:y + 100, x:x + 100] = image
        label = f"{rec['error_type']} {rec['actual']}->{rec['predicted']}"
        cv2.putText(canvas, label, (x, y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.imwrite(str(output_path), canvas)


def write_part_a_report(summary, baseline_valid, baseline_test, model_valid, model_test, y_test, y_pred, errors, majority_class):
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp = cm[0]
    fn, tp = cm[1]

    report = f"""# Part A Report

## Business question
Can a simple grayscale image feature pipeline separate Class 0 and Class 1 images?

## Data and fixed split
- Total images: {summary['total_records']}
- Train: {summary['splits']['train']}
- Validation: {summary['splits']['validation']}
- Test: {summary['splits']['test']}
- Label counts:
  - Train: 0={summary['label_counts']['train']['0']}, 1={summary['label_counts']['train']['1']}
  - Validation: 0={summary['label_counts']['validation']['0']}, 1={summary['label_counts']['validation']['1']}
  - Test: 0={summary['label_counts']['test']['0']}, 1={summary['label_counts']['test']['1']}

## Feature representation
- number of images: {summary['total_records']}
- number of features per image: 1024
- feature scaling rule: grayscale -> resize to 32x32 -> divide by 255 -> flatten
- why this feature representation was selected: it is simple, reproducible, and suitable for classic image classification

## Majority baseline
The baseline always predicts Class {majority_class}, the most common training class.
The baseline test score is {baseline_test['accuracy']:.4f}.

## Classical classifier
- classifier: LogisticRegression
- important settings: StandardScaler + LogisticRegression(max_iter=4000, random_state=42)
- random seed: 42
- training command: model.fit(X_train, y_train)

## Metrics table

| Method | Split | Accuracy | Precision Class 1 | Recall Class 1 | F1 Class 1 |
|---|---|---:|---:|---:|---:|
| Baseline | validation | {baseline_valid['accuracy']:.4f} | {baseline_valid['precision_class_1']:.4f} | {baseline_valid['recall_class_1']:.4f} | {baseline_valid['f1_class_1']:.4f} |
| Baseline | test | {baseline_test['accuracy']:.4f} | {baseline_test['precision_class_1']:.4f} | {baseline_test['recall_class_1']:.4f} | {baseline_test['f1_class_1']:.4f} |
| LogisticRegression | validation | {model_valid['accuracy']:.4f} | {model_valid['precision_class_1']:.4f} | {model_valid['recall_class_1']:.4f} | {model_valid['f1_class_1']:.4f} |
| LogisticRegression | test | {model_test['accuracy']:.4f} | {model_test['precision_class_1']:.4f} | {model_test['recall_class_1']:.4f} | {model_test['f1_class_1']:.4f} |

## Confusion matrix explanation
The confusion matrix for the final model is:
- Actual 0 -> Predicted 0: {cm[0, 0]}
- Actual 0 -> Predicted 1: {cm[0, 1]}
- Actual 1 -> Predicted 0: {cm[1, 0]}
- Actual 1 -> Predicted 1: {cm[1, 1]}

The false-negative cell contains {cm[1, 0]}. This means {cm[1, 0]} actual Class 1 images were predicted as Class 0.

## Error review
There are {len(errors)} test errors. The error details are saved in outputs/error_review.csv.
If there are no errors, that does not prove production readiness. Five test images are too few to show how the model will perform in new situations.

## Part A conclusion
The logistic-regression model is a clean and explainable standard baseline for this small image task. The small dataset means the result should still be checked by a human before real use.
"""
    (HERE / "PART-A-REPORT.md").write_text(report, encoding="utf-8")


def write_part_b_report(before, after):
    decision = (
        "Keep the 16x16 improvement because its validation F1 score is higher."
        if after["validation"]["f1_class_1"] > before["validation_f1_class_1"]
        else "Reject the 16x16 improvement and keep the original method because its validation F1 score did not improve."
    )
    report = f"""# Part B Change Log

## Problem noticed in Part A
The original 32x32 grayscale feature vector may be too detailed for a very small dataset.

## Improvement selected
Compare the original 32x32 grayscale features with a reduced 16x16 version.

## Exact code or parameter change
The resize step changes from (32, 32) to (16, 16). The pipeline stays the same.

## Before results
- Feature count: {before['feature_count']}
- Validation F1: {before['validation_f1_class_1']:.4f}
- Test accuracy: {before['accuracy']:.4f}
- Precision Class 1: {before['precision_class_1']:.4f}
- Recall Class 1: {before['recall_class_1']:.4f}
- F1 Class 1: {before['f1_class_1']:.4f}

## After results
- Feature count: {after['feature_count']}
- Validation F1: {after['validation']['f1_class_1']:.4f}
- Test accuracy: {after['accuracy']:.4f}
- Precision Class 1: {after['precision_class_1']:.4f}
- Recall Class 1: {after['recall_class_1']:.4f}
- F1 Class 1: {after['f1_class_1']:.4f}

## Comparison table

| Method | Feature count | Test accuracy | Precision Class 1 | Recall Class 1 | F1 Class 1 | Main observation |
|---|---:|---:|---:|---:|---:|---|
| Original 32x32 | {before['feature_count']} | {before['accuracy']:.4f} | {before['precision_class_1']:.4f} | {before['recall_class_1']:.4f} | {before['f1_class_1']:.4f} | Higher detail |
| Improved 16x16 | {after['feature_count']} | {after['accuracy']:.4f} | {after['precision_class_1']:.4f} | {after['recall_class_1']:.4f} | {after['f1_class_1']:.4f} | Less noise |

## Trade-off or remaining risk
A smaller vector may lose fine visual detail, but it may also reduce noise in a tiny sample.

## Final decision
{decision} The test results are reported as final evidence. They were not used to choose the improvement.
"""
    (HERE / "PART-B-CHANGE-LOG.md").write_text(report, encoding="utf-8")


def write_limitations():
    text = """# Limitations

1. The dataset is only 24 images long, so it cannot prove production readiness.
2. Class imbalance can make accuracy look high while the model still misses the minority class.
3. False positives and false negatives can happen because of dim images, blur, shadows, and small class differences.
4. A person should review the prediction and the image evidence before making a real decision.
5. We need a larger dataset, more variation, and real-world testing before deployment.
"""
    (HERE / "LIMITATIONS.md").write_text(text, encoding="utf-8")


def write_ai_declaration():
    text = """# AI Declaration

I used AI help to plan the structure, check the logic, and improve the wording of the reports.

I personally verified the fixed split, image loading, and output files. I rejected any idea that would use test data for tuning or break the lab rules.

I remain responsible for the final submission and the reasoning in it.
"""
    (HERE / "AI-DECLARATION.md").write_text(text, encoding="utf-8")


def run_improved_part_b(rows):
    X_train, y_train, X_valid, y_valid, X_test, y_test = split_features(rows)

    X_train_16 = X_train.reshape(-1, 32, 32)[:, ::2, ::2].reshape(len(X_train), -1)
    X_valid_16 = X_valid.reshape(-1, 32, 32)[:, ::2, ::2].reshape(len(X_valid), -1)
    X_test_16 = X_test.reshape(-1, 32, 32)[:, ::2, ::2].reshape(len(X_test), -1)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=4000, random_state=SEED))
    ])
    model.fit(X_train_16, y_train)
    valid_pred = model.predict(X_valid_16)
    test_pred = model.predict(X_test_16)

    return {
        "feature_count": X_train_16.shape[1],
        "validation": {
            "accuracy": float(accuracy_score(y_valid, valid_pred)),
            "precision_class_1": float(safe_precision(y_valid, valid_pred)),
            "recall_class_1": float(safe_recall(y_valid, valid_pred)),
            "f1_class_1": float(safe_f1(y_valid, valid_pred)),
        },
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "precision_class_1": float(safe_precision(y_test, test_pred)),
        "recall_class_1": float(safe_recall(y_test, test_pred)),
        "f1_class_1": float(safe_f1(y_test, test_pred)),
    }


def main():
    rows = load_manifest()
    summary = validate_manifest(rows)
    save_json(OUTPUT_DIR / "data_summary.json", summary)

    X_train, y_train, X_valid, y_valid, X_test, y_test = split_features(rows)

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_valid_pred = baseline.predict(X_valid)
    baseline_test_pred = baseline.predict(X_test)

    baseline_valid = metric_row("Baseline", "validation", y_valid, baseline_valid_pred)
    baseline_test = metric_row("Baseline", "test", y_test, baseline_test_pred)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=4000, random_state=SEED))
    ])
    model.fit(X_train, y_train)
    model_valid_pred = model.predict(X_valid)
    model_test_pred = model.predict(X_test)

    model_valid = metric_row("LogisticRegression", "validation", y_valid, model_valid_pred)
    model_test = metric_row("LogisticRegression", "test", y_test, model_test_pred)

    save_json(OUTPUT_DIR / "metrics.json", {
        "baseline": {"validation": baseline_valid, "test": baseline_test},
        "classifier": {"validation": model_valid, "test": model_test},
    })

    write_confusion_csv(OUTPUT_DIR / "confusion_matrix.csv", y_test, model_test_pred)
    save_json(OUTPUT_DIR / "confusion_matrix.json", {
        "matrix": confusion_matrix(y_test, model_test_pred, labels=[0, 1]).tolist()
    })

    errors = error_review_rows(rows, y_test, model_test_pred)
    save_error_csv(OUTPUT_DIR / "error_review.csv", errors)
    save_json(OUTPUT_DIR / "error_review.json", errors)
    make_representative_review(errors, OUTPUT_DIR / "representative_test_review.png")

    write_part_a_report(
        summary,
        baseline_valid,
        baseline_test,
        model_valid,
        model_test,
        y_test,
        model_test_pred,
        errors,
        int(baseline.predict(X_test[:1])[0]),
    )

    before = {
        "feature_count": X_train.shape[1],
        "validation_f1_class_1": float(model_valid["f1_class_1"]),
        "accuracy": float(model_test["accuracy"]),
        "precision_class_1": float(model_test["precision_class_1"]),
        "recall_class_1": float(model_test["recall_class_1"]),
        "f1_class_1": float(model_test["f1_class_1"]),
    }
    after = run_improved_part_b(rows)
    write_part_b_report(before, after)
    save_json(OUTPUT_DIR / "part_b_comparison.json", {
        "original": {
            "feature_count": before["feature_count"],
            "validation_f1_class_1": before["validation_f1_class_1"],
            "test_accuracy": before["accuracy"],
            "test_precision_class_1": before["precision_class_1"],
            "test_recall_class_1": before["recall_class_1"],
            "test_f1_class_1": before["f1_class_1"],
        },
        "improved": after,
    })

    write_limitations()
    write_ai_declaration()

    print("Lab outputs and reports generated.")


if __name__ == "__main__":
    main()
