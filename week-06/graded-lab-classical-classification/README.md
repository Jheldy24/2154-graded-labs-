# AIDA 2154 Week 6 — Graded Lab: Classical Image Classification

**Individual assessment — one D2L submission — 5% of the course grade**

## The task

Build and explain a reproducible classical image-classification workflow that predicts supplied image labels and evaluates the predictions with a baseline, metrics, a confusion matrix, and error analysis.

## Exactly what you must create

```text
graded-lab-classical-classification/
├── your_solution.py                  YOU CREATE THIS
├── outputs/
│   ├── data_summary.csv or .json
│   ├── metrics.csv or .json
│   ├── confusion_matrix.csv or .json
│   ├── error_review.csv or .json
│   └── representative_test_review.png
├── PART-A-REPORT.md                  YOU CREATE THIS
├── PART-B-CHANGE-LOG.md              YOU CREATE THIS
├── LIMITATIONS.md                    YOU CREATE THIS
├── AI-DECLARATION.md                 YOU CREATE THIS
└── tests/test_solution.py            YOU ADD AT LEAST TWO TESTS
```

You may use a notebook instead of `your_solution.py`, but it must run from start to finish and create the required output files.

## Marks and one-submission rule

This is one lab and one D2L submission:

- Part A — guided classification baseline: 3%
- Part B — independent improvement: 2%
- Total: 5%

Do not submit Part A and Part B separately. Keep both parts inside this one lab folder and submit one private-repository link or final commit link in D2L.

## Rules before you start

1. Work in your assigned private AIDA 2154 repository.
2. Do not push to the instructor `AIDA_2154_Graded_Labs` repository.
3. Do not edit, rename, or move files inside `data/`.
4. Keep the supplied train, validation, and test split unchanged.
5. Do not include `.venv`, `__pycache__`, credentials, personal images, unrelated data, or another student's work.
6. You may use the lecture notes and normal Python/scikit-learn documentation.
7. Do not copy the completed Week 6 practice answer package.

## Step 1 — Copy and open the lab

1. Open VS Code.
2. Open your private AIDA 2154 repository folder.
3. Copy the complete `graded-lab-classical-classification` folder into it.
4. Choose **File → Open Folder**.
5. Open the copied `graded-lab-classical-classification` folder.
6. Confirm that you see `README.md`, `requirements.txt`, `data`, and `starter_check.py`.

## Step 2 — Open the correct terminal

Choose **Terminal → New Terminal** and run:

```powershell
Get-Location
```

The final folder name must be `graded-lab-classical-classification`. If it is not, right-click the lab folder and choose **Open in Integrated Terminal**.

## Step 3 — Install the supplied packages

Run each command separately:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

The supplied requirements file installs NumPy, OpenCV, and scikit-learn. Do not create a second requirements file.

If activation is blocked, run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe starter_check.py
```

## Step 4 — Verify the supplied data

Run:

```powershell
py starter_check.py
```

The output should show 24 records: 14 train, 5 validation, and 5 test. This only verifies the supplied data; it does not complete the lab.

## Part A — Guided classification baseline (3%)

Create `your_solution.py` or a fully runnable notebook. Complete these tasks in order.

### A1. Read and inspect the manifest

1. Read `data/manifest.csv`.
2. Load all 24 image files.
3. Confirm every image is 96 × 96 with three colour channels.
4. Count records in train, validation, and test.
5. Count labels 0 and 1 in each split.
6. Save the counts as `outputs/data_summary.csv` or `outputs/data_summary.json`.
7. Record the counts in `PART-A-REPORT.md`.

### A2. Create the feature matrix

For every image:

1. Load it with OpenCV.
2. Convert it to grayscale.
3. Resize it to a documented size, such as 32 × 32.
4. Convert pixels from 0–255 to 0–1 using `pixel / 255`.
5. Flatten the resized image into one row of numbers.
6. Add the row to feature matrix `X`.
7. Add the manifest label to target vector `y`.
8. Record the final shapes of `X` and `y`.

Your report must state:

```text
number of images:
number of features per image:
feature scaling rule:
why this feature representation was selected:
```

### A3. Preserve the fixed split

1. Use the `split` column from the manifest.
2. Create `X_train`, `y_train`.
3. Create `X_validation`, `y_validation`.
4. Create `X_test`, `y_test`.
5. Do not move images between groups.
6. Do not train on test data.
7. Do not use test results to choose parameters.

Record the number of rows in every group.

### A4. Build the majority-class baseline

1. Use `DummyClassifier(strategy="most_frequent")` or an equivalent documented rule.
2. Fit it using only training data.
3. Predict validation labels and test labels.
4. Calculate baseline accuracy and at least one class-specific metric.
5. Save the baseline results.

Explain in plain language:

```text
The baseline always predicts __________________.
The baseline test score is __________________.
```

### A5. Build the classical classifier

1. Select logistic regression, k-nearest neighbours, or another approved classical classifier.
2. Document the classifier name and important settings.
3. Apply feature scaling inside a reproducible pipeline if needed.
4. Fit using only the training data.
5. Use validation data for development evidence.
6. Use test data for the final evaluation.

Record:

```text
classifier:
important settings:
random seed, if used:
training command:
```

### A6. Calculate and save metrics

For both the baseline and classifier, calculate:

- accuracy;
- precision for Class 1;
- recall for Class 1;
- F1 score for Class 1;
- confusion matrix.

Use these formulas in your explanation:

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### A7. Create and explain the confusion matrix

Save `outputs/confusion_matrix.csv` or `.json` using this order:

|  | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | TN | FP |
| Actual 1 | FN | TP |

Explain at least one cell in words. For example:

```text
The false-negative cell contains ___. This means ___ actual Class 1 images were predicted as Class 0.
```

### A8. Review predictions and errors

Create `outputs/error_review.csv` or `.json` with:

```text
image, actual, predicted, error_type, visual_observation
```

If an error exists, include at least one false positive or false negative and save `outputs/representative_test_review.png`.

If there are no errors on this small test split, still create the file, record zero errors, and explain why zero errors do not prove production readiness.

### A9. Create the Part A report

Create `PART-A-REPORT.md` with:

```text
# Part A Report
## Business question
## Data and fixed split
## Feature representation
## Majority baseline
## Classical classifier
## Metrics table
## Confusion matrix explanation
## Error review
## Part A conclusion
```

Include this table:

| Method | Split | Accuracy | Precision Class 1 | Recall Class 1 | F1 Class 1 |
|---|---|---:|---:|---:|---:|

## Part B — Independent improvement (2%)

Choose one meaningful improvement:

1. Compare flattened grayscale pixels with a simple histogram representation.
2. Compare two approved classical classifiers.
3. Compare 16 × 16 and 32 × 32 image features.
4. Add a documented preprocessing change for dim images.
5. Add a class-distribution or class-weight analysis.
6. Improve the error-review output with clearer image evidence.

Complete Part B in this order:

1. Identify a problem or question from Part A.
2. State exactly one improvement.
3. Change one important feature, preprocessing step, or classifier setting.
4. Run the original Part A method again.
5. Run the improved method.
6. Keep the original Part A results unchanged.
7. Compare both methods using the same validation/test rules.
8. Save comparison values and visual evidence.
9. Decide whether to keep, revise, or reject the improvement.

Create `PART-B-CHANGE-LOG.md` with:

```text
# Part B Change Log
## Problem noticed in Part A
## Improvement selected
## Exact code or parameter change
## Before results
## After results
## Comparison table
## Trade-off or remaining risk
## Final decision
```

Include this table:

| Method | Feature count | Test accuracy | Precision Class 1 | Recall Class 1 | F1 Class 1 | Main observation |
|---|---:|---:|---:|---:|---:|---|

## Required limitations file

Create `LIMITATIONS.md` and answer:

1. Why is the dataset too small to prove production readiness?
2. How could class imbalance make accuracy misleading?
3. What could cause false positives and false negatives?
4. Why must a human review classification evidence?
5. What data or testing should happen next?

## Required AI declaration

Create `AI-DECLARATION.md` and state whether AI was used, what it helped with, what you tested personally, what suggestions you rejected, and that you remain responsible for the submission.

## Required tests

Add at least two meaningful tests in `tests/`, such as image loading, feature-row count, label alignment, confusion-matrix shape, or required report fields.

Run:

```powershell
py -m unittest discover -s tests -v
```

## Final checklist

Run:

```powershell
py starter_check.py
py your_solution.py
py -m unittest discover -s tests -v
git status
```

Confirm that the manifest/data were not changed, the split was preserved, baseline and classifier results are saved, metrics and confusion matrix are saved, error review is saved, both reports are complete, limitations and AI declaration are complete, tests pass, and `.venv`/`__pycache__` are excluded.

## One Git and D2L submission

From the root of your private repository:

```powershell
git status
git add week-06/graded-lab-classical-classification
git commit -m "Complete AIDA 2154 Week 6 graded lab"
git push
```

Submit the private repository link or final commit link in D2L. This is one submission worth 5%: Part A is 3% and Part B is 2%.
