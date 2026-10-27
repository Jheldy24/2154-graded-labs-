# Part A Report

## Business question
Can a simple grayscale image feature pipeline separate Class 0 and Class 1 images?

## Data and fixed split
- Total images: 24
- Train: 14
- Validation: 5
- Test: 5
- Label counts:
  - Train: 0=7, 1=7
  - Validation: 0=3, 1=2
  - Test: 0=2, 1=3

## Feature representation
- number of images: 24
- number of features per image: 1024
- feature scaling rule: grayscale -> resize to 32x32 -> divide by 255 -> flatten
- why this feature representation was selected: it is simple, reproducible, and suitable for classic image classification

## Majority baseline
The baseline always predicts Class 0, the most common training class.
The baseline test score is 0.4000.

## Classical classifier
- classifier: LogisticRegression
- important settings: StandardScaler + LogisticRegression(max_iter=4000, random_state=42)
- random seed: 42
- training command: model.fit(X_train, y_train)

## Metrics table

| Method | Split | Accuracy | Precision Class 1 | Recall Class 1 | F1 Class 1 |
|---|---|---:|---:|---:|---:|
| Baseline | validation | 0.6000 | 0.0000 | 0.0000 | 0.0000 |
| Baseline | test | 0.4000 | 0.0000 | 0.0000 | 0.0000 |
| LogisticRegression | validation | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| LogisticRegression | test | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Confusion matrix explanation
The confusion matrix for the final model is:
- Actual 0 -> Predicted 0: 2
- Actual 0 -> Predicted 1: 0
- Actual 1 -> Predicted 0: 0
- Actual 1 -> Predicted 1: 3

The false-negative cell contains 0. This means 0 actual Class 1 images were predicted as Class 0.

## Error review
There are 0 test errors. The error details are saved in outputs/error_review.csv.
If there are no errors, that does not prove production readiness. Five test images are too few to show how the model will perform in new situations.

## Part A conclusion
The logistic-regression model is a clean and explainable standard baseline for this small image task. The small dataset means the result should still be checked by a human before real use.
