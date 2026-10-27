# Part B Change Log

## Problem noticed in Part A
The original 32x32 grayscale feature vector may be too detailed for a very small dataset.

## Improvement selected
Compare the original 32x32 grayscale features with a reduced 16x16 version.

## Exact code or parameter change
The resize step changes from (32, 32) to (16, 16). The pipeline stays the same.

## Before results
- Feature count: 1024
- Validation F1: 1.0000
- Test accuracy: 1.0000
- Precision Class 1: 1.0000
- Recall Class 1: 1.0000
- F1 Class 1: 1.0000

## After results
- Feature count: 256
- Validation F1: 1.0000
- Test accuracy: 1.0000
- Precision Class 1: 1.0000
- Recall Class 1: 1.0000
- F1 Class 1: 1.0000

## Comparison table

| Method | Feature count | Test accuracy | Precision Class 1 | Recall Class 1 | F1 Class 1 | Main observation |
|---|---:|---:|---:|---:|---:|---|
| Original 32x32 | 1024 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Higher detail |
| Improved 16x16 | 256 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Less noise |

## Trade-off or remaining risk
A smaller vector may lose fine visual detail, but it may also reduce noise in a tiny sample.

## Final decision
Reject the 16x16 improvement and keep the original method because its validation F1 score did not improve. The test results are reported as final evidence. They were not used to choose the improvement.
