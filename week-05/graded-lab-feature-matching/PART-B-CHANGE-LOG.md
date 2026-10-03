# Part B Change Log

## Problem noticed in Part A

The baseline match set was too permissive and could include correspondences that were only superficially similar. In a compact synthetic inspection image, repeated edges or texture can create false positives even when the Hamming distance is small.

## Improvement selected

I compared cross-check matching with the Lowe ratio test on the same image pair to see which filtering strategy produced a more stable and interpretable correspondence set.

## What changed

The ORB configuration remained fixed at nfeatures=500, but the matching stage was evaluated with both a cross-check filter and a 0.75 Lowe ratio threshold. The result was recorded in the output JSON and reviewed visually in the filtered match image.

## Comparison

Base raw matches: 18

Cross-check matches: 8

Lowe ratio matches: 8

## Trade-off and decision

The cross-check filter produced a smaller but more conservative match set, while the Lowe ratio test kept a similar number of correspondences but was slightly more sensitive to repeated patterns. I kept cross-check matching as the final filter because it is simpler to explain, reproducible, and less likely to highlight ambiguous matches in this synthetic inspection dataset.
