# Part B — Change Log

## Problem
The inverted grayscale Otsu mask includes the dark dial outline and can miss the red indicator.

## Change
I added an HSV color mask to isolate red pixels. It uses hue ranges 0–10 and 170–180, saturation 80–255, and value 40–255.

## File and parameters
The change is in `your_solution.py`. It keeps the Part A Otsu mask unchanged and writes side-by-side comparisons and measurements for validation and test images.

## Evidence
The HSV mask found one connected red component in each validation and test image. It detected 414 red pixels across the 5 validation images and 437 across the 5 test images. This includes the dim images. The Otsu mask marked 3,674 and 3,697 foreground pixels in those groups, respectively.

Evidence is saved in `outputs/part_b_summary.csv` and `outputs/part_b_comparison_inspection_014.png` through `outputs/part_b_comparison_inspection_023.png`.

## Trade-off and recommendation
HSV isolates the red indicator but does not segment the dark dial outline. Its color thresholds may fail if lighting changes the red hue or if other red objects appear. 
