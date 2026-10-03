# Part A — Report

## Business question
Can thresholding and morphology segment the visible features in the inspection images?

## Data and method
I used the supplied synthetic images and kept the train, validation, and test groups separate. Each image is 96 × 96 pixels. Grayscale images use the `uint8` data type.

I used inverted Otsu thresholding (`THRESH_BINARY_INV + THRESH_OTSU`). For `inspection_001.png`, the threshold was 71. Otsu calculates a threshold for each image.

I applied morphological closing with a 3 × 3 elliptical kernel.

## Results
- Validation: 5 images, 3,674 foreground pixels, and 10 connected components before and after closing.
- Test: 5 images, 3,697 foreground pixels, and 10 connected components before and after closing.
- Evidence: `baseline_inspection_001.png`, `cleaned_inspection_001.png`, `comparison_inspection_014.png` through `comparison_inspection_023.png`, and `validation_test_summary.csv`.

## Visual inspection
- Possible false positive: If the indicator is the target, the circular dial outline is also segmented.
- Possible false negative: The red indicator can be missed because it appears bright in grayscale.
- Effect of closing: The pixel and component counts did not change. I saw no clear improvement in the comparisons.

## Limitation and next step
- Limitation: Synthetic images do not show how the method performs on real inspection images.
- Next validation: Test on representative real images with verified masks, including different lighting conditions.
