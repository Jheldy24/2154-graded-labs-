# AIDA 2154 Week 5 — Graded Lab: ORB Feature Matching

**Individual assessment — one D2L submission — 5% of the course grade**

## The task in one sentence

Build and explain a reproducible Python/OpenCV workflow that finds and evaluates visual feature matches between two supplied inspection images.

## What you will deliver

You will create the files listed below inside this one lab folder. These are the exact deliverables.

```text
graded-lab-feature-matching/
├── README.md                         supplied instructions
├── requirements.txt                  supplied dependency list
├── rubric.md                         supplied marking guide
├── starter_check.py                  supplied data check
├── data/                             supplied data; do not edit
│   ├── README.md
│   ├── manifest.csv
│   └── inspection_000.png ... inspection_023.png
├── your_solution.py                  YOU CREATE THIS
├── outputs/                          YOU SAVE YOUR EVIDENCE HERE
│   ├── keypoints_*.png
│   ├── matches_*.png
│   ├── matching_summary.csv or .json
│   └── other evidence files
├── PART-A-REPORT.md                  YOU CREATE THIS
├── PART-B-CHANGE-LOG.md              YOU CREATE THIS
├── LIMITATIONS.md                    YOU CREATE THIS
├── AI-DECLARATION.md                 YOU CREATE THIS
└── tests/                            YOU ADD AT LEAST TWO TESTS
    └── test_solution.py
```

You may use a notebook instead of `your_solution.py`, but the notebook must run from start to finish and must create the same required evidence.

## Marks and one-submission rule

This is one lab and one submission:

- **Part A — guided challenge: 3%**
- **Part B — independent extension: 2%**
- **Total: 5%**

Do not submit Part A and Part B separately. Keep both parts in the same lab folder and submit one private-repository link or commit link in D2L.

## Rules before you start

1. Work in your assigned private AIDA 2154 repository.
2. Do not push to the instructor `AIDA_2154_Graded_Labs` repository.
3. Do not edit or rename files inside `data/`.
4. Use only the supplied data unless the instructor approves another data source.
5. Do not include `.venv`, `__pycache__`, credentials, personal images, unrelated data, or another student's work.
6. You may read the lecture notes and normal Python/OpenCV documentation.
7. Do not copy the completed Week 5 practice answer package.

## Step 1 — Copy the lab into your private repository

1. Open VS Code.
2. Open your private AIDA 2154 repository folder.
3. Copy the complete `graded-lab-feature-matching` folder into that repository.
4. Choose **File → Open Folder** in VS Code.
5. Open the copied `graded-lab-feature-matching` folder.
6. In the Explorer, confirm that you see `README.md`, `requirements.txt`, `data`, and `starter_check.py`.

## Step 2 — Open the correct terminal folder

Choose **Terminal → New Terminal**.

Run:

```powershell
Get-Location
```

The final folder name in the path must be:

```text
graded-lab-feature-matching
```

If it is not, use the VS Code Explorer to right-click the lab folder and choose **Open in Integrated Terminal**.

## Step 3 — Create the Python environment

Run these commands one at a time:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

The supplied `requirements.txt` installs NumPy and OpenCV. Do not create a different dependency file.

If PowerShell does not allow activation, run commands with the full Python path:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe starter_check.py
```

## Step 4 — Run the supplied data check

Run:

```powershell
py starter_check.py
```

You should see:

```text
status: starter_ready
records: 24
train: 14
validation: 5
test: 5
```

This check does not complete the lab. It only confirms that the supplied data can be read.

## Part A — Guided challenge, worth 3%

Create `your_solution.py`. Complete the following tasks in order.

### A1. Load and describe the supplied images

1. Read `data/manifest.csv`.
2. Load images with `cv2.imread()`.
3. Confirm that each image is 96 × 96 with three colour channels.
4. Convert the selected image pair to grayscale with `cv2.cvtColor()`.
5. Choose one pair of supplied images and record both file names in `PART-A-REPORT.md`.
6. State the split and condition for each selected image.

### A2. Detect ORB keypoints

1. Create an ORB detector with a documented `nfeatures` value.
2. Run `detectAndCompute()` on both grayscale images.
3. Count the keypoints in each image.
4. Draw the keypoints on both images.
5. Save:

   ```text
   outputs/keypoints_image_a.png
   outputs/keypoints_image_b.png
   ```

6. Explain whether the keypoints appear on meaningful visual structures or mostly on background noise.

### A3. Create and compare descriptors

1. Confirm that both descriptor arrays were created.
2. Record the descriptor shape for each image.
3. State that ORB creates binary descriptors.
4. Explain why Hamming distance is used for ORB descriptors.

### A4. Match descriptors with BFMatcher

1. Create a `BFMatcher` using `cv2.NORM_HAMMING`.
2. Run matching between the two descriptor arrays.
3. Sort the matches by distance from lowest to highest.
4. Record:

   - total raw matches;
   - minimum distance;
   - median distance;
   - maximum distance.

5. Draw the best 20 or 30 matches.
6. Save:

   ```text
   outputs/matches_baseline.png
   ```

### A5. Apply one declared match filter

Choose one filter and document it:

- cross-check matching; or
- a maximum Hamming-distance threshold; or
- the Lowe ratio test.

Your report must state:

```text
filter name:
parameter value:
why this value was selected:
number of matches before filtering:
number of matches after filtering:
```

Save the filtered visualization as:

```text
outputs/matches_filtered.png
```

### A6. Inspect the visual evidence

1. Open `matches_baseline.png`.
2. Open `matches_filtered.png`.
3. Identify one visually plausible match.
4. Identify one false, weak, or uncertain match if one is visible.
5. Describe where the match lines appear: target object, border, text, repeated pattern, or background.
6. Do not claim that every low-distance match is correct.

### A7. Create the Part A report

Create `PART-A-REPORT.md` using these headings:

```text
# Part A Report
## Business question
## Input images and split
## Keypoint settings
## Descriptor details
## Matching method
## Filter and parameter
## Results table
## Visual inspection
## One plausible match
## One false or uncertain match
## Part A conclusion
```

The results table must include:

| Item | Value |
|---|---|
| Image A file | your value |
| Image B file | your value |
| Keypoints in Image A | your value |
| Keypoints in Image B | your value |
| Raw matches | your value |
| Filtered matches | your value |
| Minimum distance | your value |
| Median distance | your value |
| Maximum distance | your value |

## Part B — Independent extension, worth 2%

Part B must be a real improvement, not only a comment or file rename. Choose one option:

1. Compare cross-check matching with the Lowe ratio test.
2. Compare two Hamming-distance thresholds.
3. Test a second approved image pair from the supplied data.
4. Test the effect of blur or brightness change.
5. Add a connected visual error-review output.
6. Compare two ORB `nfeatures` settings.

Complete Part B in this order:

1. State the problem noticed in Part A.
2. State the improvement selected.
3. Change one important method or parameter.
4. Run the original Part A method again.
5. Run the improved method.
6. Save both results; do not overwrite the Part A evidence.
7. Compare the results using numbers and visual evidence.
8. Decide whether to keep, revise, or reject the improvement.

Create `PART-B-CHANGE-LOG.md` with exactly these headings:

```text
# Part B Change Log
## Problem noticed in Part A
## Improvement selected
## Exact code or parameter change
## Before evidence
## After evidence
## Comparison table
## Trade-off or remaining risk
## Final decision
```

The comparison table must include:

| Method | Keypoints A | Keypoints B | Matches before filter | Matches after filter | Median distance | Visual finding |
|---|---:|---:|---:|---:|---:|---|

## Required final files

Before committing, confirm that your folder contains all of these:

- `your_solution.py` or a runnable notebook;
- `PART-A-REPORT.md`;
- `PART-B-CHANGE-LOG.md`;
- `LIMITATIONS.md`;
- `AI-DECLARATION.md`;
- `outputs/keypoints_image_a.png`;
- `outputs/keypoints_image_b.png`;
- `outputs/matches_baseline.png`;
- `outputs/matches_filtered.png`;
- `outputs/matching_summary.csv` or `outputs/matching_summary.json`;
- at least two tests in `tests/`.

## Required limitations file

Create `LIMITATIONS.md` and answer:

1. Why does a small synthetic dataset not prove production readiness?
2. How could lighting, blur, rotation, or repeated patterns create false matches?
3. Why should a human inspect matching evidence before an operational decision?
4. What data or testing should happen next?

## Required AI declaration

Create `AI-DECLARATION.md` and state:

- whether you used an AI tool;
- what the tool helped with, if anything;
- what code and outputs you personally tested;
- what you changed or rejected;
- that you remain responsible for the final submission.

## Required tests

Add at least two meaningful tests. Examples:

- test that the selected image files load;
- test that keypoints are not empty;
- test that descriptors are created;
- test that the output visualization exists;
- test that the summary contains match counts.

Run:

```powershell
py -m unittest discover -s tests -v
```

## Final verification checklist

Run these commands before submission:

```powershell
py starter_check.py
py your_solution.py
py -m unittest discover -s tests -v
git status
```

Then check manually:

- Can the code run from the lab folder?
- Are all output images saved inside `outputs/`?
- Are the values in the reports copied from the actual run?
- Did you keep Part A evidence separate from Part B evidence?
- Are the limitations and AI declaration complete?
- Did you remove `.venv` and `__pycache__` from the submission?

## One Git and D2L submission

From the root of your private repository:

```powershell
git status
git add week-05/graded-lab-feature-matching
git commit -m "Complete AIDA 2154 Week 5 graded lab"
git push
```

Submit the private repository link or final commit link in D2L. This is one submission worth 5%: Part A is 3% and Part B is 2%.
