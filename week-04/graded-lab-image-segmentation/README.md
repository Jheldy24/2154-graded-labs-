# AIDA 2154 Week 4 — Graded Lab: Image Segmentation and Morphology

**Individual assessment — one submission — 5% of the course grade**

This lab asks you to build and explain a small image-segmentation workflow for supplied fictional inspection images. You will use Python, NumPy, and OpenCV in VS Code with the supplied files.

## What you are expected to do

You will complete two parts inside this one lab folder:

1. **Part A — guided morphology/segmentation challenge (3%)**: implement the required steps, run your code on the supplied images, save the requested evidence, and explain your choices.
2. **Part B — independent production extension (2%)**: make one useful improvement of your choice, document exactly what changed, and compare the result with Part A.

These are marking components inside one submission. Do not make two submissions.

## Important rules

- Work in your assigned private AIDA 2154 repository.
- Do not push to the instructor `AIDA_2154_Graded_Labs` repository.
- Use only the supplied `data/` images unless your instructor approves another dataset.
- Do not submit credentials, personal photographs, unrelated files, or another student's work.
- Do not copy the completed Week 4 practice answer package. You may use the lecture notes and normal Python/OpenCV documentation.

## Folder contents

```text
graded-lab-image-segmentation/
├── README.md
├── requirements.txt
├── rubric.md
├── data/
│   ├── README.md
│   ├── manifest.csv
│   └── inspection_000.png ... inspection_023.png
├── starter_check.py
├── your_solution.py              <- create this file, or use a notebook
├── outputs/                      <- save your evidence here
├── tests/                        <- add at least two meaningful checks
├── PART-A-REPORT.md              <- create this report
├── PART-B-CHANGE-LOG.md          <- create this report
├── LIMITATIONS.md                <- create this report
└── AI-DECLARATION.md             <- create this report
```

## Step 1 — Copy the lab into your private repository

1. Open VS Code.
2. Open your private AIDA 2154 repository folder.
3. Copy the complete `graded-lab-image-segmentation` folder into that repository.
4. In VS Code, select **File → Open Folder** and open the copied lab folder.
5. In the VS Code Explorer, confirm that you can see `README.md`, `requirements.txt`, `data`, and `starter_check.py`.

## Step 2 — Create the Python environment

Open **Terminal → New Terminal** in VS Code. Confirm the terminal path is the lab folder. Then run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

`requirements.txt` is already supplied in this folder. Do not create a second one.

If PowerShell blocks activation, run the Python commands with the full path instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe starter_check.py
```

## Step 3 — Verify the supplied data

Run:

```powershell
py starter_check.py
```

This check only confirms that the supplied files are present and readable. It is not your solution.

## Part A — Guided challenge (3%)

Create `your_solution.py` or a notebook and complete each item below.

### A1. Load and describe the images

1. Read the images from `data/` with `cv2.imread`.
2. Convert each image to grayscale with `cv2.cvtColor`.
3. Record the image width, height, and grayscale data type.
4. Use the manifest to separate the supplied `train`, `validation`, and `test` groups.
5. Do not mix the groups when reporting results.

### A2. Create a baseline segmentation mask

1. Choose and state a threshold method: a justified fixed threshold, Otsu thresholding, or adaptive thresholding.
2. Create a binary mask where background is `0` and the detected object is `255`.
3. Save at least one baseline mask in `outputs/`.
4. Write the pixel rule in your report. For a fixed threshold it has the form:

   `if pixel >= T: output = 255; otherwise: output = 0`

### A3. Apply morphology

1. Create one structuring element with `cv2.getStructuringElement`.
2. Apply either opening or closing and explain why you selected it.
3. Save the cleaned mask in `outputs/`.
4. State the kernel shape and size.
5. Explain one risk: erosion can remove small valid objects; dilation can add pixels or join nearby objects; opening can remove small regions; closing can fill gaps.

### A4. Inspect and summarize results

1. Count foreground pixels before and after morphology.
2. Count connected components or external contours.
3. Save a visual comparison showing the original image, baseline mask, and cleaned mask.
4. Report the result separately for the validation and test groups.
5. Identify at least one false-positive pattern and one false-negative pattern by visual inspection.

### A5. Write the Part A report

Create `PART-A-REPORT.md` with:

- the business question in one sentence;
- the files and split used;
- the threshold method and exact parameter;
- the morphology operation, kernel shape, and kernel size;
- the requested counts and saved-output file names;
- what improved and what became worse;
- one limitation and one next validation step.

## Part B — Independent production extension (2%)

Choose exactly one meaningful improvement. Examples:

- compare two threshold methods and justify the better choice;
- compare two kernel sizes and explain the object-size trade-off;
- add an illumination-robust step and test it on the `dim` images;
- add a connected-component filter and explain the minimum area;
- add a clear error-overlay image and use it to inspect errors.

Your extension must be your own work and must be different from simply renaming files or changing comments.

Create `PART-B-CHANGE-LOG.md` with:

1. the problem you noticed;
2. the change you made;
3. the exact file and parameter changed;
4. evidence before and after the change;
5. one trade-off or remaining risk;
6. your final recommendation for this bounded teaching dataset.

## Required outputs

Your `outputs/` folder must contain, at minimum:

- one baseline binary mask;
- one post-morphology binary mask;
- one visual comparison or error overlay;
- one CSV or JSON summary of measurements;
- evidence for both Part A and Part B.

## Required quality and responsibility notes

Create these files:

- `LIMITATIONS.md`: explain why this controlled dataset does not prove field or production readiness.
- `AI-DECLARATION.md`: state whether AI tools were used. If used, identify what they helped with and what you personally verified. You remain responsible for the code and interpretation.

## Add and run tests

Add at least two tests in `tests/`. For example, test that the image list is non-empty and that your output mask has the same width and height as its input. Run:

```powershell
py -m unittest discover -s tests -v
```

## One submission

Before submitting:

1. Save all code, reports, tests, and evidence inside this one lab folder.
2. Do not include `.venv`, `__pycache__`, or credentials.
3. Commit and push the lab folder to your private student repository.
4. Submit the private repository link or required commit link in D2L.
5. Confirm that the repository link opens and that the instructor can see the final commit.

Use this Git sequence from the root of your private repository:

```powershell
git status
git add week-04/graded-lab-image-segmentation
git commit -m "Complete AIDA 2154 Week 4 graded lab"
git push
```

This lab has one D2L submission worth 5%: Part A is 3% and Part B is 2%.
