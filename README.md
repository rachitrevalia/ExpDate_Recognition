# ExpDate_Recognition

A simple app that scans a product photo, detects the expiration date printed
on the packaging, and reads it automatically. Built for Ithina to evaluate
feasibility of automated expiry date capture for retail/pharmacy shelf
management.

## What it does

1. Take a photo of a product using your phone or laptop camera
2. A custom-trained object detection model finds the expiration date
   (and, separately, due/production/batch code markings) on the packaging
3. The detected date region is read using OCR (optical character recognition)
4. If multiple date-like stamps are found on the same package (e.g. a
   manufacture date and a use-by date), the app picks the chronologically
   latest one, since an expiry date is always later than a manufacture date
5. The result is displayed instantly, along with any other candidate dates
   found for transparency

## How it works

```
Photo -> Detect date region (YOLOv11) -> Crop + upscale -> Read text (PaddleOCR)
       -> Parse into a real date -> Pick latest valid date -> Show result
```

## The detection model, in detail

**Architecture:** YOLOv11-small (`yolo11s`), a modern real-time object
detection model. Started from weights pretrained on general images
(COCO), then fine-tuned specifically for this task -- it did not start
from random weights, but it had no prior exposure to expiration dates
before this training.

**Classes the model detects:**
| Class | Meaning |
|---|---|
| `date` | The expiration date (unified from the dataset's `date`/`exp` labels) |
| `due` | A due/best-before style marking |
| `prod` | A production/manufacture date marking |
| `code` | A batch or lot code |

**Training data:** 12,960 images total --
1,102 real photographs of actual product packaging, plus 11,858
synthetically generated product images with computer-rendered dates, both
part of the ExpDate dataset (see Dataset & Citation below). The synthetic
images add variety in fonts, date formats, and backgrounds beyond what the
smaller real-photo set alone provides.

**Evaluation results** (on 665 held-out real test images, never seen
during training):

| Class | Precision | Recall | mAP50 |
|---|---|---|---|
| date | 0.979 | 0.986 | 0.994 |
| due | 0.947 | 0.852 | 0.941 |
| prod | 0.939 | 0.825 | 0.896 |
| code | 0.870 | 0.856 | 0.912 |
| **all classes** | 0.934 | 0.880 | **0.936** |

In plain terms: on the test set, the model correctly finds the actual
expiration date on essentially every image (98.6% of real dates found,
97.9% of the time it's actually a date when it says so).

**A known, real limitation:** in roughly half of real-world test photos,
the model additionally flags the same date region as `due` alongside
`date`, since the two visually resemble each other closely and the
training data has fewer `due` examples than `date` examples. This does not
show up as a formal misclassification in the confusion matrix (both boxes
land on the correct spot), but it does mean the app sometimes has to
choose between two overlapping labels for what is really one date. The
"pick the latest valid date" logic in the app was specifically added to
route around this, since it does not depend on knowing which label is
"correct."

## OCR (text reading) performance

Measured on the same 729 real date crops from the test set:

| Metric | Result |
|---|---|
| Exact string match | 73.7% |
| Normalized match (ignoring punctuation differences, e.g. `.` vs `/`) | 83.0% |
| No text read at all | 5.2% |

A meaningful chunk of the remaining errors were traced to the crop being
too tight around the text, cutting off edge characters -- fixed by adding
a small padding margin around each detected box before running OCR, which
took exact-match accuracy from roughly 24% to 74%.

## Known limitations

- The detection model can occasionally confuse `date` and `due` classes
  when both are visually similar and close together on packaging (see
  above).
- Real-world packaging formats not well represented in the training data
  (e.g. certain regional labeling conventions, dot-matrix stamps mixed with
  adjacent batch codes, or short `MM.YY`-only date stamps) can reduce
  accuracy compared to the dataset's own held-out test results. Several
  such gaps were found and patched during real-device testing (see date
  parsing patterns in `streamlit_app.py`).
- The model has no understanding of printed labels like "M" (manufactured)
  or "U" (use by) -- it treats all date-like stamps the same visually. The
  "pick the latest date" logic compensates for this in most cases, but it
  is a heuristic, not true label comprehension.
- Standard retail barcodes (EAN/UPC) do not encode expiry date information
  -- this app relies entirely on reading the printed date directly, not on
  barcode lookup.
- This app is a functional prototype for internal evaluation, not a
  production-hardened system.

## Running locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`. Requires camera access in your browser.

## Deployment (Railway)

Required files, all in the repo root:
- `streamlit_app.py` -- the app (uses a relative path to find `best.pt`)
- `best.pt` -- trained model weights (~19 MB, included directly in the repo)
- `requirements.txt` -- pinned Python dependencies
- `Procfile` -- start command
- `.python-version` -- pins the Python version Railway builds with

**Required service Variables** (set in Railway's Variables tab, not just
files -- these were needed to get a clean deploy):
| Variable | Value | Why |
|---|---|---|
| `RAILPACK_PYTHON_VERSION` | `3.10` | Ensures compatible wheels are available for the pinned older PaddlePaddle/PaddleOCR versions |
| `RAILPACK_DEPLOY_APT_PACKAGES` | `libgl1 libglib2.0-0` | OpenCV needs these system graphics libraries at runtime, even in "headless" mode |

**Dependency versions are pinned deliberately and are not arbitrary** --
they form a chain of compatibility constraints discovered during
deployment (older PaddleOCR/PaddlePaddle require an older `protobuf`,
which requires an older `streamlit`, which requires an older `pillow`).
Changing any one of these versions without checking the others is likely
to reintroduce a dependency conflict.

## Dataset & Citation

The detection model was trained on the **ExpDate** dataset, created by
researchers at the Korea Institute of Science and Technology (KIST), and
released alongside their 2022 paper. Commercial use permission for this
dataset was granted directly by the dataset authors, on the condition that
their paper is cited and KIST is acknowledged as the dataset's creator
wherever this capability is used or described.

**Citation:**
```bibtex
@article{seker,
    title={A Generalized Framework for Recognition of Expiration Dates on Product Packages Using Fully Convolutional Networks},
    author={Seker, Ahmet Cagatay and Ahn, Sang Chul},
    journal={Expert Systems with Applications},
    volume={203},
    number={117310},
    month={10},
    year={2022},
    doi={10.1016/j.eswa.2022.117310}
}
```

Dataset created by the Korea Institute of Science and Technology (KIST) AI
& Robotics Institute.

## Notes

This is a from-scratch reimplementation trained independently on the
ExpDate dataset, and is not based on or derived from KIST's own reference
implementation or trained model.