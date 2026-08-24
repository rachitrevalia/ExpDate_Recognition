"""
Simple Streamlit app: take a photo of a product, detect the expiry
date region, read the date, and show the result.

Usage:
    pip install streamlit
    streamlit run streamlit_app.py

This opens in your browser. To test on your phone, make sure your
phone and laptop are on the same WiFi, then open:
    http://<your-laptop-ip>:8501
on your phone's browser (find your laptop's IP with `ipconfig` on
Windows, look for "IPv4 Address"). Note: phone browsers may require
HTTPS to allow camera access from anything other than localhost --
if the camera doesn't open on your phone, we'll set up a quick HTTPS
tunnel (e.g. ngrok) as a next step.
"""

import cv2
import numpy as np
import re
import streamlit as st
from datetime import datetime
from PIL import Image
from ultralytics import YOLO
from paddleocr import PaddleOCR

import os

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")
)

st.set_page_config(page_title="Expiry Date Scanner", layout="centered")

DATE_PATTERNS = [
    ("%Y.%m.%d", r"\d{4}\.\d{2}\.\d{2}"),
    ("%Y/%m/%d", r"\d{4}/\d{2}/\d{2}"),
    ("%Y-%m-%d", r"\d{4}-\d{2}-\d{2}"),
    ("%d/%m/%Y", r"\d{2}/\d{2}/\d{4}"),
    ("%d.%m.%Y", r"\d{2}\.\d{2}\.\d{4}"),
    ("%d-%m-%Y", r"\d{2}-\d{2}-\d{4}"),
    ("%m/%Y", r"\d{2}/\d{4}"),
    ("%m.%Y", r"\d{2}\.\d{4}"),
    ("%y.%m.%d", r"\d{2}\.\d{2}\.\d{2}"),
    ("%d/%m/%y", r"\d{2}/\d{2}/\d{2}"),
    ("%m/%y", r"\d{2}/\d{2}"),
    ("%m.%y", r"\d{2}\.\d{2}"),
]


def is_blurry(image, threshold=100.0):
    """Estimate how blurry an image is using the variance of the Laplacian
    (a standard, cheap sharpness measure). Lower variance = blurrier.
    Returns True if the image looks too blurry to read reliably."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold


def try_parse_date(text):
    """Attempt to extract a real calendar date from noisy OCR text.
    Returns a datetime object if a plausible date pattern is found,
    otherwise None. Used to compare multiple detected dates and pick
    the latest one (expiry is always after manufacture date)."""
    text = text.strip()
    for fmt, pattern in DATE_PATTERNS:
        match = re.search(pattern, text)
        if match:
            try:
                return datetime.strptime(match.group(), fmt)
            except ValueError:
                continue
    return None


@st.cache_resource
def load_models():
    """Load the detection model and OCR engine once, and reuse them
    across app interactions instead of reloading on every photo."""
    model = YOLO(MODEL_PATH)
    ocr = PaddleOCR(use_textline_orientation=True, lang="en", enable_mkldnn=False)
    return model, ocr


def run_ocr(ocr, crop_img):
    """Run OCR on a cropped image, supporting both old and new PaddleOCR APIs."""
    recognized_text = ""

    if hasattr(ocr, "predict"):
        try:
            ocr_result = ocr.predict(crop_img)
            pieces = []
            for res in ocr_result:
                texts = res.get("rec_texts") if hasattr(res, "get") else None
                if texts:
                    pieces.extend(texts)
            recognized_text = " ".join(pieces)
        except (AttributeError, TypeError):
            pass

    if not recognized_text and hasattr(ocr, "ocr"):
        ocr_result = ocr.ocr(crop_img)
        if ocr_result and ocr_result[0]:
            pieces = [line[1][0] for line in ocr_result[0]]
            recognized_text = " ".join(pieces)

    return recognized_text


st.title("Expiry Date Scanner")
st.write("Take a photo of a product to find and read its expiry date.")

model, ocr = load_models()

photo = st.camera_input("Scan a product")

if photo is not None:
    # Convert the uploaded photo into an OpenCV image
    image = Image.open(photo).convert("RGB")
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    if is_blurry(img_bgr):
        st.warning("This photo looks a bit blurry. Try holding steady, "
                   "getting closer, or improving lighting, then take another photo.")
        st.image(image, caption="Blurry photo")
        st.stop()

    with st.spinner("Scanning..."):
        results = model.predict(source=img_bgr, imgsz=800, conf=0.5, verbose=False)
        result = results[0]

        candidates = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            if cls_name != "date" or conf < 0.5:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            pad = 8
            img_h, img_w = img_bgr.shape[:2]
            px1 = max(0, x1 - pad)
            py1 = max(0, y1 - pad)
            px2 = min(img_w, x2 + pad)
            py2 = min(img_h, y2 + pad)
            crop = img_bgr[py1:py2, px1:px2]

            crop_h, crop_w = crop.shape[:2]
            if crop_h > 0 and crop_w > 0:
                crop = cv2.resize(crop, (crop_w * 2, crop_h * 2), interpolation=cv2.INTER_CUBIC)

            text = run_ocr(ocr, crop)
            parsed = try_parse_date(text)

            candidates.append({
                "box": (x1, y1, x2, y2),
                "conf": conf,
                "text": text,
                "parsed": parsed,
            })

        if not candidates:
            st.warning("No expiry date found. Try getting closer or improving lighting.")
        else:
            parsed_candidates = [c for c in candidates if c["parsed"] is not None]

            if parsed_candidates:
                best = max(parsed_candidates, key=lambda c: c["parsed"])
            else:
                best = max(candidates, key=lambda c: c["conf"])

            x1, y1, x2, y2 = best["box"]
            date_text = best["text"]
            best_conf = best["conf"]

            display_img = img_bgr.copy()
            for c in candidates:
                color = (0, 255, 0) if c is best else (0, 220, 255)
                bx1, by1, bx2, by2 = c["box"]
                cv2.rectangle(display_img, (bx1, by1), (bx2, by2), color, 3)
            display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)

            st.image(display_img_rgb, caption="Detected region(s)")

            st.markdown("### Result")
            if date_text and best["parsed"] is not None:
                st.success(f"**Date found:** {date_text}")
            elif date_text:
                st.warning(f"Found text, but it doesn't look like a valid date: \"{date_text}\"")
            else:
                st.warning("Found a date region, but couldn't read the text clearly.")
            st.caption(f"Detection confidence: {best_conf * 100:.0f}%")

            if len(candidates) > 1:
                with st.expander(f"Other dates detected ({len(candidates) - 1})"):
                    for c in candidates:
                        if c is not best:
                            st.write(f"- \"{c['text']}\" (confidence: {c['conf']*100:.0f}%)")