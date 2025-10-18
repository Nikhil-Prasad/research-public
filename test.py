#!/usr/bin/env python3
"""
Extract 9-digit Tax ID from scanned PDFs with boxed input fields.

Pipeline:
- PDF -> 400 DPI grayscale images
- Preprocess: deskew (Hough), CLAHE, slight sharpen, adaptive threshold
- Anchor detection: fuzzy match lines containing "Enter Tax ID" (client name varies)
- ROI: crop a band under the anchor
- Cell detection: find ~square boxes in the ROI and pick the row with 9 cells
- Per-cell OCR: Tesseract, digits-only, --psm 10
- Output: best 9-digit string per page with confidences

Dependencies (Python):
    pip install pdf2image opencv-python pytesseract numpy

System deps:
    - Tesseract OCR (>= 4.1): https://tesseract-ocr.github.io/tessdoc/Installation.html
    - Poppler (for pdf2image): e.g., `brew install poppler` (macOS), apt-get install poppler-utils (Debian/Ubuntu)

If tesseract isn't on PATH, set env:
    export TESSERACT_CMD=/usr/bin/tesseract
"""

import os
import re
import json
import math
import argparse
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from pdf2image import convert_from_path


# ---------- Configuration knobs ----------
DPI = 400
ANCHOR_PATTERNS = [
    re.compile(r"\benter\s+tax\s*id\b", re.I),
    re.compile(r"\btax\s+id\s+below\b", re.I),
    re.compile(r"\btax\s+identification\b", re.I),
    re.compile(r"\bTIN\b", re.I)
]
MIN_CELL_COUNT = 7      # tolerate partial detection; we'll choose best row
TARGET_CELL_COUNT = 9
MIN_CONF = 60           # per-cell tesseract confidence gate (0-100)
ROI_HEIGHT_FRAC = 0.28  # scan ~28% of page height under the anchor
ROI_TOP_OFFSET = 0.6    # offset in multiples of line height under anchor
DEBUG_DEFAULT = False


# ---------- Data structures ----------
@dataclass
class CellOCR:
    idx: int
    char: str
    conf: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) relative to ROI

@dataclass
class PageResult:
    page_index: int
    anchor_text: Optional[str]
    anchor_bbox: Optional[Tuple[int, int, int, int]]  # (x1,y1,x2,y2)
    digits: List[CellOCR]
    value: Optional[str]
    mean_conf: Optional[float]
    debug_images: List[str]


# ---------- Utilities ----------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def as_gray(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def rotate_image(image, angle_deg):
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle_deg, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def deskew_by_hough(gray: np.ndarray) -> Tuple[np.ndarray, float]:
    # thin edges help Hough be stable
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 180)
    if lines is None:
        return gray, 0.0
    angles = [(theta - np.pi / 2.0) for rho, theta in lines[:, 0]]
    angle = np.degrees(np.median(angles))
    if abs(angle) > 0.05:
        return rotate_image(gray, angle), angle
    return gray, 0.0

def preprocess_page(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Returns (gray_clean, bin_inv, skew_angle)"""
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    gray, skew = deskew_by_hough(gray)

    # Contrast normalize + gentle sharpen
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    g = cv2.GaussianBlur(g, (0, 0), 0.8)
    g = cv2.addWeighted(g, 1.5, cv2.GaussianBlur(g, (0, 0), 1.6), -0.5, 0)

    # Adaptive threshold -> binary inverted (white background=255 -> we return inverted)
    bin_inv = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    return g, bin_inv, skew

def pdf_to_gray_images(pdf_path: str, dpi: int = DPI) -> List[np.ndarray]:
    pages = convert_from_path(pdf_path, dpi=dpi)
    return [as_gray(np.array(p)) for p in pages]


# ---------- OCR helpers ----------
def _tesseract_data(img_for_ocr: np.ndarray, psm=6, whitelist=None):
    cfg = f"--psm {psm} --oem 1"
    if whitelist:
        cfg += f" -c tessedit_char_whitelist={whitelist}"
    return pytesseract.image_to_data(img_for_ocr, output_type=Output.DICT, config=cfg)

def _gather_lines(data):
    """Group tesseract tokens into lines with bounding boxes."""
    lines = {}
    n = len(data["text"])
    for i in range(n):
        if int(data["conf"][i]) < 0:
            continue
        key = (data["page_num"][i], data["block_num"][i], data["par_num"][i], data["line_num"][i])
        if key not in lines:
            lines[key] = {
                "text": [],
                "lefts": [],
                "tops": [],
                "rights": [],
                "bottoms": []
            }
        txt = data["text"][i]
        if txt is None:
            continue
        lines[key]["text"].append(txt)
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        lines[key]["lefts"].append(x)
        lines[key]["tops"].append(y)
        lines[key]["rights"].append(x + w)
        lines[key]["bottoms"].append(y + h)

    out = []
    for key, v in lines.items():
        txt = " ".join([t for t in v["text"] if t]).strip()
        if not txt:
            continue
        x1, y1 = min(v["lefts"]), min(v["tops"])
        x2, y2 = max(v["rights"]), max(v["bottoms"])
        line_h = np.median(np.array(v["bottoms"]) - np.array(v["tops"]))
        out.append({"text": txt, "bbox": (x1, y1, x2, y2), "line_h": line_h})
    return out

def normalize_text(s: str) -> str:
    # lowercase, collapse spaces, strip punctuation-ish characters
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def find_anchor(gray_for_ocr: np.ndarray) -> Optional[Tuple[Tuple[int,int,int,int], str, float]]:
    """
    Return (bbox(x1,y1,x2,y2), line_text, line_height) for the first matching anchor.
    """
    data = _tesseract_data(gray_for_ocr, psm=6)
    lines = _gather_lines(data)

    # score by regex matches on normalized text
    best = None
    best_idx = None
    for i, line in enumerate(lines):
        norm = normalize_text(line["text"])
        if any(p.search(norm) for p in ANCHOR_PATTERNS):
            # prefer the earliest match on the page
            best = (line["bbox"], line["text"], float(line["line_h"]))
            best_idx = i
            break

    # light fallback: token overlap with "enter tax id"
    if best is None:
        target_tokens = set("enter tax id below for".split())
        max_overlap = 0
        for i, line in enumerate(lines):
            tokens = set(normalize_text(line["text"]).split())
            overlap = len(tokens & target_tokens)
            if overlap >= 3 and overlap > max_overlap:
                best = (line["bbox"], line["text"], float(line["line_h"]))
                best_idx = i
                max_overlap = overlap

    return best  # or None


# ---------- Cell detection & OCR ----------
def crop_roi(binv_page: np.ndarray, anchor_bbox: Tuple[int,int,int,int], line_h: float) -> Tuple[np.ndarray, Tuple[int,int]]:
    x1, y1, x2, y2 = anchor_bbox
    h, w = binv_page.shape[:2]
    dy_top = int(ROI_TOP_OFFSET * line_h)
    y_top = min(max(y2 + dy_top, 0), h - 1)
    roi_h = int(ROI_HEIGHT_FRAC * h)
    y_bot = min(h, y_top + max(roi_h, int(6 * line_h)))  # guard lower bound
    roi = binv_page[y_top:y_bot, 0:w]
    return roi, (0, y_top)  # ROI origin in page coords

def detect_cell_boxes(roi_binv: np.ndarray) -> List[Tuple[int,int,int,int]]:
    """
    Detect candidate cell boxes inside ROI; return list of (x,y,w,h) in ROI coords.
    Strategy: light line suppression, then contour filter by aspect/area; cluster by rows later.
    """
    H, W = roi_binv.shape[:2]

    # Gentle line removal to prevent strokes fusing with digits
    h_len = max(15, W // 40)
    v_len = max(15, H // 12)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    horiz = cv2.morphologyEx(roi_binv, cv2.MORPH_OPEN, h_kernel, iterations=1)
    vert  = cv2.morphologyEx(roi_binv, cv2.MORPH_OPEN, v_kernel, iterations=1)
    lines = cv2.add(horiz, vert)
    cleaned = cv2.subtract(roi_binv, lines)

    # Slight close to join fragmented edges
    closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    areas = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        # plausible cell filters (scale-invariant)
        if w < 12 or h < 12:
            continue
        ar = w / float(h)
        if 0.6 <= ar <= 1.8 and 150 <= area <= 8000:
            boxes.append((x, y, w, h))
            areas.append(area)

    # tighten by area consistency
    if len(areas) >= 3:
        med = np.median(areas)
        boxes = [b for b, a in zip(boxes, areas) if 0.4 * med <= a <= 2.5 * med]

    # sort canonical
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes

def group_rows(boxes: List[Tuple[int,int,int,int]]) -> List[List[Tuple[int,int,int,int]]]:
    """Group boxes into rows using y proximity to median height."""
    if not boxes:
        return []
    heights = [h for _,_,_,h in boxes]
    h_med = max(1, int(np.median(heights)))
    rows = []
    current = [boxes[0]]
    for b in boxes[1:]:
        if abs(b[1] - current[-1][1]) <= h_med * 0.6:
            current.append(b)
        else:
            rows.append(current)
            current = [b]
    rows.append(current)
    # sort each row left->right
    rows = [sorted(r, key=lambda b: b[0]) for r in rows]
    return rows

def choose_best_row(rows: List[List[Tuple[int,int,int,int]]], anchor_y_in_roi: int) -> Optional[List[Tuple[int,int,int,int]]]:
    """
    Pick the row closest to the anchor and with size close to TARGET_CELL_COUNT.
    """
    if not rows:
        return None
    scored = []
    for r in rows:
        ys = [y for _, y, _, _ in r]
        row_y = int(np.median(ys))
        size = len(r)
        size_penalty = abs(size - TARGET_CELL_COUNT) * 5
        dist = abs(row_y - anchor_y_in_roi)
        score = dist + size_penalty * 50  # prioritize correct count over distance
        scored.append((score, r))
    scored.sort(key=lambda x: x[0])
    best = scored[0][1]
    # trim or pad to 9 by picking the most evenly spaced
    if len(best) >= TARGET_CELL_COUNT:
        return best[:TARGET_CELL_COUNT]
    if len(best) >= MIN_CELL_COUNT:
        return best  # caller can attempt to fill via OCR/fallbacks
    return None

def ocr_digit_cell(cell_binv: np.ndarray) -> Tuple[str, float]:
    """Return (digit_char, confidence). Empty char '' if not a digit."""
    # invert to black text on white; pad
    cell = 255 - cell_binv
    cell = cv2.copyMakeBorder(cell, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=255)

    data = _tesseract_data(cell, psm=10, whitelist="0123456789")
    n = len(data["text"])
    best_char, best_conf = "", -1.0
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        conf = float(data["conf"][i]) if data["conf"][i] != "-1" else -1.0
        if len(txt) == 1 and txt.isdigit() and conf > best_conf:
            best_char, best_conf = txt, conf

    # fallback: Otsu binarize then retry once
    if best_conf < MIN_CONF:
        thr = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        data2 = _tesseract_data(thr, psm=10, whitelist="0123456789")
        for i in range(len(data2["text"])):
            txt = (data2["text"][i] or "").strip()
            conf = float(data2["conf"][i]) if data2["conf"][i] != "-1" else -1.0
            if len(txt) == 1 and txt.isdigit() and conf > best_conf:
                best_char, best_conf = txt, conf

    return (best_char if best_char.isdigit() else "", best_conf if best_conf >= 0 else 0.0)


# ---------- Main per-page routine ----------
def process_page(page_gray: np.ndarray, page_index: int, debug_dir: Optional[str] = None) -> PageResult:
    debug_paths = []
    gray_clean, bin_inv, skew = preprocess_page(page_gray)

    if debug_dir:
        ensure_dir(debug_dir)
        p0 = os.path.join(debug_dir, f"page{page_index:02d}_0_gray_clean.png")
        p1 = os.path.join(debug_dir, f"page{page_index:02d}_1_bin_inv.png")
        cv2.imwrite(p0, gray_clean)
        cv2.imwrite(p1, bin_inv)
        debug_paths += [p0, p1]

    # Use clean (not inverted) image for line OCR
    img_for_ocr = gray_clean  # black-ish text on white
    anchor = find_anchor(img_for_ocr)

    if not anchor:
        return PageResult(page_index, None, None, [], None, None, debug_paths)

    anchor_bbox, anchor_text, line_h = anchor
    roi_binv, roi_origin = crop_roi(bin_inv, anchor_bbox, line_h)

    if debug_dir:
        p2 = os.path.join(debug_dir, f"page{page_index:02d}_2_roi.png")
        cv2.imwrite(p2, roi_binv)
        debug_paths.append(p2)

    boxes = detect_cell_boxes(roi_binv)
    rows = group_rows(boxes)
    # anchor y relative to ROI
    _, ay1, _, ay2 = anchor_bbox
    anchor_y_in_roi = max(0, (ay2 + int(0.5 * line_h)) - roi_origin[1])
    chosen = choose_best_row(rows, anchor_y_in_roi)

    if not chosen:
        return PageResult(page_index, anchor_text, anchor_bbox, [], None, None, debug_paths)

    # ensure left->right and take up to 9
    chosen = sorted(chosen, key=lambda b: b[0])[:TARGET_CELL_COUNT]

    digits: List[CellOCR] = []
    for idx, (x, y, w, h) in enumerate(chosen):
        cell_crop = roi_binv[y:y+h, x:x+w]
        # shrink a touch to avoid border lines intruding
        shrink = max(0, min(w, h) // 12)
        cy1, cy2 = y + shrink, y + h - shrink
        cx1, cx2 = x + shrink, x + w - shrink
        cell_crop = roi_binv[cy1:cy2, cx1:cx2]
        dchar, conf = ocr_digit_cell(cell_crop)
        digits.append(CellOCR(idx=idx, char=dchar, conf=conf, bbox=(x, y, w, h)))

        if debug_dir:
            p_cell = os.path.join(debug_dir, f"page{page_index:02d}_cell{idx}_{dchar or 'blank'}_{int(conf)}.png")
            cv2.imwrite(p_cell, cell_crop)
            debug_paths.append(p_cell)

    # Build final value
    chars = [c.char for c in digits]
    value = "".join([c for c in chars if c.isdigit()])
    mean_conf = float(np.mean([c.conf for c in digits])) if digits else None

    # If we got < 9 digits but rows had >= 9 boxes, it means some OCR failed. We could try micro-retries here;
    # keeping it simple: return what we have and let the caller decide whether to re-run or escalate.
    if len(chars) >= TARGET_CELL_COUNT and len(value) != TARGET_CELL_COUNT:
        # best-effort: put placeholders
        # Keep positions to signal where retries are needed
        value = "".join([c if (c and c.isdigit()) else "?" for c in chars])[:TARGET_CELL_COUNT]

    if len(value) == TARGET_CELL_COUNT and "?" not in value:
        # fully successful
        pass
    else:
        # try a conservative second pass for missing cells: erode/dilate then OCR just those cells
        for i, c in enumerate(digits):
            if c.char and c.char.isdigit() and c.conf >= MIN_CONF:
                continue
            x, y, w, h = c.bbox
            shrink = max(0, min(w, h) // 14)
            cell = roi_binv[y+shrink:y+h-shrink, x+shrink:x+w-shrink]
            # two alternates: slight erosion and slight dilation
            for op in ["erode", "dilate"]:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                alt = cv2.erode(cell, k, iterations=1) if op == "erode" else cv2.dilate(cell, k, iterations=1)
                dchar, conf = ocr_digit_cell(alt)
                if dchar.isdigit() and conf > c.conf:
                    digits[i] = CellOCR(idx=i, char=dchar, conf=conf, bbox=(x,y,w,h))
                    break

        chars = [c.char for c in digits]
        value2 = "".join([c for c in chars if c.isdigit()])
        if len(value2) == TARGET_CELL_COUNT:
            value = value2

    return PageResult(
        page_index=page_index,
        anchor_text=anchor_text,
        anchor_bbox=anchor_bbox,
        digits=digits,
        value=(value if value and len(value) == TARGET_CELL_COUNT and value.isdigit() else None),
        mean_conf=mean_conf,
        debug_images=debug_paths
    )


# ---------- Orchestrator ----------
def extract_tax_ids(pdf_path: str, debug_dir: Optional[str] = None) -> List[PageResult]:
    pages = pdf_to_gray_images(pdf_path, dpi=DPI)
    results: List[PageResult] = []
    for i, pg in enumerate(pages):
        res = process_page(pg, i, debug_dir=debug_dir)
        results.append(res)
    return results


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Extract 9-digit Tax IDs from scanned PDF forms.")
    ap.add_argument("input_pdf", help="Path to input PDF")
    ap.add_argument("--debug", default=str(DEBUG_DEFAULT), help="true/false to save debug crops")
    ap.add_argument("--debug-dir", default="debug_out", help="Directory for debug images (if --debug=true)")
    ap.add_argument("--json-out", default=None, help="Optional path to write JSON results")
    args = ap.parse_args()

    debug = str(args.debug).lower() in ("1", "true", "yes", "y")
    debug_dir = args.debug_dir if debug else None
    if debug_dir and debug:
        ensure_dir(debug_dir)

    results = extract_tax_ids(args.input_pdf, debug_dir=debug_dir)

    # Render console summary
    summary = []
    for r in results:
        status = "OK" if r.value else "MISSING"
        print(f"[page {r.page_index}] anchor={'found' if r.anchor_text else 'none'}  value={r.value or '-'}  mean_conf={r.mean_conf if r.mean_conf is not None else '-'}  [{status}]")
        summary.append({
            "page_index": r.page_index,
            "anchor_text": r.anchor_text,
            "value": r.value,
            "mean_conf": r.mean_conf,
            "digits": [asdict(d) for d in r.digits],
            "debug_images": r.debug_images if debug else []
        })

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote JSON to {args.json_out}")

if __name__ == "__main__":
    # Optional: if Tesseract isn't on PATH, uncomment and set explicitly:
    # pytesseract.pytesseract.tesseract_cmd = os.environ.get("TESSERACT_CMD", "/usr/bin/tesseract")
    main()
