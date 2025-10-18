# pdf_tin_locator.py
# DI-free pipeline: take a PDF and caller-provided anchor boxes, isolate the
# 9-box grid below each anchor using OpenCV only. No OCR in this module.
#
# You bring the anchor detection (e.g., Azure Document Intelligence). This module:
#   - Renders PDF pages to BGR (PyMuPDF)
#   - Given anchor bbox(es), finds the 9-cell grid region via morphology
#   - Returns cropped, cleaned grayscale grid images (+ optional crop of the anchor text line)
#
# Public entry points:
#   - extract_grids_from_pdf(pdf, anchors_provider, ...)
#     anchors_provider: Callable[[np.ndarray, int], Sequence[AnchorLike]]
#   - extract_grids_from_pdf_with_anchors(pdf, anchors_by_page, ...)
#     anchors_by_page: Dict[page_index, Sequence[AnchorLike]]
#   - extract_grids_from_page(page_bgr, anchors, ...)
#
# AnchorLike: either Anchor dataclass or a 4-tuple (x, y, w, h) or dict {"bbox": (x,y,w,h), "text": ...}

from __future__ import annotations

import io
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

# Standalone rendering and OpenCV grid localization (no external imports)

def pdf_page_to_bgr(pdf: Union[str, bytes, io.BytesIO], page_index: int = 0, dpi: int = 300) -> np.ndarray:
    """Render a single PDF page to a BGR numpy array using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
    except Exception as e:
        raise RuntimeError("PyMuPDF not available. Install 'pymupdf' to render PDFs.") from e

    if isinstance(pdf, (bytes, bytearray)):
        doc = fitz.open(stream=pdf, filetype="pdf")
    elif isinstance(pdf, io.IOBase):
        data = pdf.read()
        doc = fitz.open(stream=data, filetype="pdf")
    else:
        doc = fitz.open(pdf)

    try:
        page = doc.load_page(page_index)
        zoom = float(dpi) / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:  # grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img
    finally:
        doc.close()


def prepare_tin_cells_opencv(
    page_bgr: np.ndarray,
    anchor_bbox: Tuple[int, int, int, int],
    roi_y_min_pct: float = 0.01,
    roi_y_max_pct: float = 0.18,
    equal_split_fallback: bool = True,
    upscale: float = 2.0,
) -> Dict[str, Any]:
    """OpenCV-only pipeline that localizes the 9-box TIN grid and prepares per-cell crops."""
    page_h, page_w = page_bgr.shape[:2]
    ax, ay, aw, ah = anchor_bbox

    # Define ROI band below the anchor
    y0 = ay + ah
    y1 = int(min(page_h - 1, max(0, y0 + roi_y_min_pct * page_h)))
    y2 = int(min(page_h - 1, max(y1 + 10, y0 + roi_y_max_pct * page_h)))
    x1 = int(0.05 * page_w)
    x2 = int(0.95 * page_w)
    roi_rect = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))

    roi_bgr = _crop(page_bgr, roi_rect)
    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # Binarize and detect grid lines
    th_bin = _adaptive_binarize(roi_gray)
    vmask, hmask = _vertical_horizontal_masks(th_bin)

    # Grid boundary detection
    x_lines = _cluster_verticals(vmask)
    y_top, y_bottom = _top_bottom_from_horizontal(hmask, roi_gray.shape[0])

    # Inpaint borders on grayscale ROI so OCR never sees them
    border_mask = cv2.bitwise_or(vmask, hmask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    border_mask = cv2.dilate(border_mask, kernel, iterations=1)
    roi_clean = cv2.inpaint(roi_gray, border_mask, 3, cv2.INPAINT_TELEA)

    # Compute cell rectangles (ROI-local)
    cells_rects_roi: List[Tuple[int, int, int, int]] = []
    if len(x_lines) >= 10 and y_bottom - y_top > 10:
        x_lines = sorted(x_lines)
        for i in range(9):
            lx = int(x_lines[i])
            rx = int(x_lines[i + 1])
            pad_x = max(1, int(0.10 * (rx - lx)))
            pad_y = max(1, int(0.12 * (y_bottom - y_top)))
            cells_rects_roi.append(
                (
                    lx + pad_x,
                    y_top + pad_y,
                    max(1, (rx - lx) - 2 * pad_x),
                    max(1, (y_bottom - y_top) - 2 * pad_y),
                )
            )
    elif equal_split_fallback:
        lx = 0
        rx = roi_gray.shape[1]
        y_top2 = int(0.15 * roi_gray.shape[0])
        y_bot2 = int(0.85 * roi_gray.shape[0])
        for i in range(9):
            xL = int(lx + i * (rx - lx) / 9.0)
            xR = int(lx + (i + 1) * (rx - lx) / 9.0)
            pad_x = max(1, int(0.10 * (xR - xL)))
            pad_y = max(1, int(0.12 * (y_bot2 - y_top2)))
            cells_rects_roi.append(
                (
                    xL + pad_x,
                    y_top2 + pad_y,
                    max(1, (xR - xL) - 2 * pad_x),
                    max(1, (y_bot2 - y_top2) - 2 * pad_y),
                )
            )

    # Build crops (upscaled grayscale)
    crops_gray: List[np.ndarray] = []
    for (cx, cy, cw, ch) in cells_rects_roi:
        crop = roi_clean[cy : cy + ch, cx : cx + cw]
        if crop.size == 0:
            crops_gray.append(np.zeros((1, 1), dtype=np.uint8))
            continue
        if upscale and upscale != 1.0:
            crop = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_LANCZOS4)
        crops_gray.append(crop)

    # Convert ROI-local rects to page coords
    rx, ry, rw, rh = roi_rect
    cells_rects_page = [(rx + cx, ry + cy, cw, ch) for (cx, cy, cw, ch) in cells_rects_roi]

    debug = {
        "roi_rect": roi_rect,
        "grid_boundaries": {"x_lines": x_lines, "y_top": y_top, "y_bottom": y_bottom},
        "vmask_sum": int(vmask.sum()),
        "hmask_sum": int(hmask.sum()),
    }

    return {
        "roi_rect": roi_rect,
        "cells_rects_roi": cells_rects_roi,
        "cells_rects_page": cells_rects_page,
        "crops_gray": crops_gray,
        "debug": debug,
    }


def draw_grid_overlay(
    page_bgr: np.ndarray,
    roi_rect: Tuple[int, int, int, int],
    x_lines: List[int],
    y_top: int,
    y_bottom: int,
) -> np.ndarray:
    """Draw only the grid overlay using OpenCV artifacts."""
    out = page_bgr.copy()
    x, y, w, h = roi_rect
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for xl in (x_lines or []):
        cv2.line(out, (x + int(xl), y + int(y_top)), (x + int(xl), y + int(y_bottom)), (0, 0, 255), 1)
    return out


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class Anchor:
    bbox: Tuple[int, int, int, int]
    text: Optional[str] = None

AnchorLike = Union[
    Anchor,
    Tuple[int, int, int, int],
    Dict[str, Any],  # expects at least {"bbox": (x,y,w,h)} and optional "text"
]


@dataclass
class TinBoxCrop:
    page_index: int
    anchor_text: Optional[str]
    anchor_bbox: Tuple[int, int, int, int]  # page coords (x, y, w, h)
    grid_crop: np.ndarray                   # grayscale, cropped to the 9-box grid (inpainted)
    above_text_crop: Optional[np.ndarray]   # small BGR crop of the anchor line region
    debug: Dict[str, Any]


# ----------------------------
# Public API (no OCR here)
# ----------------------------

def extract_grids_from_pdf(
    pdf: Union[str, bytes, io.BytesIO],
    anchors_provider: Callable[[np.ndarray, int], Sequence[AnchorLike]],
    dpi: int = 300,
    roi_y_min_pct: float = 0.01,
    roi_y_max_pct: float = 0.18,
    equal_split_fallback: bool = True,
    upscale: float = 2.0,
    return_above_text: bool = True,
) -> List[TinBoxCrop]:
    """
    For each page, call anchors_provider(page_bgr, page_index) to obtain anchor boxes,
    then localize and crop the 9-box grid below each anchor.

    anchors_provider should return a list of anchors (AnchorLike):
      - Anchor(bbox=(x,y,w,h), text=...)
      - or (x, y, w, h)
      - or {"bbox": (x,y,w,h), "text": "..."}
    """
    pages = _iter_pdf_pages(pdf, dpi=dpi)
    results: List[TinBoxCrop] = []

    for page_index, page_bgr in enumerate(pages):
        anchors_like = anchors_provider(page_bgr, page_index) or []
        anchors = _normalize_anchors(anchors_like)
        if not anchors:
            continue
        results.extend(
            extract_grids_from_page(
                page_bgr=page_bgr,
                anchors=anchors,
                page_index=page_index,
                roi_y_min_pct=roi_y_min_pct,
                roi_y_max_pct=roi_y_max_pct,
                equal_split_fallback=equal_split_fallback,
                upscale=upscale,
                return_above_text=return_above_text,
            )
        )
    return results


def extract_grids_from_pdf_with_anchors(
    pdf: Union[str, bytes, io.BytesIO],
    anchors_by_page: Dict[int, Sequence[AnchorLike]],
    dpi: int = 300,
    roi_y_min_pct: float = 0.01,
    roi_y_max_pct: float = 0.18,
    equal_split_fallback: bool = True,
    upscale: float = 2.0,
    return_above_text: bool = True,
) -> List[TinBoxCrop]:
    """
    Same as extract_grids_from_pdf but you provide a dict mapping page_index -> anchors.
    """
    # Determine total pages to iterate in order
    page_count = _pdf_page_count(pdf)
    results: List[TinBoxCrop] = []
    for page_index in range(page_count):
        page_bgr = pdf_page_to_bgr(pdf, page_index=page_index, dpi=dpi)
        anchors_like = anchors_by_page.get(page_index, []) or []
        anchors = _normalize_anchors(anchors_like)
        if not anchors:
            continue
        results.extend(
            extract_grids_from_page(
                page_bgr=page_bgr,
                anchors=anchors,
                page_index=page_index,
                roi_y_min_pct=roi_y_min_pct,
                roi_y_max_pct=roi_y_max_pct,
                equal_split_fallback=equal_split_fallback,
                upscale=upscale,
                return_above_text=return_above_text,
            )
        )
    return results


def extract_grids_from_page(
    page_bgr: np.ndarray,
    anchors: Sequence[AnchorLike],
    page_index: int = 0,
    roi_y_min_pct: float = 0.01,
    roi_y_max_pct: float = 0.18,
    equal_split_fallback: bool = True,
    upscale: float = 2.0,
    return_above_text: bool = True,
) -> List[TinBoxCrop]:
    """
    Localize and crop 9-box grids for the provided anchors on a single page image (BGR).
    """
    results: List[TinBoxCrop] = []
    norm_anchors = _normalize_anchors(anchors)

    for a in norm_anchors:
        out = prepare_tin_cells_opencv(
            page_bgr,
            a.bbox,
            roi_y_min_pct=roi_y_min_pct,
            roi_y_max_pct=roi_y_max_pct,
            equal_split_fallback=equal_split_fallback,
            upscale=upscale,
        )

        roi_rect = out["roi_rect"]  # (rx, ry, rw, rh)
        gb = (out.get("debug") or {}).get("grid_boundaries") or {}
        x_lines: List[int] = gb.get("x_lines") or []
        y_top: Optional[int] = gb.get("y_top")
        y_bottom: Optional[int] = gb.get("y_bottom")

        # Compute a conservative grid bounding rect inside ROI (ROI-local coords)
        rx, ry, rw, rh = roi_rect
        if y_top is None or y_bottom is None or y_bottom <= y_top:
            # fallback to interior band (20%..80%) if boundaries unavailable
            y_top = int(0.20 * rh)
            y_bottom = int(0.80 * rh)

        if len(x_lines) >= 2:
            gx0 = int(x_lines[0])
            gx1 = int(x_lines[-1])
        else:
            gx0 = 0
            gx1 = int(rw - 1)

        # Clamp and pad slightly
        pad = 2
        gx0 = max(0, gx0 - pad)
        gx1 = min(rw - 1, gx1 + pad)
        gy0 = max(0, int(y_top) - pad)
        gy1 = min(rh - 1, int(y_bottom) + pad)

        # Build ROI grayscale and inpaint grid region
        roi_bgr = _crop(page_bgr, roi_rect)
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        grid_gray = roi_gray[gy0:gy1, gx0:gx1]
        grid_crop = _inpaint_lines(grid_gray)

        # Optional above-text crop (small band around anchor line)
        above_crop = None
        if return_above_text:
            ax, ay, aw, ah = a.bbox
            above_crop = _crop(
                page_bgr,
                _expand_bbox(ax, ay, aw, ah, page_bgr.shape[1], page_bgr.shape[0], pad=4),
            )

        results.append(
            TinBoxCrop(
                page_index=page_index,
                anchor_text=a.text,
                anchor_bbox=a.bbox,
                grid_crop=grid_crop,
                above_text_crop=above_crop,
                debug={
                    "roi_rect": roi_rect,
                    "grid_roi_rect": (gx0, gy0, max(1, gx1 - gx0), max(1, gy1 - gy0)),
                    "x_lines": x_lines,
                    "y_top": y_top,
                    "y_bottom": y_bottom,
                    "vmask_sum": out.get("debug", {}).get("vmask_sum"),
                    "hmask_sum": out.get("debug", {}).get("hmask_sum"),
                },
            )
        )

    return results


# ----------------------------
# Helpers (no OCR)
# ----------------------------

def _normalize_anchors(anchors: Sequence[AnchorLike]) -> List[Anchor]:
    out: List[Anchor] = []
    for a in anchors:
        if isinstance(a, Anchor):
            out.append(a)
        elif isinstance(a, (tuple, list)) and len(a) == 4:
            out.append(Anchor(bbox=(int(a[0]), int(a[1]), int(a[2]), int(a[3])), text=None))
        elif isinstance(a, dict) and "bbox" in a:
            bbox = a["bbox"]
            text = a.get("text")
            if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
                out.append(Anchor(bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])), text=text))
        # else: ignore invalid entries
    return out


def _iter_pdf_pages(pdf: Union[str, bytes, io.BytesIO], dpi: int = 300) -> Iterable[np.ndarray]:
    """
    Iterate BGR images for each page in the PDF using PyMuPDF.
    """
    count = _pdf_page_count(pdf)
    for page_index in range(count):
        yield pdf_page_to_bgr(pdf, page_index=page_index, dpi=dpi)


def _pdf_page_count(pdf: Union[str, bytes, io.BytesIO]) -> int:
    try:
        import fitz
    except Exception as e:
        raise RuntimeError("PyMuPDF not available. Install 'pymupdf'.") from e

    if isinstance(pdf, (bytes, bytearray)):
        doc = fitz.open(stream=pdf, filetype="pdf")
    elif isinstance(pdf, io.IOBase):
        data = pdf.read()
        doc = fitz.open(stream=data, filetype="pdf")
    else:
        doc = fitz.open(pdf)
    try:
        return doc.page_count
    finally:
        doc.close()


def _crop(img: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = rect
    H, W = img.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)
    if x1 >= x2 or y1 >= y2:
        return np.zeros((1, 1, img.shape[2] if img.ndim == 3 else 1), dtype=img.dtype)
    return img[y1:y2, x1:x2].copy()


def _expand_bbox(x: int, y: int, w: int, h: int, page_w: int, page_h: int, pad: int = 4) -> Tuple[int, int, int, int]:
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(page_w, x + w + pad)
    y2 = min(page_h, y + h + pad)
    return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))


def _adaptive_binarize(gray: np.ndarray) -> np.ndarray:
    try:
        th = cv2.ximgproc.niBlackThreshold(gray, maxValue=255, type=cv2.THRESH_BINARY, blockSize=41, k=-0.2)
    except Exception:
        th = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=35, C=10
        )
    return th


def _vertical_horizontal_masks(th_bin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    bin_inv = cv2.bitwise_not(th_bin)
    h, w = bin_inv.shape[:2]
    v_len = max(15, int(0.60 * h))
    h_len = max(15, int(0.08 * w))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    vmask = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, v_kernel, iterations=1)
    hmask = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, h_kernel, iterations=1)
    vmask = cv2.morphologyEx(vmask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    hmask = cv2.morphologyEx(hmask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    return vmask, hmask


def _cluster_verticals(vmask: np.ndarray) -> List[int]:
    """Get x centers of vertical line components; merge near-duplicates."""
    contours, _ = cv2.findContours(vmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = vmask.shape[:2]
    xs: List[int] = []
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        if hh > 0.55 * h and ww < 0.20 * w:  # tall and thin
            xs.append(int(x + ww / 2))
    if not xs:
        return []
    xs.sort()
    merged = [xs[0]]
    tol = max(3, int(0.015 * w))
    for x in xs[1:]:
        if abs(x - merged[-1]) > tol:
            merged.append(x)
    return merged


def _top_bottom_from_horizontal(hmask: np.ndarray, roi_h: int) -> Tuple[int, int]:
    contours, _ = cv2.findContours(hmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    y_candidates: List[int] = []
    w_total = hmask.shape[1]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 0.60 * w_total and h < int(0.25 * roi_h):
            y_candidates.append(y)
    if len(y_candidates) >= 2:
        ys_sorted = sorted(y_candidates)
        y_top = ys_sorted[0]
        y_bottom = ys_sorted[-1]
        return (int(y_top), int(min(roi_h - 1, y_bottom + 1)))
    return (int(0.20 * roi_h), int(0.80 * roi_h))


def _inpaint_lines(gray: np.ndarray) -> np.ndarray:
    """
    Inpaint grid/border lines inside a grayscale crop to reduce OCR interference.
    """
    if gray.size == 0:
        return gray
    th = _adaptive_binarize(gray)
    vmask, hmask = _vertical_horizontal_masks(th)
    border_mask = cv2.bitwise_or(vmask, hmask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    border_mask = cv2.dilate(border_mask, kernel, iterations=1)
    return cv2.inpaint(gray, border_mask, 3, cv2.INPAINT_TELEA)


# ----------------------------
# CLI (no OCR; requires anchors)
# ----------------------------

def main(argv: List[str]) -> int:
    print(
        "This module does not perform OCR. Provide anchors via extract_grids_from_pdf(..., anchors_provider=...) "
        "or extract_grids_from_pdf_with_anchors(...).",
        file=sys.stderr,
    )
    print("Example sketch:", file=sys.stderr)
    print("""
from pdf_tin_locator import extract_grids_from_pdf, Anchor

def get_anchors(page_bgr, page_index):
    # Replace this with your Azure DI call.
    return [Anchor(bbox=(x, y, w, h), text="Enter your Tax ID below")]

crops = extract_grids_from_pdf("form.pdf", anchors_provider=get_anchors)
for c in crops:
    # c.grid_crop is a grayscale image (np.ndarray) of the 9-box region
    pass
""".strip(), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
