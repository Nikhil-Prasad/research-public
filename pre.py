#!/usr/bin/env python
"""
deepseek_signature_preprocess.py

Preprocess PDFs for DeepSeek-OCR signature-page extraction.

For each page:
  - Uses PyMuPDF to inspect text layout (if any) and estimate density
  - Renders the page at fixed DPI and clamps max image size
  - Produces:
      * header tile (top fraction of page)
      * footer tile (bottom fraction of page)
      * optional vertical footer tiles for dense pages
  - Saves PNGs and emits index.json with metadata

Dependencies:
  pip install pymupdf pillow
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import fitz  # PyMuPDF
from PIL import Image


# ----------------------- CONFIG ---------------------------------

DPI = 220  # 200–240 is a good range
LONG_SIDE_MAX = 1500  # clamp rendered page longest side (px)

# header/footer geometry (fractions of page height)
HEADER_FRACTION = 0.28
FOOTER_FRACTION = 0.40

# density thresholds for digital pages
AREA_THRESHOLD = 0.55
WORD_THRESHOLD = 800

# pixel threshold for "dense" scanned pages (after rendering)
SCAN_PIXEL_THRESHOLD = 1_800_000  # ~ 1340 x 1340

# dense footer tiling
ENABLE_DENSE_FOOTER_TILING = True
FOOTER_TILES_N = 3
FOOTER_TILE_OVERLAP_FRACTION = 0.08
FOOTER_TILE_LONG_SIDE_MAX = 1100


# ----------------------- DATA STRUCTURES -------------------------


@dataclass
class TileInfo:
    kind: str  # "header", "footer", "footer_tile"
    path: str
    width: int
    height: int
    extra: Dict[str, Any]


@dataclass
class PageInfo:
    page_index: int
    pdf_path: str
    is_digital: bool
    density: float
    word_count: int
    is_dense: bool
    page_image_path: str
    tiles: List[TileInfo]


# ----------------------- CORE FUNCTIONS -------------------------


def render_page_to_image(
    page: fitz.Page,
    dpi: int = DPI,
    long_side_max: int = LONG_SIDE_MAX,
) -> Image.Image:
    """Render a PDF page to a grayscale PIL image at target dpi, clamped in size."""
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
    mode = "L"  # 8-bit grayscale

    img = Image.frombytes(mode, [pix.w, pix.h], pix.samples)

    # clamp longest side
    w, h = img.size
    long_side = max(w, h)
    if long_side > long_side_max:
        scale = long_side_max / long_side
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)

    return img


def measure_density_digital(page: fitz.Page) -> Tuple[float, int]:
    """
    Approximate text density and word count for a digital page (has text layer),
    using text blocks geometry.
    """
    page_rect = page.rect
    page_area = page_rect.width * page_rect.height
    if page_area <= 0:
        return 0.0, 0

    blocks = page.get_text("blocks")
    text_area = 0.0
    words = 0

    for b in blocks:
        if len(b) < 5:
            continue
        x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
        if not text or not text.strip():
            continue
        block_w = max(0.0, x1 - x0)
        block_h = max(0.0, y1 - y0)
        text_area += block_w * block_h
        words += len(text.split())

    density = text_area / page_area
    return density, words


def classify_page_density(
    page: fitz.Page,
    rendered_image: Image.Image,
    area_threshold: float = AREA_THRESHOLD,
    word_threshold: int = WORD_THRESHOLD,
    scan_pixel_threshold: int = SCAN_PIXEL_THRESHOLD,
) -> Tuple[bool, bool, float, int]:
    """
    Decide if a page is 'dense'.

    Returns:
        (is_dense, is_digital, density, word_count)
    """
    txt = page.get_text("text") or ""
    is_digital = bool(txt.strip())

    if is_digital:
        density, words = measure_density_digital(page)
        is_dense = (density >= area_threshold) or (words >= word_threshold)
        return is_dense, True, density, words

    # scanned page: use pixel count of rendered image
    w, h = rendered_image.size
    n_pixels = w * h
    is_dense = n_pixels >= scan_pixel_threshold
    return is_dense, False, 0.0, 0


def crop_header_footer(image: Image.Image) -> Tuple[Image.Image, Image.Image]:
    """Return header and footer crops from the full-page image."""
    w, h = image.size
    header_h = int(HEADER_FRACTION * h)
    footer_h = int(FOOTER_FRACTION * h)

    header = image.crop((0, 0, w, header_h))
    footer = image.crop((0, max(0, h - footer_h), w, h))
    return header, footer


def split_vertical(
    image: Image.Image,
    n: int = FOOTER_TILES_N,
    overlap_fraction: float = FOOTER_TILE_OVERLAP_FRACTION,
    long_side_max: int = FOOTER_TILE_LONG_SIDE_MAX,
) -> List[Image.Image]:
    """Split an image into n vertical tiles with horizontal overlap."""
    w, h = image.size
    tiles: List[Image.Image] = []
    step = w / max(1, n)

    for i in range(n):
        x0 = int(max(0, i * step - overlap_fraction * w))
        x1 = int(min(w, (i + 1) * step + overlap_fraction * w))
        tile = image.crop((x0, 0, x1, h))

        # clamp tile size
        tw, th = tile.size
        long_side = max(tw, th)
        if long_side > long_side_max:
            scale = long_side_max / long_side
            new_w = int(tw * scale)
            new_h = int(th * scale)
            tile = tile.resize((new_w, new_h), Image.BILINEAR)

        tiles.append(tile)

    return tiles


def save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(path), format="PNG")


def preprocess_pdf(
    pdf_path: str,
    out_dir: str,
    dpi: int = DPI,
) -> List[PageInfo]:
    """
    Main entry point: preprocess all pages of a PDF.

    Returns:
        list of PageInfo describing generated tiles.
    """
    pdf_path = str(pdf_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    pages_info: List[PageInfo] = []
    pdf_name = Path(pdf_path).stem

    for i, page in enumerate(doc):
        # 1. Render page
        page_img = render_page_to_image(page, dpi=dpi, long_side_max=LONG_SIDE_MAX)
        page_rel = f"{pdf_name}_page_{i+1:03d}.png"
        page_path = out / page_rel
        save_image(page_img, page_path)

        # 2. Classify density
        is_dense, is_digital, density, words = classify_page_density(page, page_img)

        # 3. Crop header + footer
        header_img, footer_img = crop_header_footer(page_img)

        header_rel = f"{pdf_name}_page_{i+1:03d}_header.png"
        footer_rel = f"{pdf_name}_page_{i+1:03d}_footer.png"
        header_path = out / header_rel
        footer_path = out / footer_rel

        save_image(header_img, header_path)
        save_image(footer_img, footer_path)

        tiles: List[TileInfo] = []

        tiles.append(
            TileInfo(
                kind="header",
                path=header_rel,
                width=header_img.size[0],
                height=header_img.size[1],
                extra={},
            )
        )
        tiles.append(
            TileInfo(
                kind="footer",
                path=footer_rel,
                width=footer_img.size[0],
                height=footer_img.size[1],
                extra={"dense_page": is_dense},
            )
        )

        # 4. Optional: extra vertical footer tiles for dense pages
        if is_dense and ENABLE_DENSE_FOOTER_TILING:
            footer_tiles = split_vertical(footer_img)
            for j, timg in enumerate(footer_tiles):
                t_rel = f"{pdf_name}_page_{i+1:03d}_footer_tile_{j+1}.png"
                t_path = out / t_rel
                save_image(timg, t_path)
                tiles.append(
                    TileInfo(
                        kind="footer_tile",
                        path=t_rel,
                        width=timg.size[0],
                        height=timg.size[1],
                        extra={"tile_index": j + 1},
                    )
                )

        page_info = PageInfo(
            page_index=i,
            pdf_path=pdf_path,
            is_digital=is_digital,
            density=density,
            word_count=words,
            is_dense=is_dense,
            page_image_path=page_rel,
            tiles=tiles,
        )
        pages_info.append(page_info)

    doc.close()
    return pages_info


def write_index(pages: List[PageInfo], out_dir: str) -> None:
    out = Path(out_dir)
    index_path = out / "index.json"
    data = [asdict(p) for p in pages]
    index_path.write_text(json.dumps(data, indent=2))


# ----------------------- CLI ------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess PDFs for DeepSeek-OCR signature pages."
    )
    parser.add_argument("pdf", help="Path to input PDF")
    parser.add_argument(
        "-o",
        "--out",
        help="Output directory for images + index.json",
        default="preprocessed",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DPI,
        help=f"DPI for rendering (default {DPI})",
    )

    args = parser.parse_args()

    pages = preprocess_pdf(args.pdf, args.out, dpi=args.dpi)
    write_index(pages, args.out)

    print(f"Processed {len(pages)} pages into {args.out}")


if __name__ == "__main__":
    main()
