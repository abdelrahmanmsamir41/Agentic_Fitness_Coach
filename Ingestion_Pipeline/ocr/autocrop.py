"""Pre-OCR crop: shrink pages that are mostly blank around their real content."""
import contextlib
import tempfile
from pathlib import Path

import pymupdf as fitz

from console import print_step

_MIN_BBOX_RATIO = 0.50  # content bbox under this share of the page -> crop
_SPARSE_BBOX_RATIO = 0.70  # under this AND sparse (see below) -> crop too
_MAX_SPARSE_OCCUPANCY = 0.05  # ink pixels under this share -> bbox is mostly hollow
_PADDING = 20  # points of margin kept around the detected content
_THUMBNAIL_SCALE = 0.3  # low-res render is enough to find the content edges
_INK_THRESHOLD = 250  # grayscale value below which a pixel counts as "ink"


def _measure_content(page):
    matrix = fitz.Matrix(_THUMBNAIL_SCALE, _THUMBNAIL_SCALE)
    pixmap = page.get_pixmap(matrix=matrix, colorspace=fitz.csGRAY, alpha=False)
    width, height, samples = pixmap.width, pixmap.height, pixmap.samples

    ink_rows = [row for row in range(height) if min(samples[row * width:(row + 1) * width]) < _INK_THRESHOLD]
    if not ink_rows:
        return None
    ink_cols = [col for col in range(width)
                if any(samples[row * width + col] < _INK_THRESHOLD for row in range(height))]
    occupancy_ratio = sum(1 for pixel in samples if pixel < _INK_THRESHOLD) / (width * height)

    pixel_bbox = fitz.Rect(min(ink_cols), min(ink_rows), max(ink_cols) + 1, max(ink_rows) + 1)
    return pixel_bbox * ~matrix, occupancy_ratio  # thumbnail pixels -> PDF points


def _should_crop(bbox_ratio, occupancy_ratio):
    if bbox_ratio < _MIN_BBOX_RATIO:
        return True
    return bbox_ratio < _SPARSE_BBOX_RATIO and occupancy_ratio < _MAX_SPARSE_OCCUPANCY


def _crop_sparse_pages(doc):
    """Tightens each sparse page's cropbox to its rendered content; returns the cropped page numbers."""
    cropped_pages = []
    for page in doc:
        metrics = _measure_content(page)
        if metrics is None:
            continue
        bbox, occupancy_ratio = metrics
        if not _should_crop(bbox.get_area() / page.rect.get_area(), occupancy_ratio):
            continue
        padded_bbox = fitz.Rect(bbox.x0 - _PADDING, bbox.y0 - _PADDING,
                            bbox.x1 + _PADDING, bbox.y1 + _PADDING) & page.rect
        # cropbox, not mediabox -- set_mediabox silently resets cropbox to an
        # unrelated offset as a side effect (confirmed against a real PDF).
        page.set_cropbox(padded_bbox)
        cropped_pages.append(page.number)
    return cropped_pages


@contextlib.contextmanager
def cropped_copy(pdf_path):
    """Yields a path to submit for OCR: the original if no page needed cropping, else a cropped temp copy."""
    doc = fitz.open(pdf_path)
    cropped_pages = _crop_sparse_pages(doc)
    if not cropped_pages:
        print_step("✂️ autocrop: no sparse pages found, using original PDF")
        doc.close()
        yield pdf_path
        return

    print_step(f"✂️ autocrop: cropping {len(cropped_pages)} sparse page(s) {cropped_pages}")
    temp_path = tempfile.mktemp(suffix=".pdf")
    doc.save(temp_path)
    doc.close()
    try:
        yield temp_path
    finally:
        Path(temp_path).unlink(missing_ok=True)
