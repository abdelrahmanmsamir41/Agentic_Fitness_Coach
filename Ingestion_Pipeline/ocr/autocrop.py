"""Pre-OCR crop: shrink pages that are mostly blank around their real content."""
import contextlib
import tempfile
from pathlib import Path

import pymupdf as fitz

_MIN_BBOX_RATIO = 0.50  # content bbox under this share of the page -> crop
_SPARSE_BBOX_RATIO = 0.70  # under this AND sparse (see below) -> crop too
_MAX_SPARSE_OCCUPANCY = 0.05  # ink pixels under this share -> bbox is mostly hollow
_PADDING = 20  # points of margin kept around the detected content
_THUMBNAIL_SCALE = 0.3  # low-res render is enough to find the content edges
_INK_THRESHOLD = 250  # grayscale value below which a pixel counts as "ink"


def _content_metrics(page):
    matrix = fitz.Matrix(_THUMBNAIL_SCALE, _THUMBNAIL_SCALE)
    pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csGRAY, alpha=False)
    w, h, samples = pix.width, pix.height, pix.samples

    ink_rows = [y for y in range(h) if min(samples[y * w:(y + 1) * w]) < _INK_THRESHOLD]
    if not ink_rows:
        return None
    ink_cols = [x for x in range(w) if any(samples[y * w + x] < _INK_THRESHOLD for y in range(h))]
    occupancy_ratio = sum(1 for b in samples if b < _INK_THRESHOLD) / (w * h)

    px_bbox = fitz.Rect(min(ink_cols), min(ink_rows), max(ink_cols) + 1, max(ink_rows) + 1)
    return px_bbox * ~matrix, occupancy_ratio  # thumbnail pixels -> PDF points


def _should_crop(bbox_ratio, occupancy_ratio):
    if bbox_ratio < _MIN_BBOX_RATIO:
        return True
    return bbox_ratio < _SPARSE_BBOX_RATIO and occupancy_ratio < _MAX_SPARSE_OCCUPANCY


@contextlib.contextmanager
def cropped_copy(pdf_path):
    """Yields a path to submit for OCR -- the original path unchanged if no
    page needed cropping, otherwise a temp copy with the sparse pages'
    cropbox tightened to their real (rendered) content."""
    doc = fitz.open(pdf_path)
    cropped_pages = []
    for page in doc:
        metrics = _content_metrics(page)
        if metrics is None:
            continue
        bbox, occupancy_ratio = metrics
        if not _should_crop(bbox.get_area() / page.rect.get_area(), occupancy_ratio):
            continue
        padded = fitz.Rect(bbox.x0 - _PADDING, bbox.y0 - _PADDING,
                            bbox.x1 + _PADDING, bbox.y1 + _PADDING) & page.rect
        # cropbox, not mediabox -- set_mediabox silently resets cropbox to an
        # unrelated offset as a side effect (confirmed against a real PDF).
        page.set_cropbox(padded)
        cropped_pages.append(page.number)

    if not cropped_pages:
        print("✂️ autocrop: no sparse pages found, using original PDF")
        doc.close()
        yield pdf_path
        return

    print(f"✂️ autocrop: cropping {len(cropped_pages)} sparse page(s) {cropped_pages}")
    tmp_path = tempfile.mktemp(suffix=".pdf")
    doc.save(tmp_path)
    doc.close()
    try:
        yield tmp_path
    finally:
        Path(tmp_path).unlink(missing_ok=True)
