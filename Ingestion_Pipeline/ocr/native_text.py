"""PyMuPDF native-text fallback: only step in when OCR genuinely missed the
page's real content, not whenever native text merely exists -- OCR can
reformat real content into a better structure (e.g. a proper table) than raw
native text ever would, so replacing it outright would be a regression.
"""
import re

import pymupdf as fitz

from console import print_step
from text_utils import NUMBER as _NUMBER

_MIN_NATIVE_LEN = 200  # below this, too little to be a real content page
_WORD = re.compile(r"[A-Za-z؀-ۿ]{4,}")
_MIN_WORD_OVERLAP = 0.5  # below this share of native's real words seen in OCR -> OCR missed the page
_MIN_JUDGEABLE_LEN = 20  # shorter than this, too little to call garbled either way
_MIN_ALPHA_RATIO = 0.3  # below this share of letters in non-whitespace chars -> looks garbled


def extract_native_text(pdf_path):
    # blocks joined with blank lines, not plain get_text() -- that returns
    # single-newline-separated text that markdown folds into one paragraph.
    doc = fitz.open(pdf_path)
    text_by_page = {}
    for page in doc:
        blocks = page.get_text("blocks", sort=True)
        text_by_page[page.number] = "\n\n".join(block[4].strip() for block in blocks if block[4].strip())
    doc.close()
    return text_by_page


def _significant_words(text):
    return {word.lower() for word in _WORD.findall(text)}


def _did_ocr_miss_page(native_page_text, ocr_markdown):
    native_words = _significant_words(native_page_text)
    if not native_words:
        return False
    overlap = len(native_words & _significant_words(ocr_markdown)) / len(native_words)
    return overlap < _MIN_WORD_OVERLAP


def prefer_native_when_ocr_diverges(pages, native_text_by_page):
    swapped_pages = []
    for page in pages:
        native_page_text = native_text_by_page.get(page["page_index"], "").strip()
        if len(native_page_text) < _MIN_NATIVE_LEN:
            continue
        if _did_ocr_miss_page(native_page_text, page["markdown"]):
            page["markdown"] = native_page_text
            swapped_pages.append(page["page_index"])
    if swapped_pages:
        print_step(f"🔀 native text preferred over OCR for pages {swapped_pages} (OCR missed the content)")
    return pages


def _looks_garbled(text):
    visible_chars = [char for char in text if not char.isspace()]
    if len(visible_chars) < _MIN_JUDGEABLE_LEN:
        return False
    return sum(1 for char in visible_chars if char.isalpha()) / len(visible_chars) < _MIN_ALPHA_RATIO


def flag_unreliable_pages(pages, native_text_by_page):
    """No native text to cross-check against (a genuine scan) means no
    fallback exists if OCR garbled this page -- surface it instead of
    silently trusting whatever came back."""
    return [
        page["page_index"] for page in pages
        if len(native_text_by_page.get(page["page_index"], "").strip()) < _MIN_NATIVE_LEN
        and _looks_garbled(page["markdown"])
    ]


def append_missing_native_numbers(pages, native_text_by_page):
    """A page can mix a native-text overlay (the real prescription) with a
    background/template image OCR reads instead -- both are real, so a
    number native has that OCR's output doesn't gets appended, not swapped
    in. No length gate: a short prescription snippet is exactly the case
    this exists for, unlike the whole-page replace above."""
    appended_pages = []
    for page in pages:
        native_page_text = native_text_by_page.get(page["page_index"], "").strip()
        if not native_page_text:
            continue
        missing_numbers = set(_NUMBER.findall(native_page_text)) - set(_NUMBER.findall(page["markdown"]))
        if missing_numbers:
            page["markdown"] += f"\n\n{native_page_text}"
            appended_pages.append(page["page_index"])
    if appended_pages:
        print_step(f"➕ appended missing native numbers for pages {appended_pages}")
    return pages
