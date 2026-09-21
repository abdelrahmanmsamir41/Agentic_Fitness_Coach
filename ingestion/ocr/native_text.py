"""PyMuPDF native-text fallback: only step in when OCR genuinely missed the
page's real content, not whenever native text merely exists -- OCR can
reformat real content into a better structure (e.g. a proper table) than raw
native text ever would, so replacing it outright would be a regression.
"""
import re

import pymupdf as fitz

_MIN_NATIVE_LEN = 200  # below this, too little to be a real content page
_WORD = re.compile(r"[A-Za-z؀-ۿ]{4,}")
_MIN_WORD_OVERLAP = 0.5  # below this share of native's real words seen in OCR -> OCR missed the page
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_MIN_JUDGEABLE_LEN = 20  # shorter than this, too little to call garbled either way
_MIN_ALPHA_RATIO = 0.3  # below this share of letters in non-whitespace chars -> looks garbled


def extract_native_text(pdf_path):
    # blocks joined with blank lines, not plain get_text() -- that returns
    # single-newline-separated text that markdown folds into one paragraph.
    doc = fitz.open(pdf_path)
    by_page = {}
    for page in doc:
        blocks = page.get_text("blocks", sort=True)
        by_page[page.number] = "\n\n".join(b[4].strip() for b in blocks if b[4].strip())
    doc.close()
    return by_page


def _significant_words(text):
    return {w.lower() for w in _WORD.findall(text)}


def _ocr_missed_the_page(native, ocr_markdown):
    native_words = _significant_words(native)
    if not native_words:
        return False
    overlap = len(native_words & _significant_words(ocr_markdown)) / len(native_words)
    return overlap < _MIN_WORD_OVERLAP


def prefer_native_when_ocr_diverges(pages, native_text_by_page):
    swapped = []
    for page in pages:
        native = native_text_by_page.get(page["page_index"], "").strip()
        if len(native) < _MIN_NATIVE_LEN:
            continue
        if _ocr_missed_the_page(native, page["markdown"]):
            page["markdown"] = native
            swapped.append(page["page_index"])
    if swapped:
        print(f"🔀 native text preferred over OCR for pages {swapped} (OCR missed the content)")
    return pages


def _looks_garbled(text):
    chars = [c for c in text if not c.isspace()]
    if len(chars) < _MIN_JUDGEABLE_LEN:
        return False
    return sum(1 for c in chars if c.isalpha()) / len(chars) < _MIN_ALPHA_RATIO


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
    appended = []
    for page in pages:
        native = native_text_by_page.get(page["page_index"], "").strip()
        if not native:
            continue
        missing = set(_NUMBER.findall(native)) - set(_NUMBER.findall(page["markdown"]))
        if missing:
            page["markdown"] += f"\n\n{native}"
            appended.append(page["page_index"])
    if appended:
        print(f"➕ appended missing native numbers for pages {appended}")
    return pages
