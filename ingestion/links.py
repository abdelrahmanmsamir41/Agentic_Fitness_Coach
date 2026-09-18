"""PyMuPDF pass: extract hyperlinks per page (PaddleOCR-VL's API doesn't return these)."""
import sys
from pathlib import Path
import re
import pymupdf as fitz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


_ARABIC_RUN = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF][\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\s]*[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]|[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")


def fix_rtl(s):
    if not s or not _ARABIC_RUN.search(s):
        return s
    return _ARABIC_RUN.sub(lambda m: m.group(0)[::-1], s)

def extract_links(pdf_path):
    doc = fitz.open(pdf_path)
    by_page = {}
    for page in doc:
        links = [l for l in page.get_links() if l.get("uri")]
        if not links:
            continue
        # get_links() follows the PDF's /Annots order, not necessarily reading
        # order -- sort top-to-bottom/left-to-right so link order matches the
        # rows/paragraphs a reader (or the LLM) would associate them with.
        links.sort(key=lambda l: (round(l["from"].y0), l["from"].x0))
        entries = []
        for l in links:
            anchor_text = fix_rtl(" ".join(page.get_textbox(l["from"]).split())) or "None"
            entries.append({"url": l["uri"], "anchor_text": anchor_text})
        by_page[page.number] = entries
    doc.close()
    return by_page


def merge_links_into_pages(pages, links_by_page):

    for page in pages:
        entries = links_by_page.get(page["page_index"])
        if not entries:
            continue
        lines = ["\n_Links on this page:_"]
        lines += [f"- [{e['anchor_text']}]({e['url']})" for e in entries]
        page["markdown"] += "\n" + "\n".join(lines)
    return pages


def _demo():
    pages = [{"page_index": 0, "markdown": "hello"}, {"page_index": 1, "markdown": "world"}]
    links = {0: [{"url": "https://example.com", "anchor_text": "video"}]}
    merged = merge_links_into_pages(pages, links)
    assert "[video](https://example.com)" in merged[0]["markdown"]
    assert merged[1]["markdown"] == "world"
    print("links self-check passed")


if __name__ == "__main__":
    _demo()
