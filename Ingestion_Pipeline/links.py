"""PyMuPDF pass: extract hyperlinks per page (PaddleOCR-VL's API doesn't return these)."""
import sys
from pathlib import Path
import re
import pymupdf as fitz

from console import print_step

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


_ARABIC_RUN = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF][\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\s]*[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]|[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")


def fix_rtl(text):
    if not text or not _ARABIC_RUN.search(text):
        return text
    return _ARABIC_RUN.sub(lambda match: match.group(0)[::-1], text)

def extract_links(pdf_path):
    doc = fitz.open(pdf_path)
    links_by_page = {}
    for page in doc:
        page_links = [link for link in page.get_links() if link.get("uri")]
        if not page_links:
            continue
        # get_links() follows /Annots order, not reading order
        page_links.sort(key=lambda link: (round(link["from"].y0), link["from"].x0))
        entries = []
        for link in page_links:
            anchor_text = fix_rtl(" ".join(page.get_textbox(link["from"]).split())) or "None"
            entries.append({"url": link["uri"], "anchor_text": anchor_text})
        links_by_page[page.number] = entries
    doc.close()
    total_links = sum(len(entries) for entries in links_by_page.values())
    print_step(f"🔗 extracted {total_links} link(s) across {len(links_by_page)} page(s)")
    return links_by_page


def merge_links_into_pages(pages, links_by_page):

    for page in pages:
        entries = links_by_page.get(page["page_index"])
        if not entries:
            continue
        lines = ["\n_Links:"]
        lines += [f"- [{entry['anchor_text']}]({entry['url']})" for entry in entries]
        page["markdown"] += "\n" + "\n".join(lines)
    return pages
