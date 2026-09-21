"""PaddleOCR-VL parse -> merge PyMuPDF links/native-text fallback into pages."""
import asyncio

from ocr import autocrop, native_text, paddle_api
import links


async def parse_and_link(doc_id, pdf_path):
    """parse_pdf (PaddleOCR-VL, network-bound, ~30s/page) and extract_links
    (local PyMuPDF, sub-second) don't depend on each other -- run them
    concurrently. Real time saved is small since the OCR call dominates, but
    there's no reason to block one on the other."""
    print(f"🔍 [{doc_id}] parsing via PaddleOCR-VL + extracting links via PyMuPDF...")
    with autocrop.cropped_copy(str(pdf_path)) as ocr_path:
        pages, links_by_page, native_by_page = await asyncio.gather(
            paddle_api.parse_pdf(ocr_path),  # return list of {"page_index": int, "markdown": str}
            asyncio.to_thread(links.extract_links, str(pdf_path)), # return a dict of {page_index:[{"url": l["uri"], "anchor_text": anchor_text},{...},...]}
            asyncio.to_thread(native_text.extract_native_text, str(pdf_path)),  # return a dict of {page_index: raw_text}
        )
    pages = native_text.prefer_native_when_ocr_diverges(pages, native_by_page)
    pages = native_text.append_missing_native_numbers(pages, native_by_page)
    flagged = native_text.flag_unreliable_pages(pages, native_by_page)
    if flagged:
        print(f"⚠️ [{doc_id}] WARNING: pages {flagged} have no native-text fallback and OCR output looks unreliable")
    return links.merge_links_into_pages(pages, links_by_page)  # return list of {"page_index": int, "markdown": str} with links merged into markdown
