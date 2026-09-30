"""PaddleOCR-VL parse -> merge PyMuPDF links/native-text fallback into pages."""
import asyncio

from ocr import autocrop, native_text, paddle_api
import links
from console import print_step


async def parse_and_link(doc_id, pdf_path):
    """OCR, link extraction and native text don't depend on each other, so they run concurrently."""
    print_step(f"🔍 [{doc_id}] parsing via PaddleOCR-VL + extracting links via PyMuPDF...")
    with autocrop.cropped_copy(str(pdf_path)) as ocr_path:
        pages, links_by_page, native_text_by_page = await asyncio.gather(
            paddle_api.parse_pdf(ocr_path),
            asyncio.to_thread(links.extract_links, str(pdf_path)),
            asyncio.to_thread(native_text.extract_native_text, str(pdf_path)),
        )
    pages = native_text.prefer_native_when_ocr_diverges(pages, native_text_by_page)
    pages = native_text.append_missing_native_numbers(pages, native_text_by_page)
    flagged_pages = native_text.flag_unreliable_pages(pages, native_text_by_page)
    if flagged_pages:
        print_step(f"⚠️ [{doc_id}] WARNING: pages {flagged_pages} have no native-text fallback and OCR output looks unreliable")
    return links.merge_links_into_pages(pages, links_by_page)
