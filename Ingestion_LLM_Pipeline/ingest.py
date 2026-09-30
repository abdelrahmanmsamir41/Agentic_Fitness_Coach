"""LLM-first ingestion: PDF -> Gemini (no OCR), then the shared
enrich -> chunk -> Mongo -> Qdrant steps in Ingestion_Pipeline/pipeline.py.
Only parse and extract differ from Ingestion_Pipeline; everything else is imported from it.
Run: python Ingestion_LLM_Pipeline/ingest.py
"""
import asyncio
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.append(str(Path(__file__).resolve().parent.parent / "Ingestion_Pipeline"))  # after this folder, so local extract wins

import enrich
import extract
import links
import pipeline
from ocr import native_text

enrich.get_model = extract.get_model  # day assignment uses the same capped-thinking model as extraction

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "ingestion_llm_output"

DOCUMENTS = [
    (REPO_ROOT / "Splits" / "shady_splits" / "W-1401-260906-0202 - Abdelrhman Mohamed Samer.pdf", "training_plan"),
]


async def parse_native_text(doc_id, pdf_path):
    # page_text for chunks = PDF's own text layer + links; no OCR
    text_by_page = native_text.extract_native_text(str(pdf_path))
    pages = [{"page_index": page_index, "markdown": text_by_page[page_index]} for page_index in sorted(text_by_page)]
    print(f"📄 [{doc_id}] {len(pages)} pages read from the PDF text layer")
    return links.merge_links_into_pages(pages, links.extract_links(str(pdf_path)))


if __name__ == "__main__":
    asyncio.run(pipeline.run(DOCUMENTS, parse=parse_native_text, extract=extract.extract_document, output_dir=OUTPUT_DIR))
