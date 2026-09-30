"""OCR ingestion: PaddleOCR-VL parse -> batched Gemini extraction, then the
shared enrich -> chunk -> Mongo -> Qdrant steps in pipeline.py.
Run: python Ingestion_Pipeline/ingest.py
"""
import asyncio
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import pipeline
from extraction.extract import extract_document
from parse import parse_and_link

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "ingestion_output"

DOCUMENTS = [
    (REPO_ROOT / "splits" / "W-1401-260906-0202 - Abdelrhman Mohamed Samer _260909_114328.pdf", "training_plan"),
    # (REPO_ROOT / "Splits" / "belghamdi_splits" / "BelghamdiSplit-Anterior-x-Posterior.pdf", "training_plan"),
    # (REPO_ROOT / "splits" / "shady_splits" / "W-1401-260906-0202 - Abdelrhman Mohamed Samer.pdf", "training_plan"),
    # (REPO_ROOT / "splits" / "D-1401-260906-0202 - Abdelrhman Mohamed Samer _260909_114305.pdf", "diet_plan"),
    # (REPO_ROOT / "splits" / "Recipes E-Book _260909_114355.pdf", "recipe_book"),
]


def _extract_from_page_text(doc_id, document_type, pages, _pdf_path):
    return extract_document(doc_id, document_type, pages)  # OCR extraction works from page text only


if __name__ == "__main__":
    asyncio.run(pipeline.run(DOCUMENTS, parse=parse_and_link, extract=_extract_from_page_text, output_dir=OUTPUT_DIR))
