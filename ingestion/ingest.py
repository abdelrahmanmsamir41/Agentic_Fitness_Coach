"""Orchestrates: PaddleOCR-VL parse -> merge PyMuPDF links -> per-section
normalize+extract -> chunk -> store (Mongo + Qdrant).

Run once for all source documents; the Qdrant collection is wiped and
rebuilt from scratch each run (see store_qdrant.reindex), so this is not
incremental -- fine at this scale (a handful of documents for one client).

Fully async: PaddleOCR-VL parsing, link extraction, Gemini calls, and
Mongo/Qdrant I/O all run through asyncio. Two concurrency choices worth
being explicit about:
- Within one document, all its normalize+extract batches run concurrently
  (capped by _LLM_CONCURRENCY) -- a real speedup, since Gemini calls are
  independent per batch.
- Across documents, ingestion stays sequential (a plain for-loop with
  `await`) rather than gathered -- submitting 3 large-PDF OCR jobs to
  PaddleOCR-VL at once risks the kind of transient 500 we already hit once
  under plain serial load; not worth it without evidence it's needed.
"""
import asyncio
import datetime
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import chunk
import links
import normalize_extract
import paddle_api
import store_mongo
import store_qdrant

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "ingestion_output"

COACH_ID = "coach_self"
CLIENT_ID = "client_self"

BATCH_SIZE = 2
LLM_CONCURRENCY = 4  # cap on concurrent Gemini calls, to avoid rate-limit errors

DOCUMENTS = [
    (REPO_ROOT / "Splits" / "shady_splits" / "W-1401-260906-0202 - Abdelrhman Mohamed Samer.pdf", "training_plan"),
    (REPO_ROOT / "Splits" / "shady_splits" / "D-1401-260906-0202 - Abdelrhman Mohamed Samer.pdf", "diet_plan"),
    (REPO_ROOT / "Splits" / "shady_splits" / "Recipes E-Book.pdf", "recipe_book"),
]

_llm_semaphore = asyncio.Semaphore(LLM_CONCURRENCY)


def _doc_id(pdf_path):
    return re.sub(r"[^a-z0-9]+", "-", pdf_path.stem.lower()).strip("-")


async def _parse_and_link(doc_id, pdf_path):
    """parse_pdf (PaddleOCR-VL, network-bound, ~30s/page) and extract_links
    (local PyMuPDF, sub-second) don't depend on each other -- run them
    concurrently. Real time saved is small since the OCR call dominates, but
    there's no reason to block one on the other."""
    print(f"[{doc_id}] parsing via PaddleOCR-VL + extracting links via PyMuPDF...")
    pages, links_by_page = await asyncio.gather(
        paddle_api.parse_pdf(str(pdf_path)),
        asyncio.to_thread(links.extract_links, str(pdf_path)),
    )
    return links.merge_links_into_pages(pages, links_by_page)


async def _process_batch(doc_id, document_type, common, batch_start, batch):
    async with _llm_semaphore:
        print(f"[{doc_id}] normalize+extract batch: {[s['heading'] for s in batch]}")
        extractions = await normalize_extract.normalize_and_extract_batch(
            document_type, [{"heading": s["heading"], "text": s["text"]} for s in batch]
        )

    parents, leaves = [], []
    for offset, (section, extraction) in enumerate(zip(batch, extractions)):
        parent, leaf_chunks = chunk.build_chunks(doc_id, document_type, section, extraction, batch_start + offset)
        parent.update(common)
        for leaf in leaf_chunks:
            leaf.update(common)
        parents.append(parent)
        leaves.extend(leaf_chunks)
    return parents, leaves


async def ingest_document(pdf_path, document_type, coach_id, client_id):
    doc_id = _doc_id(pdf_path)
    doc_out_dir = OUTPUT_DIR / doc_id
    doc_out_dir.mkdir(parents=True, exist_ok=True)
    pages_cache = doc_out_dir / "pages.json"

    if pages_cache.exists():
        print(f"[{doc_id}] reusing cached parse from {pages_cache}")
        pages = json.loads(pages_cache.read_text(encoding="utf-8"))
    else:
        pages = await _parse_and_link(doc_id, pdf_path)
        pages_cache.write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")

    merged_markdown = "\n\n".join(p["markdown"] for p in pages)
    (doc_out_dir / "merged.md").write_text(merged_markdown, encoding="utf-8")

    effective_date = datetime.date.today().isoformat()
    manifest = {
        "document_id": doc_id,
        "document_type": document_type,
        "source_pdf": str(pdf_path),
        "coach_id": coach_id,
        "client_id": client_id,
        "page_count": len(pages),
        "ingested_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "effective_date": effective_date,
        "active": True,
    }
    (doc_out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    await store_mongo.mark_previous_versions_inactive(client_id, document_type)
    await store_mongo.save_document_manifest(manifest)

    sections = chunk.split_sections(pages)
    common = {
        "coach_id": coach_id,
        "client_id": client_id,
        "active": True,
        "effective_date": effective_date,
        "knowledge_tier": "client_plan",
        "language": "ar-en",
        "source_pdf": pdf_path.name,
    }

    batch_starts = list(range(0, len(sections), BATCH_SIZE))
    batch_results = await asyncio.gather(*(
        _process_batch(doc_id, document_type, common, bs, sections[bs:bs + BATCH_SIZE])
        for bs in batch_starts
    ))

    all_parents, all_leaves = [], []
    for parents, leaves in batch_results:
        all_parents.extend(parents)
        all_leaves.extend(leaves)

    await store_mongo.save_parents_and_entities(all_parents, all_leaves)
    print(f"[{doc_id}] {len(sections)} sections, {len(all_leaves)} leaf chunks, "
          f"{sum(1 for l in all_leaves if l['needs_review'])} flagged for review")
    return all_leaves


async def main():
    all_leaf_chunks = []
    for pdf_path, document_type in DOCUMENTS:
        all_leaf_chunks.extend(await ingest_document(pdf_path, document_type, COACH_ID, CLIENT_ID))

    print(f"Reindexing Qdrant with {len(all_leaf_chunks)} total leaf chunks...")
    await store_qdrant.reindex(all_leaf_chunks)
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
