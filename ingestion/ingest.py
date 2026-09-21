"""Orchestrates: PaddleOCR-VL parse -> merge PyMuPDF links -> write manifest.

Mongo/Qdrant/Gemini stages are commented out for now (ingest_document
returns right after manifest.json) -- uncomment once Mongo/Qdrant/Gemini
env vars are ready.
"""
import asyncio
import datetime
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from ocr import autocrop, native_text, paddle_api
import links
import chunks
import sections as sections_module
from extraction.extract import normalize_and_extract_batch
import store_mongo
# import store_qdrant

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "ingestion_output"

COACH_ID = "coach_self"
CLIENT_ID = "client_self"

LLM_CONCURRENCY = 4  # cap on concurrent Gemini calls, to avoid rate-limit errors
PAGES_PER_LLM_CALL = 4  # pages batched into one extraction call, instead of one call per page
DOCUMENTS = [
    (REPO_ROOT / "splits" / "W-1401-260906-0202 - Abdelrhman Mohamed Samer _260909_114328.pdf", "training_plan"),
    # (REPO_ROOT / "Splits" / "belghamdi_splits" / "BelghamdiSplit-Anterior-x-Posterior.pdf", "training_plan"),
    # (REPO_ROOT / "splits" / "shady_splits" / "W-1401-260906-0202 - Abdelrhman Mohamed Samer.pdf", "training_plan"),
    # (REPO_ROOT / "splits" / "D-1401-260906-0202 - Abdelrhman Mohamed Samer _260909_114305.pdf", "diet_plan"),
    # (REPO_ROOT / "splits" / "Recipes E-Book _260909_114355.pdf", "recipe_book"),
]

_llm_semaphore = asyncio.Semaphore(LLM_CONCURRENCY)


def _doc_id(pdf_path):
    return re.sub(r"[^a-z0-9]+", "-", pdf_path.stem.lower()).strip("-")


async def _parse_and_link(doc_id, pdf_path):
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


def _batch(indexed_sections, size):
    return [indexed_sections[i:i + size] for i in range(0, len(indexed_sections), size)]


async def _extract_batch(doc_id, document_type, common, batch_num, total_batches, indexed_batch):
    sections = [section for _, section in indexed_batch]
    async with _llm_semaphore:
        print(f"🤖 [{doc_id}] extracting batch {batch_num}/{total_batches} ({len(sections)} pages)...")
        extraction_by_page = await normalize_and_extract_batch(document_type, sections)  
        # THE RESULT IS 
    print(f"✅ [{doc_id}] batch {batch_num}/{total_batches} done")

    results = []
    for idx, section in indexed_batch:
        extraction = extraction_by_page[section["page"]]
        parent, leaf_chunks = chunks.build_chunks(doc_id, document_type, section, extraction, idx)
        parent.update(common)
        for leaf in leaf_chunks:
            leaf.update(common)
        results.append((parent, leaf_chunks))
    return results, extraction_by_page


async def ingest_document(pdf_path, document_type, coach_id, client_id):
    doc_id = _doc_id(pdf_path)
    doc_out_dir = OUTPUT_DIR / doc_id
    doc_out_dir.mkdir(parents=True, exist_ok=True)
    pages_cache = doc_out_dir / "pages.json"

    if pages_cache.exists():
        print(f"💾 [{doc_id}] reusing cached parse from {pages_cache}")
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
    print(f"📝 [{doc_id}] {len(pages)} pages parsed, manifest written to {doc_out_dir}")

    print(f"🍃 [{doc_id}] saving manifest to Mongo...")
    await store_mongo.mark_previous_versions_inactive(client_id, document_type)
    await store_mongo.save_document_manifest(manifest)

    sections = sections_module.split_sections(pages)  # PAGES IS A LIST OF {"page_index": int, "markdown": str} IN DOCUMENT ORDER
    # sections= OUTPUT IS A LIST OF {"page": int, "text": str} IN DOCUMENT ORDER
    print(f"✂️ [{doc_id}] split into {len(sections)} sections")
    
    
    common = {
        "coach_id": coach_id,
        "client_id": client_id,
        "active": True,
        "effective_date": effective_date,
        "knowledge_tier": "client_plan",
        "language": "ar-en",
        "source_pdf": pdf_path.name,
    }
    
    indexed_batches = _batch(list(enumerate(sections)), PAGES_PER_LLM_CALL) # PROCESSING 4 PAGES AT A TIME
    
    print(f"🤖 [{doc_id}] running LLM extraction on {len(indexed_batches)} batches "
          f"({LLM_CONCURRENCY} concurrent)...")
    
    batches = await asyncio.gather(*(
        _extract_batch(doc_id, document_type, common, i + 1, len(indexed_batches), indexed_batch)
        for i, indexed_batch in enumerate(indexed_batches)
    ))

    all_parents, all_leaves, all_extractions = [], [], {}
    for results, extraction_by_page in batches:
        for parent, leaf_chunks in results:
            all_parents.append(parent)
            all_leaves.extend(leaf_chunks)
        all_extractions.update(extraction_by_page)

    extractions_path = doc_out_dir / "extractions.json"
    extractions_path.write_text(
        json.dumps({str(k): all_extractions[k] for k in sorted(all_extractions)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"📦 [{doc_id}] LLM extraction output written to {extractions_path}")

    print(f"🍃 [{doc_id}] saving {len(all_parents)} parents + {len(all_leaves)} leaf chunks to Mongo...")
    await store_mongo.save_parents_and_entities(all_parents, all_leaves)
    print(f"✅ [{doc_id}] {len(sections)} pages, {len(all_leaves)} leaf chunks, "
          f"{sum(1 for l in all_leaves if l['needs_review'])} flagged for review")
    return all_leaves


async def main():
    for pdf_path, document_type in DOCUMENTS:
        await ingest_document(pdf_path, document_type, COACH_ID, CLIENT_ID)
    print("🎉 Done.")

    # all_leaf_chunks = []
    # for pdf_path, document_type in DOCUMENTS:
    #     all_leaf_chunks.extend(await ingest_document(pdf_path, document_type, COACH_ID, CLIENT_ID))
    #
    # print(f"Reindexing Qdrant with {len(all_leaf_chunks)} total leaf chunks...")
    # await store_qdrant.reindex(all_leaf_chunks)
    # print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
