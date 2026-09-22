"""Orchestrates one document's ingestion: parse -> normalize+extract ->
enrich (day numbers, exercise-reference matching) -> chunk -> Mongo
(manifests/parents/entities) -> Qdrant (leaf embeddings), run across every
configured document in main(). Both stores are synced incrementally per
document (store_mongo.sync_document / store_qdrant.sync_document) --
unchanged chunks are never re-embedded, and a document left out of DOCUMENTS
is simply not touched rather than dropped.
"""
import asyncio
import datetime
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import chunks
import store_mongo
import store_qdrant
from enrich import enrich_document
from extraction.extract import extract_document
from parse import parse_and_link

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "ingestion_output"

COACH_ID = "coach_self"
CLIENT_ID = "client_self"

DOCUMENTS = [
    # (REPO_ROOT / "splits" / "W-1401-260906-0202 - Abdelrhman Mohamed Samer _260909_114328.pdf", "training_plan"),
    # (REPO_ROOT / "Splits" / "belghamdi_splits" / "BelghamdiSplit-Anterior-x-Posterior.pdf", "training_plan"),
    (REPO_ROOT / "splits" / "shady_splits" / "W-1401-260906-0202 - Abdelrhman Mohamed Samer.pdf", "training_plan"),
    # (REPO_ROOT / "splits" / "D-1401-260906-0202 - Abdelrhman Mohamed Samer _260909_114305.pdf", "diet_plan"),
    # (REPO_ROOT / "splits" / "Recipes E-Book _260909_114355.pdf", "recipe_book"),
]


def _doc_id(pdf_path):
    return re.sub(r"[^a-z0-9]+", "-", pdf_path.stem.lower()).strip("-")


async def ingest_document(pdf_path, document_type, coach_id, client_id):
    doc_id = _doc_id(pdf_path)
    doc_out_dir = OUTPUT_DIR / doc_id
    doc_out_dir.mkdir(parents=True, exist_ok=True)
    pages_cache = doc_out_dir / "pages.json"

    if pages_cache.exists():
        print(f"💾 [{doc_id}] reusing cached parse from {pages_cache}")
        pages = json.loads(pages_cache.read_text(encoding="utf-8"))
    else:
        pages = await parse_and_link(doc_id, pdf_path)
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

    await store_mongo.mark_previous_versions_inactive(client_id, document_type)

    common = {
        "coach_id": coach_id,
        "client_id": client_id,
        "active": True,
        "effective_date": effective_date,
        "knowledge_tier": "client_plan",
        "language": "ar-en",
        "source_pdf": pdf_path.name,
    }

    extractions_path = doc_out_dir / "extractions.json"
    if extractions_path.exists():
        print(f"💾 [{doc_id}] reusing cached extraction from {extractions_path}")
        cached = json.loads(extractions_path.read_text(encoding="utf-8"))
        all_extractions = {int(k): v for k, v in cached.items()}
    else:
        all_extractions = await extract_document(doc_id, document_type, pages)
        await enrich_document(doc_id, all_extractions)
        extractions_path.write_text(
            json.dumps({str(k): all_extractions[k] for k in sorted(all_extractions)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"📦 [{doc_id}] LLM extraction output written to {extractions_path}")

    all_parents, all_leaves = chunks.build_all(doc_id, document_type, common, pages, all_extractions)

    await asyncio.gather(
        store_mongo.sync_document(manifest, all_parents, all_leaves),
        store_qdrant.sync_document(doc_id, all_leaves),
    )
    print(f"✅ [{doc_id}] {len(pages)} pages, {len(all_leaves)} leaf chunks, "
          f"{sum(1 for l in all_leaves if l['needs_review'])} flagged for review")
    return all_leaves


async def main():
    await store_qdrant.ensure_collection()

    all_leaf_chunks = []
    for pdf_path, document_type in DOCUMENTS:
        try:
            all_leaf_chunks.extend(await ingest_document(pdf_path, document_type, COACH_ID, CLIENT_ID))
        except Exception as e:
            print(f"❌ [{_doc_id(pdf_path)}] ingestion failed, skipping: {e}")

    print(f"🎉 Done: {len(all_leaf_chunks)} leaf chunk(s) synced across {len(DOCUMENTS)} document(s).")


if __name__ == "__main__":
    asyncio.run(main())
