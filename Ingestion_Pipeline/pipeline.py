"""Shared ingestion orchestration: parse -> extract -> enrich -> chunk -> Mongo
(manifests/parents/entities) -> Qdrant (leaf embeddings). The OCR and LLM
pipelines differ only in their parse and extract steps, passed in as callables:
  parse(doc_id, pdf_path) -> awaitable list of {"page_index": int, "markdown": str}
  extract(doc_id, document_type, pages, pdf_path) -> awaitable {page: extraction}
Both stores are synced incrementally per document (store_mongo.sync_document /
store_qdrant.sync_document) -- unchanged chunks are never re-embedded, and a
document left out of the run is simply not touched rather than dropped.
"""
import asyncio
import datetime
import json
import re

import chunks
import store_mongo
import store_qdrant
from enrich import build_program_view, enrich_document

COACH_ID = "coach_self"
CLIENT_ID = "client_self"


def _make_doc_id(pdf_path):
    return re.sub(r"[^a-z0-9]+", "-", pdf_path.stem.lower()).strip("-")


def _write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


async def _load_or_parse(doc_id, pdf_path, parse, doc_out_dir):
    pages_cache = doc_out_dir / "pages.json"
    if pages_cache.exists():
        print(f"💾 [{doc_id}] reusing cached parse from {pages_cache}")
        return json.loads(pages_cache.read_text(encoding="utf-8"))
    pages = await parse(doc_id, pdf_path)
    _write_json(pages_cache, pages)
    return pages


async def _load_or_extract(doc_id, document_type, pages, pdf_path, extract, doc_out_dir):
    """Extracts and enriches every page, or reuses the cached result."""
    extractions_path = doc_out_dir / "extractions.json"
    if extractions_path.exists():
        print(f"💾 [{doc_id}] reusing cached extraction from {extractions_path}")
        cached = json.loads(extractions_path.read_text(encoding="utf-8"))
        return {int(page_index): extraction for page_index, extraction in cached.items()}
    all_extractions = await extract(doc_id, document_type, pages, str(pdf_path))
    await enrich_document(doc_id, all_extractions)
    _write_json(extractions_path, {str(page_index): all_extractions[page_index] for page_index in sorted(all_extractions)})
    print(f"📦 [{doc_id}] LLM extraction output written to {extractions_path}")
    return all_extractions


def _build_manifest(doc_id, pdf_path, document_type, coach_id, client_id, page_count, effective_date):
    return {
        "document_id": doc_id,
        "document_type": document_type,
        "source_pdf": str(pdf_path),
        "coach_id": coach_id,
        "client_id": client_id,
        "page_count": page_count,
        "ingested_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "effective_date": effective_date,
        "active": True,
    }


def _build_shared_fields(pdf_path, coach_id, client_id, effective_date):
    """Fields stamped onto every parent/leaf chunk of the document."""
    return {
        "coach_id": coach_id,
        "client_id": client_id,
        "active": True,
        "effective_date": effective_date,
        "knowledge_tier": "client_plan",
        "language": "ar-en",
        "source_pdf": pdf_path.name,
    }


async def ingest_document(pdf_path, document_type, coach_id, client_id, *, parse, extract, output_dir):
    doc_id = _make_doc_id(pdf_path)
    doc_out_dir = output_dir / doc_id
    doc_out_dir.mkdir(parents=True, exist_ok=True)
    effective_date = datetime.date.today().isoformat()

    pages = await _load_or_parse(doc_id, pdf_path, parse, doc_out_dir)
    (doc_out_dir / "merged.md").write_text("\n\n".join(page["markdown"] for page in pages), encoding="utf-8")
    manifest = _build_manifest(doc_id, pdf_path, document_type, coach_id, client_id, len(pages), effective_date)
    _write_json(doc_out_dir / "manifest.json", manifest)
    print(f"📝 [{doc_id}] {len(pages)} pages parsed, manifest written to {doc_out_dir}")

    await store_mongo.mark_previous_versions_inactive(client_id, document_type)

    all_extractions = await _load_or_extract(doc_id, document_type, pages, pdf_path, extract, doc_out_dir)
    program_view = build_program_view(all_extractions)
    _write_json(doc_out_dir / "program.json", program_view)

    shared_fields = _build_shared_fields(pdf_path, coach_id, client_id, effective_date)
    all_parents, all_leaves = chunks.build_all(doc_id, document_type, shared_fields, pages, all_extractions)
    await asyncio.gather(
        store_mongo.sync_document({**manifest, "program": program_view}, all_parents, all_leaves),
        store_qdrant.sync_document(doc_id, all_leaves),
    )
    print(f"✅ [{doc_id}] {len(pages)} pages, {len(all_leaves)} leaf chunks")
    return all_leaves


async def run(documents, *, parse, extract, output_dir):
    """documents: list of (pdf_path, document_type). One failing document is skipped, not fatal."""
    await store_qdrant.ensure_collection()

    all_leaf_chunks = []
    for pdf_path, document_type in documents:
        try:
            all_leaf_chunks.extend(await ingest_document(
                pdf_path, document_type, COACH_ID, CLIENT_ID, parse=parse, extract=extract, output_dir=output_dir))
        except Exception as error:
            print(f"❌ [{_make_doc_id(pdf_path)}] ingestion failed, skipping: {error}")

    print(f"🎉 Done: {len(all_leaf_chunks)} leaf chunk(s) synced across {len(documents)} document(s).")
