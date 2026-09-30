"""Qdrant storage for leaf chunks.

sync_document(): routine per-document sync; content hashes skip re-embedding unchanged chunks.
reindex(): rare full rebuild into a fresh collection; COLLECTION_ALIAS flips to it only after an
exact point-count check, so search() never sees a half-built collection. The previous generation
is kept for manual rollback.
"""
import asyncio
import hashlib
import os
import uuid
from itertools import batched

from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    CreateAlias, CreateAliasOperation, DeleteAlias, DeleteAliasOperation,
    Distance, FieldCondition, Filter, MatchAny, MatchValue, PayloadSchemaType,
    PointIdsList, PointStruct, VectorParams,
)

from exercise_reference.embeddings import get_embedder as _get_embeddings  # one shared BGE-m3 instance
from lazy import lazy

load_dotenv()

COLLECTION_ALIAS = "cyt_chunks"  # stable name search() reads through
COLLECTION_PREFIX = "cyt_chunks_gen_"  # physical collections: cyt_chunks_gen_<uuid>
VECTOR_SIZE = 1024
UPSERT_BATCH_SIZE = 32  # one big upsert (~225 points) timed out against the remote cluster

FILTER_FIELDS = {
    "document_type": PayloadSchemaType.KEYWORD,
    "client_id": PayloadSchemaType.KEYWORD,
    "active": PayloadSchemaType.BOOL,
    "day_number": PayloadSchemaType.INTEGER,
    "chunk_type": PayloadSchemaType.KEYWORD,
    "document_id": PayloadSchemaType.KEYWORD,
}

_get_client = lazy(lambda: AsyncQdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"], timeout=60))


async def _get_alias_target(client):
    """The physical collection COLLECTION_ALIAS currently points to, or None
    on the very first-ever reindex (no alias exists yet)."""
    aliases = await client.get_aliases()
    for alias in aliases.aliases:
        if alias.alias_name == COLLECTION_ALIAS:
            return alias.collection_name
    return None


async def _cleanup_old_generations(client, keep_names):
    collections = await client.get_collections()
    for collection in collections.collections:
        if collection.name.startswith(COLLECTION_PREFIX) and collection.name not in keep_names:
            await client.delete_collection(collection.name)


async def _migrate_legacy_collection(client, collections):
    """One-time cleanup: earlier versions of this module created a real
    collection literally named COLLECTION_ALIAS (the old delete-and-recreate
    scheme). An alias can't share a name with an existing real collection,
    so it has to be dropped before the alias can ever be created."""
    if any(collection.name == COLLECTION_ALIAS for collection in collections.collections):
        await client.delete_collection(COLLECTION_ALIAS)


async def _embed_and_upsert(client, collection_name, leaf_chunks):
    if not leaf_chunks:
        return
    texts = [chunk["embed_text"] for chunk in leaf_chunks]
    # embedding is blocking CPU/GPU work
    vectors = await asyncio.to_thread(_get_embeddings().embed_documents, texts)
    points = [PointStruct(id=_to_point_id(chunk["chunk_id"]), vector=vector, payload=chunk)
              for chunk, vector in zip(leaf_chunks, vectors)]
    for batch in batched(points, UPSERT_BATCH_SIZE):
        await client.upsert(collection_name=collection_name, points=list(batch))


async def _create_collection(client):
    """A new, empty physical collection with the payload indexes; returns its name."""
    name = f"{COLLECTION_PREFIX}{uuid.uuid4().hex}"
    await client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    for field, schema in FILTER_FIELDS.items():
        await client.create_payload_index(name, field_name=field, field_schema=schema)
    return name


async def _swap_alias(client, new_name):
    """Points COLLECTION_ALIAS at new_name in one atomic update; returns the previous target."""
    await _migrate_legacy_collection(client, await client.get_collections())
    previous_name = await _get_alias_target(client)
    operations = [CreateAliasOperation(create_alias=CreateAlias(collection_name=new_name, alias_name=COLLECTION_ALIAS))]
    if previous_name:
        operations.insert(0, DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=COLLECTION_ALIAS)))
    await client.update_collection_aliases(change_aliases_operations=operations)
    return previous_name


async def reindex(leaf_chunks):
    client = _get_client()
    new_name = await _create_collection(client)
    await _embed_and_upsert(client, new_name, leaf_chunks)

    count = (await client.count(new_name, exact=True)).count
    if count != len(leaf_chunks):
        await client.delete_collection(new_name)
        raise RuntimeError(
            f"reindex verification failed: expected {len(leaf_chunks)} points, found {count} "
            f"in {new_name} -- new collection discarded, live alias untouched"
        )

    previous_name = await _swap_alias(client, new_name)
    await _cleanup_old_generations(client, keep_names={new_name, previous_name})


async def ensure_collection():
    """First-run bootstrap: create an empty live collection + alias if none
    exists yet. No-op on every later run."""
    client = _get_client()
    await _migrate_legacy_collection(client, await client.get_collections())
    if await _get_alias_target(client) is not None:
        return
    await _swap_alias(client, await _create_collection(client))


def _hash_content(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _diff_chunks(leaf_chunks, existing_points):
    """Stamps content_hash on each chunk. Returns (chunks to (re-)embed, point ids no longer produced)."""
    existing_hash_by_chunk_id = {point.payload["chunk_id"]: point.payload.get("content_hash") for point in existing_points}
    to_embed, keep_point_ids = [], set()
    for chunk in leaf_chunks:
        chunk["content_hash"] = _hash_content(chunk["embed_text"])
        keep_point_ids.add(_to_point_id(chunk["chunk_id"]))
        if existing_hash_by_chunk_id.get(chunk["chunk_id"]) != chunk["content_hash"]:
            to_embed.append(chunk)
    stale_ids = {point.id for point in existing_points} - keep_point_ids
    return to_embed, stale_ids


async def sync_document(doc_id, leaf_chunks):
    """Incrementally sync one document's chunks into the live collection:
    unchanged chunks (same content_hash) are left alone, changed/new chunks
    are (re-)embedded and upserted, and chunks no longer produced for this
    document are deleted. Every other document's points are untouched.
    ponytail: single scroll page (limit 10_000) assumed to cover one
    document's chunks -- paginate via next_page_offset if that's ever not true.
    """
    client = _get_client()
    existing, _ = await client.scroll(
        collection_name=COLLECTION_ALIAS,
        scroll_filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=doc_id))]),
        with_payload=["chunk_id", "content_hash"],
        with_vectors=False,
        limit=10_000,
    )
    to_embed, stale_ids = _diff_chunks(leaf_chunks, existing)
    if stale_ids:
        await client.delete(collection_name=COLLECTION_ALIAS, points_selector=PointIdsList(points=list(stale_ids)))

    await _embed_and_upsert(client, COLLECTION_ALIAS, to_embed)

    print(f"🔮 [{doc_id}] Qdrant sync: {len(to_embed)} embedded, "
          f"{len(leaf_chunks) - len(to_embed)} unchanged skipped, {len(stale_ids)} stale removed")


def _build_match(value):
    if isinstance(value, (list, tuple, set)):
        return MatchAny(any=list(value))
    return MatchValue(value=value)


async def search(query_text, top_k=5, filters=None):
    query_vector = await asyncio.to_thread(_get_embeddings().embed_query, query_text)
    qdrant_filter = None
    if filters:
        qdrant_filter = Filter(must=[FieldCondition(key=key, match=_build_match(value)) for key, value in filters.items()])
    result = await _get_client().query_points(
        collection_name=COLLECTION_ALIAS, query=query_vector, limit=top_k, query_filter=qdrant_filter
    )
    return result.points


def _to_point_id(chunk_id):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _demo():
    from types import SimpleNamespace as Point
    unchanged = {"chunk_id": "d:0", "embed_text": "same"}
    changed = {"chunk_id": "d:1", "embed_text": "new text"}
    existing = [
        Point(id=_to_point_id("d:0"), payload={"chunk_id": "d:0", "content_hash": _hash_content("same")}),
        Point(id=_to_point_id("d:1"), payload={"chunk_id": "d:1", "content_hash": _hash_content("old text")}),
        Point(id=_to_point_id("d:9"), payload={"chunk_id": "d:9", "content_hash": "x"}),
    ]
    to_embed, stale_ids = _diff_chunks([unchanged, changed], existing)
    assert to_embed == [changed], "only the changed chunk is re-embedded"
    assert stale_ids == {_to_point_id("d:9")}, "only the chunk no longer produced is stale"


if __name__ == "__main__":
    _demo()
    print("ok")
