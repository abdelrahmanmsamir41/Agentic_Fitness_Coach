"""Qdrant storage for leaf chunks.

sync_document() is the routine path: per document, chunks are content-hashed
so unchanged ones are never re-embedded or re-upserted, changed/new ones are,
and chunks no longer produced for that document are deleted by point id --
every other document's points are untouched.

reindex() is a separate, rare-use tool for a full-corpus rebuild (e.g.
switching embedding models): it builds into a fresh, uniquely named
collection and only atomically flips COLLECTION_ALIAS onto it once the build
is verified by an exact point count -- search() always reads through the
alias, so it never sees a missing or half-built collection during a rebuild,
and a bad build is discarded without ever touching what's live. The previous
generation is kept as an instant manual rollback; only generations older
than that get cleaned up.
"""
import asyncio
import hashlib
import os
import uuid

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    CreateAlias, CreateAliasOperation, DeleteAlias, DeleteAliasOperation,
    Distance, PayloadSchemaType, PointStruct, VectorParams,
)

load_dotenv()

COLLECTION_ALIAS = "cyt_chunks"  # stable name search() reads through
COLLECTION_PREFIX = "cyt_chunks_gen_"  # physical collections: cyt_chunks_gen_<uuid>
# BGE-M3 over multilingual-e5-large (counselling-bot's choice): stronger on
# interleaved Arabic/English content specifically (75.5% cross-lingual
# Recall@100 vs e5-large's lighter-weight tradeoff), same 1024-dim output so
# no Qdrant schema change, and an 8192-token window vs e5-large's 512.
EMBEDDING_MODEL = "BAAI/bge-m3"
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

_client = None
_embeddings = None


def _get_client():
    global _client
    if _client is None:
        _client = AsyncQdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"], timeout=60)
    return _client


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


async def _alias_target(client):
    """The physical collection COLLECTION_ALIAS currently points to, or None
    on the very first-ever reindex (no alias exists yet)."""
    aliases = await client.get_aliases()
    for a in aliases.aliases:
        if a.alias_name == COLLECTION_ALIAS:
            return a.collection_name
    return None


async def _cleanup_old_generations(client, keep):
    collections = await client.get_collections()
    for c in collections.collections:
        if c.name.startswith(COLLECTION_PREFIX) and c.name not in keep:
            await client.delete_collection(c.name)


async def _migrate_legacy_collection(client, collections):
    """One-time cleanup: earlier versions of this module created a real
    collection literally named COLLECTION_ALIAS (the old delete-and-recreate
    scheme). An alias can't share a name with an existing real collection,
    so it has to be dropped before the alias can ever be created."""
    if any(c.name == COLLECTION_ALIAS for c in collections.collections):
        await client.delete_collection(COLLECTION_ALIAS)


async def reindex(leaf_chunks):
    client = _get_client()
    new_name = f"{COLLECTION_PREFIX}{uuid.uuid4().hex}"

    await client.create_collection(
        collection_name=new_name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    for field, schema in FILTER_FIELDS.items():
        await client.create_payload_index(new_name, field_name=field, field_schema=schema)

    if leaf_chunks:
        texts = [c["embed_text"] for c in leaf_chunks]
        # sentence-transformers embedding is a blocking CPU/GPU call, no native
        # async form -- run it off the event loop instead of blocking it.
        vectors = await asyncio.to_thread(_get_embeddings().embed_documents, texts)
        points = [
            PointStruct(id=_stable_id(c["chunk_id"]), vector=vector, payload=c)
            for c, vector in zip(leaf_chunks, vectors)
        ]
        for i in range(0, len(points), UPSERT_BATCH_SIZE):
            await client.upsert(collection_name=new_name, points=points[i:i + UPSERT_BATCH_SIZE])

    count = (await client.count(new_name, exact=True)).count
    if count != len(leaf_chunks):
        await client.delete_collection(new_name)
        raise RuntimeError(
            f"reindex verification failed: expected {len(leaf_chunks)} points, found {count} "
            f"in {new_name} -- new collection discarded, live alias untouched"
        )

    await _migrate_legacy_collection(client, await client.get_collections())
    previous_name = await _alias_target(client)
    operations = [CreateAliasOperation(create_alias=CreateAlias(collection_name=new_name, alias_name=COLLECTION_ALIAS))]
    if previous_name:
        operations.insert(0, DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=COLLECTION_ALIAS)))
    await client.update_collection_aliases(change_aliases_operations=operations)

    await _cleanup_old_generations(client, keep={new_name, previous_name})


async def ensure_collection():
    """First-run bootstrap: create an empty live collection + alias if none
    exists yet. No-op on every later run."""
    client = _get_client()
    await _migrate_legacy_collection(client, await client.get_collections())
    if await _alias_target(client) is not None:
        return
    name = f"{COLLECTION_PREFIX}{uuid.uuid4().hex}"
    await client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    for field, schema in FILTER_FIELDS.items():
        await client.create_payload_index(name, field_name=field, field_schema=schema)
    await client.update_collection_aliases(change_aliases_operations=[
        CreateAliasOperation(create_alias=CreateAlias(collection_name=name, alias_name=COLLECTION_ALIAS))
    ])


def _content_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def sync_document(doc_id, leaf_chunks):
    """Incrementally sync one document's chunks into the live collection:
    unchanged chunks (same content_hash) are left alone, changed/new chunks
    are (re-)embedded and upserted, and chunks no longer produced for this
    document are deleted. Every other document's points are untouched.
    ponytail: single scroll page (limit 10_000) assumed to cover one
    document's chunks -- paginate via next_page_offset if that's ever not true.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue, PointIdsList

    client = _get_client()
    existing, _ = await client.scroll(
        collection_name=COLLECTION_ALIAS,
        scroll_filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=doc_id))]),
        with_payload=["chunk_id", "content_hash"],
        with_vectors=False,
        limit=10_000,
    )
    existing_hash_by_chunk_id = {p.payload["chunk_id"]: p.payload.get("content_hash") for p in existing}
    existing_point_ids = {p.id for p in existing}

    to_embed, keep_point_ids = [], set()
    for c in leaf_chunks:
        c["content_hash"] = _content_hash(c["embed_text"])
        keep_point_ids.add(_stable_id(c["chunk_id"]))
        if existing_hash_by_chunk_id.get(c["chunk_id"]) != c["content_hash"]:
            to_embed.append(c)

    stale_ids = existing_point_ids - keep_point_ids
    if stale_ids:
        await client.delete(collection_name=COLLECTION_ALIAS, points_selector=PointIdsList(points=list(stale_ids)))

    if to_embed:
        texts = [c["embed_text"] for c in to_embed]
        vectors = await asyncio.to_thread(_get_embeddings().embed_documents, texts)
        points = [PointStruct(id=_stable_id(c["chunk_id"]), vector=v, payload=c) for c, v in zip(to_embed, vectors)]
        for i in range(0, len(points), UPSERT_BATCH_SIZE):
            await client.upsert(collection_name=COLLECTION_ALIAS, points=points[i:i + UPSERT_BATCH_SIZE])

    print(f"🔮 [{doc_id}] Qdrant sync: {len(to_embed)} embedded, "
          f"{len(leaf_chunks) - len(to_embed)} unchanged skipped, {len(stale_ids)} stale removed")


def _match(value):
    from qdrant_client.models import MatchAny, MatchValue
    if isinstance(value, (list, tuple, set)):
        return MatchAny(any=list(value))
    return MatchValue(value=value)


async def search(query_text, top_k=5, filters=None):
    from qdrant_client.models import FieldCondition, Filter

    query_vector = await asyncio.to_thread(_get_embeddings().embed_query, query_text)
    qdrant_filter = None
    if filters:
        qdrant_filter = Filter(must=[FieldCondition(key=k, match=_match(v)) for k, v in filters.items()])
    result = await _get_client().query_points(
        collection_name=COLLECTION_ALIAS, query=query_vector, limit=top_k, query_filter=qdrant_filter
    )
    return result.points


def _stable_id(chunk_id):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
