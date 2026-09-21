"""Qdrant storage for leaf chunks. Ported from counselling-bot's Qdrant setup
(same delete-then-recreate pattern for a full reindex), using the async
client throughout.
"""
import asyncio
import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams

load_dotenv()

COLLECTION_NAME = "cyt_chunks"
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
    "day": PayloadSchemaType.KEYWORD,
    "chunk_type": PayloadSchemaType.KEYWORD,
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


async def reindex(leaf_chunks):
    """Delete and recreate the collection, then embed+upsert every leaf chunk."""
    client = _get_client()

    if await client.collection_exists(COLLECTION_NAME):
        await client.delete_collection(COLLECTION_NAME)
    await client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    for field, schema in FILTER_FIELDS.items():
        await client.create_payload_index(COLLECTION_NAME, field_name=field, field_schema=schema)

    if not leaf_chunks:
        return

    texts = [c["embed_text"] for c in leaf_chunks]
    # sentence-transformers embedding is a blocking CPU/GPU call, no native
    # async form -- run it off the event loop instead of blocking it.
    vectors = await asyncio.to_thread(_get_embeddings().embed_documents, texts)
    points = [
        PointStruct(id=_stable_id(c["chunk_id"]), vector=vector, payload=c)
        for c, vector in zip(leaf_chunks, vectors)
    ]
    for i in range(0, len(points), UPSERT_BATCH_SIZE):
        await client.upsert(collection_name=COLLECTION_NAME, points=points[i:i + UPSERT_BATCH_SIZE])


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
        collection_name=COLLECTION_NAME, query=query_vector, limit=top_k, query_filter=qdrant_filter
    )
    return result.points


def _stable_id(chunk_id):
    import uuid
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
