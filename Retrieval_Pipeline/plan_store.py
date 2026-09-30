"""Read side of the ingestion stores -- the only module here that touches Mongo, Qdrant or the reference set."""
import asyncio
import sys
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient

sys.path.append(str(Path(__file__).resolve().parent.parent / "Ingestion_Pipeline"))

import store_mongo  # noqa: E402
import store_qdrant  # noqa: E402
from exercise_reference import embeddings, match_exercises  # noqa: E402

from .matching import attach_general_cardio  # noqa: E402

TEXT_CHUNK_TYPES = ["glossary", "advice", "section"]
CHAT_TTL_SECONDS = 30 * 24 * 3600  # Mongo drops checkpoints older than 30 days via a TTL index


async def load_program(client_id):
    program = await store_mongo.get_active_program(client_id)
    if program is None:
        raise LookupError(f"no active training plan ingested for client {client_id!r}")
    return attach_general_cardio(program)


async def search_notes(client_id, query, top_k):
    points = await store_qdrant.search(query, top_k=top_k, filters={
        "client_id": client_id, "active": True, "chunk_type": TEXT_CHUNK_TYPES,
    })
    return [{"score": round(point.score, 3), "topic": point.payload["page_topic"], "text": point.payload["embed_text"]}
            for point in points]


async def lookup_reference_bodypart(exercise):
    match = (await asyncio.to_thread(match_exercises, [exercise]))[0]
    return match and match["ref_bodypart"]


async def embed_exercise(exercise):
    return (await asyncio.to_thread(embeddings.embed_queries, [exercise]))[0]


def get_reference_set():
    return embeddings.get_reference(), embeddings.get_reference_embeddings()


def warm_up():
    """Load the embedding model and reference vectors up front so the first question doesn't pay for it."""
    embeddings.get_embedder().embed_query("warm up")
    embeddings.get_reference_embeddings()


@contextmanager
def chat_checkpointer():
    """Chat history in Mongo. One pooled client for the process; MongoDBSaver takes a sync client
    (its async methods run in a thread pool) and creates its own indexes, incl. the TTL one."""
    with MongoClient(store_mongo.MONGODB_URL) as client:
        yield MongoDBSaver(
            client,
            db_name=store_mongo.DB_NAME,
            checkpoint_collection_name="chat_checkpoints",
            writes_collection_name="chat_checkpoint_writes",
            ttl=CHAT_TTL_SECONDS,
        )
