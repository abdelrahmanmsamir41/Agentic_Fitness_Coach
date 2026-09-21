"""Local MongoDB storage: document manifests (provenance), parent sections
(full context for retrieval-time expansion), and extracted entities.
"""
import asyncio
import os

from dotenv import load_dotenv
from pymongo import AsyncMongoClient

load_dotenv()

_client = None


def _db():
    # pymongo's own native async driver, not motor -- motor is now in its
    # deprecation window (deprecated May 2026) and MongoDB itself recommends
    # this as the direct replacement: same API, better latency/throughput
    # since it runs on asyncio directly instead of motor's thread-pool shim.
    global _client
    if _client is None:
        _client = AsyncMongoClient(os.environ.get("MONGODB_URL", "mongodb://localhost:27017/"))
    return _client["choose_your_trainer"]


async def delete_all():
    """Wipe every manifest/parent/entity -- mirrors store_qdrant.reindex()'s
    delete-then-rebuild pattern, so a full reingestion starts from a clean
    slate instead of piling up duplicates alongside old runs."""
    results = await asyncio.gather(*(
        _db()[name].delete_many({}) for name in ("documents", "parents", "entities")
    ))
    print(f"🍃 wiped Mongo: {sum(r.deleted_count for r in results)} document(s) deleted "
          f"across documents/parents/entities")


async def mark_previous_versions_inactive(client_id, document_type):
    result = await _db()["documents"].update_many(
        {"client_id": client_id, "document_type": document_type, "active": True},
        {"$set": {"active": False}},
    )
    if result.modified_count:
        print(f"🍃 marked {result.modified_count} previous '{document_type}' version(s) inactive")


async def save_document_manifest(manifest):
    await _db()["documents"].insert_one(dict(manifest))
    print(f"🍃 manifest for '{manifest['document_id']}' saved to Mongo")


async def save_parents_and_entities(parents, leaf_chunks):
    if parents:
        await _db()["parents"].insert_many([dict(p) for p in parents])
    if leaf_chunks:
        await _db()["entities"].insert_many([dict(c) for c in leaf_chunks])
    print(f"🍃 saved {len(parents)} parent(s) + {len(leaf_chunks)} entity chunk(s) to Mongo")


async def get_parent(parent_chunk_id):
    return await _db()["parents"].find_one({"parent_chunk_id": parent_chunk_id}, {"_id": 0})
