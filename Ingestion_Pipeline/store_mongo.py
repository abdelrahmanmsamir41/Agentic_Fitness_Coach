"""Local MongoDB storage: document manifests (provenance), parent sections
(full context for retrieval-time expansion), and extracted entities.

sync_document() incrementally replaces one document's rows in the live
collections directly (delete by document_id, then insert) -- no other
document is ever touched, so a failure here is isolated to the one document
being synced, not the whole corpus.
ponytail: delete-then-insert isn't a single atomic step (a crash between the
two briefly leaves this one document missing) -- acceptable since the blast
radius is just that document and rerunning ingest.py for it fixes it; add a
Mongo transaction (needs a replica set) if that gap ever bites.
"""
import os

from dotenv import load_dotenv
from pymongo import AsyncMongoClient

from lazy import lazy

load_dotenv()

MONGODB_URL = os.environ.get("MONGODB_URL", "mongodb://localhost:27017/")
DB_NAME = "choose_your_trainer"

# pymongo's native async client; motor is deprecated
_get_client = lazy(lambda: AsyncMongoClient(MONGODB_URL))


def _get_db():
    return _get_client()[DB_NAME]


async def mark_previous_versions_inactive(client_id, document_type):
    result = await _get_db()["documents"].update_many(
        {"client_id": client_id, "document_type": document_type, "active": True},
        {"$set": {"active": False}},
    )
    if result.modified_count:
        print(f"🍃 marked {result.modified_count} previous '{document_type}' version(s) inactive")


async def sync_document(manifest, parents, leaf_chunks):
    doc_id = manifest["document_id"]
    db = _get_db()
    await db["documents"].delete_many({"document_id": doc_id})
    await db["parents"].delete_many({"document_id": doc_id})
    await db["entities"].delete_many({"document_id": doc_id})
    await db["documents"].insert_one(dict(manifest))
    if parents:
        await db["parents"].insert_many([dict(parent) for parent in parents])
    if leaf_chunks:
        await db["entities"].insert_many([dict(chunk) for chunk in leaf_chunks])
    print(f"🍃 [{doc_id}] synced: manifest + {len(parents)} parent(s) + {len(leaf_chunks)} entity chunk(s)")


async def get_parent(parent_chunk_id):
    return await _get_db()["parents"].find_one({"parent_chunk_id": parent_chunk_id}, {"_id": 0})


async def get_active_program(client_id, document_type="training_plan"):
    document = await _get_db()["documents"].find_one(
        {"client_id": client_id, "document_type": document_type, "active": True}, {"_id": 0, "program": 1}
    )
    return (document or {}).get("program")
