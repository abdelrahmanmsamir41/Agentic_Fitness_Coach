"""Loads the static exercise reference set and its BGE-m3 embeddings -- the
reference titles are embedded once ever (cached to data/exercises_embeddings.npy)
since the reference set is static; re-running the model over it on every
ingestion run would be pure waste.
"""
import json
import os

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

from lazy import lazy

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
REFERENCE_JSON = os.path.join(DATA_DIR, "exercises.json")
EMBEDDINGS_CACHE = os.path.join(DATA_DIR, "exercises_embeddings.npy")

EMBEDDING_MODEL = "BAAI/bge-m3"  # same as store_qdrant.py -- keep both in sync if this ever changes

_reference_embeddings = None


def _load_embedder():
    print(f"🧠 loading {EMBEDDING_MODEL}...")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


get_embedder = lazy(_load_embedder)


def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def _load_reference():
    with open(REFERENCE_JSON, encoding="utf-8") as f:
        return json.load(f)


get_reference = lazy(_load_reference)


def get_reference_embeddings():
    global _reference_embeddings
    if _reference_embeddings is not None:
        return _reference_embeddings

    if os.path.exists(EMBEDDINGS_CACHE):
        _reference_embeddings = np.load(EMBEDDINGS_CACHE)
        print(f"🧠 loaded {len(_reference_embeddings)} cached reference embeddings from {EMBEDDINGS_CACHE}")
        return _reference_embeddings

    titles = [r["title"] for r in get_reference()]
    print(f"🧠 embedding {len(titles)} reference exercises (one-time, cached afterward)...")
    vectors = normalize(np.array(get_embedder().embed_documents(titles), dtype=np.float32))
    np.save(EMBEDDINGS_CACHE, vectors)
    _reference_embeddings = vectors
    print(f"🧠 reference embeddings cached to {EMBEDDINGS_CACHE}")
    return _reference_embeddings


def embed_queries(names):
    return normalize(np.array(get_embedder().embed_documents(names), dtype=np.float32))


def _demo():
    reference = get_reference()
    assert len(reference) > 2000, "reference set looks truncated"
    assert set(reference[0]) == {"title", "type", "bodypart", "equipment", "level"}

    vectors = normalize(np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32))
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0), "normalize() must produce unit vectors"


if __name__ == "__main__":
    _demo()
    print("ok")
