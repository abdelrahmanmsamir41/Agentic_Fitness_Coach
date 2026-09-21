"""Build the parent/leaf chunk records described in the ingestion plan,
for one page (build_chunks) or a whole document's pages (build_all).
"""
from .leaves import _slug, build_leaves


def build_chunks(doc_id, document_type, page, extraction, index):
    page_slug = _slug(extraction["page_topic"])
    parent_chunk_id = f"{doc_id}:{index:04d}-{page_slug}"

    parent = {
        "parent_chunk_id": parent_chunk_id,
        "document_id": doc_id,
        "document_type": document_type,
        "page_topic": extraction["page_topic"],
        "page": page["page_index"],
        "text": extraction["cleaned_text"],
        "needs_review": extraction["needs_review"],
    }

    leaves = build_leaves(parent_chunk_id, doc_id, document_type, page, extraction)
    return parent, leaves


def build_all(doc_id, document_type, common, pages, all_extractions):
    """Builds parent/leaf records for every page, applying the fields
    shared across the whole document (coach_id, effective_date, etc.)."""
    all_parents, all_leaves = [], []
    for idx, page in enumerate(pages):
        extraction = all_extractions[page["page_index"]]
        parent, leaf_chunks = build_chunks(doc_id, document_type, page, extraction, idx)
        parent.update(common)
        for leaf in leaf_chunks:
            leaf.update(common)
        all_parents.append(parent)
        all_leaves.extend(leaf_chunks)
    return all_parents, all_leaves
