"""Parent/leaf chunk records for one page (build_chunks) or a whole document (build_all).
"""
from .leaves import build_leaves, slugify


def build_chunks(doc_id, document_type, page, extraction, position):
    page_slug = slugify(extraction["page_topic"])
    parent_chunk_id = f"{doc_id}:{position:04d}-{page_slug}"

    parent = {
        "parent_chunk_id": parent_chunk_id,
        "document_id": doc_id,
        "document_type": document_type,
        "page_topic": extraction["page_topic"],
        "page": page["page_index"],
        "text": extraction["page_text"],
    }

    leaves = build_leaves(parent_chunk_id, doc_id, document_type, page, extraction)
    return parent, leaves


def build_all(doc_id, document_type, shared_fields, pages, all_extractions):
    """Parent/leaf records for every page, with the document-wide shared_fields applied."""
    all_parents, all_leaves = [], []
    for position, page in enumerate(pages):
        extraction = all_extractions[page["page_index"]]
        parent, leaf_chunks = build_chunks(doc_id, document_type, page, extraction, position)
        parent.update(shared_fields)
        for leaf in leaf_chunks:
            leaf.update(shared_fields)
        all_parents.append(parent)
        all_leaves.extend(leaf_chunks)
    return all_parents, all_leaves
