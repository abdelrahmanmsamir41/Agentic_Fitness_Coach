"""Split per-page markdown into one section per page, and build the
parent/leaf chunk records described in the ingestion plan.
"""
import re

_MD_HEADING_LINE = re.compile(r"^#{1,3}\s.*$", re.MULTILINE)
_MD_LINK = re.compile(r"\[[^\]]*\]\([^)]*\)")
_LINKS_BOILERPLATE = re.compile(r"_?Links:?", re.IGNORECASE)
_MIN_SUBSTANTIVE_LEN = 30


def _strip_links(text):
    text = _MD_LINK.sub("", text)
    text = _LINKS_BOILERPLATE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _substantive_length(text):
    return len(_strip_links(_MD_HEADING_LINE.sub("", text)))


def split_sections(pages):
    """One section per page, never merged -- each page is classified and
    extracted independently downstream, even if near-empty.

    Returns a list of {"text": str, "page": int} in document order.
    """
    sections = []
    for page in pages:
        print(f"Page {page['page_index']+1} processed")
        sections.append({
            "page": page["page_index"],
            "text": page["markdown"].strip()
        })
    
    return sections


def _slug(text):
    # Arabic (or any non-Latin-script) text always produces an empty slug here,
    # since only ASCII a-z0-9 survive -- "section" was a bad fallback for that:
    # it collided with the "00-section" fixed suffix used for whole-section
    # fallback chunks below and once caused a cleanup script to delete real
    # Arabic-named entities by mistake. "unnamed" doesn't look like anything
    # else in this file.
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "unnamed"


def _detail_bits(entity, skip_keys):
    return [f"{k}: {v}" for k, v in entity.items() if k not in skip_keys and v not in (None, "")]


def _make_leaf(parent_chunk_id, doc_id, document_type, section, extraction, idx, suffix_key, entity, embed_text, chunk_type):
    return {
        "chunk_id": f"{parent_chunk_id}:{idx:02d}-{_slug(suffix_key)}",
        "parent_chunk_id": parent_chunk_id,
        "document_id": doc_id,
        "document_type": document_type,
        "chunk_type": chunk_type,
        "page": section["page"],
        "page_topic": extraction["page_topic"],
        "entity": entity,
        "embed_text": embed_text,
        "confidence": extraction.get("confidence", 1.0),
        "needs_review": extraction["needs_review"],
    }


_MIN_ITEMS_FOR_SUMMARY = 2


def _summary_leaf(parent_chunk_id, doc_id, document_type, section, extraction, category, labels):
    embed_text = f"{extraction['page_topic']}: " + ", ".join(labels)
    return _make_leaf(
        parent_chunk_id, doc_id, document_type, section, extraction,
        len(labels), "summary", None, embed_text, f"{category}_summary",
    )



def _weight_exercise_leaves(parent_chunk_id, doc_id, document_type, section, extraction):
    leaves, names = [], []
    for idx, entity in enumerate(extraction["weight_exercises"]):
        name = entity.get("exercise") or f"item-{idx}"
        names.append(name)
        detail_bits = _detail_bits(entity, {"exercise", "video_url"})
        embed_text = f"{extraction['page_topic']} > {name}: " + ", ".join(detail_bits)
        leaves.append(_make_leaf(parent_chunk_id, doc_id, document_type, section, extraction, idx, name, entity, embed_text, "weight_exercise"))
    if len(leaves) >= _MIN_ITEMS_FOR_SUMMARY:
        leaves.append(_summary_leaf(parent_chunk_id, doc_id, document_type, section, extraction, "weight_exercise", names))
    return leaves


def _cardio_leaves(parent_chunk_id, doc_id, document_type, section, extraction):
    leaves, names = [], []
    for idx, entity in enumerate(extraction["cardio_entries"]):
        name = entity.get("activity") or f"cardio-{idx}"
        names.append(name)
        detail_bits = _detail_bits(entity, {"activity", "video_url"})
        embed_text = f"{extraction['page_topic']} > Cardio - {name}: " + ", ".join(detail_bits)
        leaves.append(_make_leaf(parent_chunk_id, doc_id, document_type, section, extraction, idx, name, entity, embed_text, "cardio"))
    if len(leaves) >= _MIN_ITEMS_FOR_SUMMARY:
        leaves.append(_summary_leaf(parent_chunk_id, doc_id, document_type, section, extraction, "cardio", names))
    return leaves


def _glossary_leaves(parent_chunk_id, doc_id, document_type, section, extraction):
    leaves, names = [], []
    for idx, entity in enumerate(extraction["glossary_terms"]):
        term = entity.get("term") or f"term-{idx}"
        names.append(term)
        embed_text = f"{term}: {entity.get('explanation', '')}"
        leaves.append(_make_leaf(parent_chunk_id, doc_id, document_type, section, extraction, idx, term, entity, embed_text, "glossary"))
    if len(leaves) >= _MIN_ITEMS_FOR_SUMMARY:
        leaves.append(_summary_leaf(parent_chunk_id, doc_id, document_type, section, extraction, "glossary", names))
    return leaves


def _advice_leaves(parent_chunk_id, doc_id, document_type, section, extraction):
    leaves, names = [], []
    for idx, entity in enumerate(extraction["advice_items"]):
        topic = entity.get("topic")
        text = entity.get("text", "")
        names.append(topic or f"advice-{idx}")
        embed_text = f"{extraction['page_topic']} - {topic}: {text}" if topic else f"{extraction['page_topic']}: {text}"
        leaves.append(_make_leaf(parent_chunk_id, doc_id, document_type, section, extraction, idx, topic or f"advice-{idx}", entity, embed_text, "advice"))
    if len(leaves) >= _MIN_ITEMS_FOR_SUMMARY:
        leaves.append(_summary_leaf(parent_chunk_id, doc_id, document_type, section, extraction, "advice", names))
    return leaves


def _program_structure_leaves(parent_chunk_id, doc_id, document_type, section, extraction):
    leaves, labels = [], []
    for idx, entity in enumerate(extraction["program_days"]):
        day = entity.get("day_number", idx)
        target = entity.get("target", "")
        labels.append(f"Day {day}: {target}")
        embed_text = f"{extraction['page_topic']} > Day {day}: {target}"
        leaves.append(_make_leaf(parent_chunk_id, doc_id, document_type, section, extraction, idx, f"day-{day}", entity, embed_text, "program_structure"))
    if len(leaves) >= _MIN_ITEMS_FOR_SUMMARY:
        leaves.append(_summary_leaf(parent_chunk_id, doc_id, document_type, section, extraction, "program_structure", labels))
    return leaves


_CATEGORY_LEAF_BUILDERS = {
    "weight_exercise": _weight_exercise_leaves,
    "cardio": _cardio_leaves,
    "glossary": _glossary_leaves,
    "advice": _advice_leaves,
    "program_structure": _program_structure_leaves,
}


def build_chunks(doc_id, document_type, section, extraction, section_index):

    section_slug = _slug(extraction["page_topic"])
    parent_chunk_id = f"{doc_id}:{section_index:04d}-{section_slug}"

    parent = {
        "parent_chunk_id": parent_chunk_id,
        "document_id": doc_id,
        "document_type": document_type,
        "page_topic": extraction["page_topic"],
        "page": section["page"],
        "text": extraction["cleaned_text"],
        "needs_review": extraction["needs_review"],
    }

    if extraction.get("category") == "marketing":
        return parent, []

    if document_type == "training_plan":
        builder = _CATEGORY_LEAF_BUILDERS.get(extraction.get("category"))
        leaves = builder(parent_chunk_id, doc_id, document_type, section, extraction) if builder else []
    else:
        leaves, names = [], []
        for idx, entity in enumerate(extraction["entities"]):
            name = entity.get("meal") or f"item-{idx}"
            names.append(name)
            detail_bits = _detail_bits(entity, {"meal", "video_url"})
            embed_text = f"{extraction['page_topic']} > {name}: " + ", ".join(detail_bits)
            leaves.append(_make_leaf(parent_chunk_id, doc_id, document_type, section, extraction, idx, name, entity, embed_text, "entity"))
        if len(leaves) >= _MIN_ITEMS_FOR_SUMMARY:
            leaves.append(_summary_leaf(parent_chunk_id, doc_id, document_type, section, extraction, "entity", names))

    if not leaves and _substantive_length(extraction["cleaned_text"]) >= _MIN_SUBSTANTIVE_LEN:
        embed_text = f"{extraction['page_topic']}: {_strip_links(extraction['cleaned_text'])}"
        leaves.append(_make_leaf(parent_chunk_id, doc_id, document_type, section, extraction, 0, "section", None, embed_text, "section"))

    return parent, leaves
