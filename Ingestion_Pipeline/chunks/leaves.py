"""Turn one page's LLM extraction into its leaf chunk records -- one leaf per
entity (exercise, cardio entry, glossary term, ...), plus a generic "section"
fallback for prose-only pages that have no structured entities.
"""
import re

from text_utils import format_range

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


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "unnamed"


def _format_value(value):
    if isinstance(value, dict) and "low" in value:
        return None if value.get("low") is None else format_range(value["low"], value.get("high"))
    return value


def _format_details(entity, skipped_keys):
    details = []
    for key, value in entity.items():
        if key in skipped_keys:
            continue
        formatted = _format_value(value)
        if formatted not in (None, ""):
            details.append(f"{key}: {formatted}")
    return details


def _make_leaf(parent_chunk_id, doc_id, document_type, page, extraction, index, suffix_key, entity, embed_text, chunk_type):
    day_number = extraction.get("day_number")
    if day_number is not None:
        # a readable phrase, not the bare digit -- gives semantic search something
        # to match on ("what's on day 1") while day_number itself stays a payload
        # field for exact filtering, since a number alone embeds with little signal.
        embed_text = f"Day {day_number} - {embed_text}"
    return {
        "chunk_id": f"{parent_chunk_id}:{index:02d}-{slugify(suffix_key)}",
        "parent_chunk_id": parent_chunk_id,
        "document_id": doc_id,
        "document_type": document_type,
        "chunk_type": chunk_type,
        "page": page["page_index"],
        "page_topic": extraction["page_topic"],
        "day_number": day_number,
        "entity": entity,
        "embed_text": embed_text,
    }


def _build_weight_exercise_leaves(parent_chunk_id, doc_id, document_type, page, extraction):
    leaves = []
    for index, entity in enumerate(extraction["weight_exercises"]):
        name = entity.get("exercise") or f"item-{index}"
        details = _format_details(entity, {"exercise", "video_url", "ref_match_score"})
        embed_text = f"{extraction['page_topic']} > {name}: " + ", ".join(details)
        leaves.append(_make_leaf(parent_chunk_id, doc_id, document_type, page, extraction, index, name, entity, embed_text, "weight_exercise"))
    return leaves


def _build_cardio_leaves(parent_chunk_id, doc_id, document_type, page, extraction):
    leaves = []
    for index, entity in enumerate(extraction["cardio_entries"]):
        name = entity.get("activity") or f"cardio-{index}"
        details = _format_details(entity, {"activity", "video_url"})
        embed_text = f"{extraction['page_topic']} > Cardio - {name}: " + ", ".join(details)
        leaves.append(_make_leaf(parent_chunk_id, doc_id, document_type, page, extraction, index, name, entity, embed_text, "cardio"))
    return leaves


def _build_glossary_leaves(parent_chunk_id, doc_id, document_type, page, extraction):
    leaves = []
    for index, entity in enumerate(extraction["glossary_terms"]):
        term = entity.get("term") or f"term-{index}"
        embed_text = f"{term}: {entity.get('explanation', '')}"
        leaves.append(_make_leaf(parent_chunk_id, doc_id, document_type, page, extraction, index, term, entity, embed_text, "glossary"))
    return leaves


def _build_advice_leaves(parent_chunk_id, doc_id, document_type, page, extraction):
    leaves = []
    for index, entity in enumerate(extraction["advice_items"]):
        text = entity.get("text", "")
        embed_text = f"{extraction['page_topic']}: {text}"
        leaves.append(_make_leaf(parent_chunk_id, doc_id, document_type, page, extraction, index, f"advice-{index}", entity, embed_text, "advice"))
    return leaves


_CATEGORY_LEAF_BUILDERS = {
    "weight_exercise": _build_weight_exercise_leaves,
    "cardio": _build_cardio_leaves,
    "glossary": _build_glossary_leaves,
    "advice": _build_advice_leaves,
}


def build_leaves(parent_chunk_id, doc_id, document_type, page, extraction):
    """Returns the list of leaf records for one page: runs every category
    builder (a page's lists aren't gated by its single dominant category
    anymore, see extraction/prompts.py, so more than one can be populated),
    or falls back to a generic "section" leaf if none of them produced
    anything. A "marketing" page gets no leaves at all -- nothing there is
    worth retrieving. Neither does "program_structure" (its day/target
    mapping only exists to stamp day_number onto other pages, see
    enrich.resolve_day_numbers -- as its own leaf it would just repeat what's
    already in every one of that day's exercise leaves) or "rest_day" (a
    standalone rest-day page carries nothing of its own worth retrieving
    either, same reasoning as program_structure)."""
    if extraction.get("category") in ("marketing", "program_structure", "rest_day"):
        return []

    leaves = [
        leaf
        for builder in _CATEGORY_LEAF_BUILDERS.values()
        for leaf in builder(parent_chunk_id, doc_id, document_type, page, extraction)
    ]

    if not leaves and _substantive_length(extraction["page_text"]) >= _MIN_SUBSTANTIVE_LEN:
        embed_text = f"{extraction['page_topic']}: {_strip_links(extraction['page_text'])}"
        leaves.append(_make_leaf(parent_chunk_id, doc_id, document_type, page, extraction, 0, "section", None, embed_text, "section"))

    return leaves


def _demo():
    assert _format_value({"low": 8.0, "high": 12.0}) == "8-12"
    assert _format_value({"low": 2.0, "high": None}) == "2"
    assert _format_value({"low": 5.5, "high": 5.5}) == "5.5"
    assert _format_value({"low": None, "high": None}) is None
    assert _format_value("60-90s") == "60-90s"
    assert _format_details({"exercise": "Squat", "reps": {"low": 8.0, "high": 12.0}, "video_url": "x"}, {"exercise", "video_url"}) == ["reps: 8-12"]


if __name__ == "__main__":
    _demo()
    print("ok")
