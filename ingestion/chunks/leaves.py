"""Turn one page's LLM extraction into its leaf chunk records -- one leaf per
entity (exercise, cardio entry, glossary term, ...), plus a generic "section"
fallback for prose-only pages that have no structured entities.
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


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "unnamed"


def _fmt_number(n):
    return str(int(n)) if n == int(n) else str(n)


def _fmt_value(v):
    if isinstance(v, dict) and "low" in v:
        low, high = v.get("low"), v.get("high")
        if low is None:
            return None
        return _fmt_number(low) if high is None or high == low else f"{_fmt_number(low)}-{_fmt_number(high)}"
    return v


def _detail_bits(entity, skip_keys):
    bits = []
    for k, v in entity.items():
        if k in skip_keys:
            continue
        v = _fmt_value(v)
        if v not in (None, ""):
            bits.append(f"{k}: {v}")
    return bits


def _make_leaf(parent_chunk_id, doc_id, document_type, page, extraction, idx, suffix_key, entity, embed_text, chunk_type):
    day_number = extraction.get("day_number")
    if day_number is not None:
        # a readable phrase, not the bare digit -- gives semantic search something
        # to match on ("what's on day 1") while day_number itself stays a payload
        # field for exact filtering, since a number alone embeds with little signal.
        embed_text = f"Day {day_number} - {embed_text}"
    return {
        "chunk_id": f"{parent_chunk_id}:{idx:02d}-{_slug(suffix_key)}",
        "parent_chunk_id": parent_chunk_id,
        "document_id": doc_id,
        "document_type": document_type,
        "chunk_type": chunk_type,
        "page": page["page_index"],
        "page_topic": extraction["page_topic"],
        "day_number": day_number,
        "entity": entity,
        "embed_text": embed_text,
        "needs_review": extraction["needs_review"],
    }


def _weight_exercise_leaves(parent_chunk_id, doc_id, document_type, page, extraction):
    leaves = []
    for idx, entity in enumerate(extraction["weight_exercises"]):
        name = entity.get("exercise") or f"item-{idx}"
        detail_bits = _detail_bits(entity, {"exercise", "video_url", "ref_match_score"})
        embed_text = f"{extraction['page_topic']} > {name}: " + ", ".join(detail_bits)
        leaves.append(_make_leaf(parent_chunk_id, doc_id, document_type, page, extraction, idx, name, entity, embed_text, "weight_exercise"))
    return leaves


def _cardio_leaves(parent_chunk_id, doc_id, document_type, page, extraction):
    leaves = []
    for idx, entity in enumerate(extraction["cardio_entries"]):
        name = entity.get("activity") or f"cardio-{idx}"
        detail_bits = _detail_bits(entity, {"activity", "video_url"})
        embed_text = f"{extraction['page_topic']} > Cardio - {name}: " + ", ".join(detail_bits)
        leaves.append(_make_leaf(parent_chunk_id, doc_id, document_type, page, extraction, idx, name, entity, embed_text, "cardio"))
    return leaves


def _glossary_leaves(parent_chunk_id, doc_id, document_type, page, extraction):
    leaves = []
    for idx, entity in enumerate(extraction["glossary_terms"]):
        term = entity.get("term") or f"term-{idx}"
        embed_text = f"{term}: {entity.get('explanation', '')}"
        leaves.append(_make_leaf(parent_chunk_id, doc_id, document_type, page, extraction, idx, term, entity, embed_text, "glossary"))
    return leaves


def _advice_leaves(parent_chunk_id, doc_id, document_type, page, extraction):
    leaves = []
    for idx, entity in enumerate(extraction["advice_items"]):
        text = entity.get("text", "")
        embed_text = f"{extraction['page_topic']}: {text}"
        leaves.append(_make_leaf(parent_chunk_id, doc_id, document_type, page, extraction, idx, f"advice-{idx}", entity, embed_text, "advice"))
    return leaves


_CATEGORY_LEAF_BUILDERS = {
    "weight_exercise": _weight_exercise_leaves,
    "cardio": _cardio_leaves,
    "glossary": _glossary_leaves,
    "advice": _advice_leaves,
}


def build_leaves(parent_chunk_id, doc_id, document_type, page, extraction):
    """Returns the list of leaf records for one page: dispatches to the
    category-specific builder, or falls back to a generic "section" leaf for
    a prose-only page with no structured entities. A "marketing" page gets no
    leaves at all -- nothing there is worth retrieving. Neither does
    "program_structure": its day/target mapping is only extracted to stamp
    day_number onto every weight_exercise/cardio leaf (see
    extraction.extract._resolve_day_numbers) -- as its own leaf it would just
    repeat what's already in every one of that day's exercise leaves."""
    if extraction.get("category") in ("marketing", "program_structure"):
        return []

    builder = _CATEGORY_LEAF_BUILDERS.get(extraction.get("category"))
    leaves = builder(parent_chunk_id, doc_id, document_type, page, extraction) if builder else []

    if not leaves and _substantive_length(extraction["cleaned_text"]) >= _MIN_SUBSTANTIVE_LEN:
        embed_text = f"{extraction['page_topic']}: {_strip_links(extraction['cleaned_text'])}"
        leaves.append(_make_leaf(parent_chunk_id, doc_id, document_type, page, extraction, 0, "section", None, embed_text, "section"))

    return leaves


def _demo():
    assert _fmt_value({"low": 8.0, "high": 12.0}) == "8-12"
    assert _fmt_value({"low": 2.0, "high": None}) == "2"
    assert _fmt_value({"low": 5.5, "high": 5.5}) == "5.5"
    assert _fmt_value({"low": None, "high": None}) is None
    assert _fmt_value("60-90s") == "60-90s"
    assert _detail_bits({"exercise": "Squat", "reps": {"low": 8.0, "high": 12.0}, "video_url": "x"}, {"exercise", "video_url"}) == ["reps: 8-12"]


if __name__ == "__main__":
    _demo()
    print("ok")
