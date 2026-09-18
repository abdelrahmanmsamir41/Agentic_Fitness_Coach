"""Split per-page markdown into one section per page, and build the
parent/leaf chunk records described in the ingestion plan.
"""
import re

_HEADING = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)

# Strip markdown headings/links/boilerplate before measuring "is there real
# content here" -- a heading fragment plus a link alone must not embed.
_MD_HEADING_LINE = re.compile(r"^#{1,3}\s.*$", re.MULTILINE)
_MD_LINK = re.compile(r"\[[^\]]*\]\([^)]*\)")
_LINKS_BOILERPLATE = re.compile(r"_?Links on this page:_?", re.IGNORECASE)
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

    Returns a list of {"heading": str, "text": str, "page": int} in document order.
    """
    sections = []
    for page in pages:
        match = _HEADING.search(page["markdown"])
        heading = match.group(1).strip() if match else f"Page {page['page_index'] + 1}"
        sections.append({"heading": heading, "text": page["markdown"].strip(), "page": page["page_index"]})
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


def _make_leaf(parent_chunk_id, doc_id, document_type, section, extraction, idx, suffix_key, entity, embed_text):
    return {
        "chunk_id": f"{parent_chunk_id}:{idx:02d}-{_slug(suffix_key)}",
        "parent_chunk_id": parent_chunk_id,
        "document_id": doc_id,
        "document_type": document_type,
        "page": section["page"],
        "heading": section["heading"],
        "entity": entity,
        "embed_text": embed_text,
        "confidence": extraction.get("confidence", 1.0),
        "needs_review": extraction["needs_review"],
    }


# Each training-plan category gets its own embed_text shape instead of one
# generic "heading > name: k: v, k: v" -- a glossary leaf reads best as a bare
# "term: explanation" (no heading noise), while program-structure leaves need
# the day number front and center since that's what a client/coach searches by.

def _weight_exercise_leaves(parent_chunk_id, doc_id, document_type, section, extraction):
    leaves = []
    for idx, entity in enumerate(extraction["weight_exercises"]):
        name = entity.get("exercise") or f"item-{idx}"
        detail_bits = _detail_bits(entity, {"exercise", "video_url"})
        embed_text = f"{section['heading']} > {name}: " + ", ".join(detail_bits)
        leaves.append(_make_leaf(parent_chunk_id, doc_id, document_type, section, extraction, idx, name, entity, embed_text))
    return leaves


def _cardio_leaves(parent_chunk_id, doc_id, document_type, section, extraction):
    leaves = []
    for idx, entity in enumerate(extraction["cardio_entries"]):
        name = entity.get("activity") or f"cardio-{idx}"
        detail_bits = _detail_bits(entity, {"activity", "video_url"})
        embed_text = f"{section['heading']} > Cardio - {name}: " + ", ".join(detail_bits)
        leaves.append(_make_leaf(parent_chunk_id, doc_id, document_type, section, extraction, idx, name, entity, embed_text))
    return leaves


def _glossary_leaves(parent_chunk_id, doc_id, document_type, section, extraction):
    leaves = []
    for idx, entity in enumerate(extraction["glossary_terms"]):
        term = entity.get("term") or f"term-{idx}"
        embed_text = f"{term}: {entity.get('explanation', '')}"
        leaves.append(_make_leaf(parent_chunk_id, doc_id, document_type, section, extraction, idx, term, entity, embed_text))
    return leaves


def _advice_leaves(parent_chunk_id, doc_id, document_type, section, extraction):
    leaves = []
    for idx, entity in enumerate(extraction["advice_items"]):
        topic = entity.get("topic")
        text = entity.get("text", "")
        embed_text = f"{section['heading']} - {topic}: {text}" if topic else f"{section['heading']}: {text}"
        leaves.append(_make_leaf(parent_chunk_id, doc_id, document_type, section, extraction, idx, topic or f"advice-{idx}", entity, embed_text))
    return leaves


def _program_structure_leaves(parent_chunk_id, doc_id, document_type, section, extraction):
    leaves = []
    for idx, entity in enumerate(extraction["program_days"]):
        day = entity.get("day_number", idx)
        target = entity.get("target", "")
        embed_text = f"{section['heading']} > Day {day}: {target}"
        leaves.append(_make_leaf(parent_chunk_id, doc_id, document_type, section, extraction, idx, f"day-{day}", entity, embed_text))
    return leaves


_CATEGORY_LEAF_BUILDERS = {
    "weight_exercise": _weight_exercise_leaves,
    "cardio": _cardio_leaves,
    "glossary": _glossary_leaves,
    "advice": _advice_leaves,
    "program_structure": _program_structure_leaves,
}


def build_chunks(doc_id, document_type, section, extraction, section_index):
    """Build one parent record (full section) + one leaf chunk per extracted entity.

    For document_type == "training_plan", extraction carries a "category" (see
    normalize_extract.SectionCategory) selecting which typed list
    (weight_exercises/cardio_entries/glossary_terms/advice_items/program_days)
    to build leaves from. Other document types keep the original generic
    "entities" list (see normalize_extract.py schemas).

    section_index is this section's position within the document (from
    enumerate(split_sections(...))). Headings are NOT reliably unique -- some
    source documents repeat a generic heading (e.g. a recipe book's recurring
    "Section" label) across genuinely different content, which would otherwise
    collide different real chunks onto the same id and silently overwrite one
    with another in Qdrant.
    """
    section_slug = _slug(section["heading"])
    parent_chunk_id = f"{doc_id}:{section_index:04d}-{section_slug}"

    parent = {
        "parent_chunk_id": parent_chunk_id,
        "document_id": doc_id,
        "document_type": document_type,
        "heading": section["heading"],
        "page": section["page"],
        "text": extraction["cleaned_text"],
        "needs_review": extraction["needs_review"],
    }

    # Promo/marketing content ("use code X for a discount") isn't coaching
    # knowledge -- embedding it would let a marketing blurb surface as the
    # answer to a real question. Keep the parent for the record, embed nothing.
    if extraction.get("category") == "marketing":
        return parent, []

    if document_type == "training_plan":
        builder = _CATEGORY_LEAF_BUILDERS.get(extraction.get("category"))
        leaves = builder(parent_chunk_id, doc_id, document_type, section, extraction) if builder else []
    else:
        leaves = []
        for idx, entity in enumerate(extraction["entities"]):
            name = entity.get("meal") or f"item-{idx}"
            detail_bits = _detail_bits(entity, {"meal", "video_url"})
            embed_text = f"{section['heading']} > {name}: " + ", ".join(detail_bits)
            leaves.append(_make_leaf(parent_chunk_id, doc_id, document_type, section, extraction, idx, name, entity, embed_text))

    if not leaves and _substantive_length(extraction["cleaned_text"]) >= _MIN_SUBSTANTIVE_LEN:
        # Explanatory/definitional sections (RIR, SUPERSETS, FAQS, ...) have no
        # extractable exercise/meal entities, so the loop above produces
        # nothing -- without this, the section is saved as a parent in Mongo
        # but never embedded, making it permanently unretrievable even though
        # it's exactly what answers "what does the coach mean by RIR?". But a
        # section that's just a heading fragment + a link (no real prose)
        # must NOT become its own chunk -- see _substantive_length above.
        embed_text = f"{section['heading']}: {_strip_links(extraction['cleaned_text'])}"
        leaves.append(_make_leaf(parent_chunk_id, doc_id, document_type, section, extraction, 0, "section", None, embed_text))

    return parent, leaves


def _demo():
    pages = [
        {"page_index": 0, "markdown": "## Day 1\n" + "text a " * 10},
        {"page_index": 1, "markdown": "## Day 2\n" + "text b " * 10},
    ]
    sections = split_sections(pages)
    assert [s["heading"] for s in sections] == ["Day 1", "Day 2"]
    assert sections[0]["page"] == 0 and sections[1]["page"] == 1

    # A near-empty page must stay its own section, never merged into a neighbor.
    tiny_pages = [{"page_index": 0, "markdown": "## REST DAY"}, {"page_index": 1, "markdown": "## TRAINING LOG\nx"}]
    tiny_sections = split_sections(tiny_pages)
    assert len(tiny_sections) == 2
    assert tiny_sections[0]["heading"] == "REST DAY" and tiny_sections[1]["heading"] == "TRAINING LOG"

    # A page with no heading at all still gets its own section (fallback label).
    headingless = split_sections([{"page_index": 5, "markdown": "just some text, no markdown heading"}])
    assert headingless[0]["heading"] == "Page 6"

    def training_extraction(category, **lists):
        base = {field: [] for field in
                ("weight_exercises", "cardio_entries", "glossary_terms", "advice_items", "program_days")}
        base.update(lists)
        return {"category": category, "cleaned_text": "cleaned", "needs_review": False, "confidence": 0.97, **base}

    # weight_exercise category -> one leaf per exercise, embed_text carries heading + details.
    extraction = training_extraction("weight_exercise", weight_exercises=[{"exercise": "Lat Pulldown", "working_sets": "3"}])
    parent, leaves = build_chunks("doc1", "training_plan", sections[1], extraction, section_index=1)
    assert parent["parent_chunk_id"] == "doc1:0001-day-2"
    assert leaves[0]["chunk_id"] == "doc1:0001-day-2:00-lat-pulldown"
    assert "Lat Pulldown" in leaves[0]["embed_text"] and "working_sets: 3" in leaves[0]["embed_text"]

    # glossary category -> "term: explanation", no heading prefix (self-contained, not day-specific).
    glossary_section = {"heading": "FAQS", "text": "x", "page": 2}
    extraction = training_extraction("glossary", glossary_terms=[{"term": "RIR", "explanation": "reps in reserve"}])
    _, leaves = build_chunks("doc1", "training_plan", glossary_section, extraction, section_index=2)
    assert leaves[0]["embed_text"] == "RIR: reps in reserve"

    # cardio category -> distinct field shape (duration/speed/incline) from weight_exercise.
    cardio_section = {"heading": "Cardio", "text": "x", "page": 16}
    extraction = training_extraction("cardio", cardio_entries=[{"activity": "Treadmill", "duration": "35", "incline": "7"}])
    _, leaves = build_chunks("doc1", "training_plan", cardio_section, extraction, section_index=3)
    assert "Cardio - Treadmill" in leaves[0]["embed_text"] and "incline: 7" in leaves[0]["embed_text"]

    # program_structure category -> one leaf per day, day number front and center.
    split_section = {"heading": "Training Split", "text": "x", "page": 4}
    extraction = training_extraction("program_structure", program_days=[
        {"day_number": 1, "target": "chest & back"}, {"day_number": 4, "target": "rest"},
    ])
    _, leaves = build_chunks("doc1", "training_plan", split_section, extraction, section_index=4)
    assert len(leaves) == 2
    assert leaves[0]["embed_text"] == "Training Split > Day 1: chest & back"

    # marketing category must never be embedded, even with long, substantive-looking text.
    promo_section = {"heading": "Promo Codes", "text": "x", "page": 18}
    extraction = training_extraction("marketing")
    extraction["cleaned_text"] = "Use my promo code SHADY10 for a discount on supplements and gym clothing."
    _, leaves = build_chunks("doc1", "training_plan", promo_section, extraction, section_index=5)
    assert leaves == []

    # Two sections with the identical (generic/repeated) heading must not collide.
    generic = {"heading": "Section", "text": "x", "page": 0}
    real_prose = "how many reps you have got left in the tank before hitting failure, basically"
    p0, leaves0 = build_chunks("doc2", "recipe_book", generic, {"cleaned_text": real_prose, "entities": [], "needs_review": False, "confidence": 1.0}, section_index=0)
    p1, _ = build_chunks("doc2", "recipe_book", generic, {"cleaned_text": real_prose, "entities": [], "needs_review": False, "confidence": 1.0}, section_index=1)
    assert p0["parent_chunk_id"] != p1["parent_chunk_id"]

    # A section with zero extractable entities but real prose (e.g. "RIR")
    # must still produce one embeddable leaf -- otherwise it's saved as a
    # parent in Mongo but never searchable in Qdrant. Applies to non-training
    # document types (recipe_book/diet_plan) that still use the generic path.
    assert len(leaves0) == 1
    assert leaves0[0]["entity"] is None
    assert real_prose in leaves0[0]["embed_text"]

    # A section that's just a heading fragment + an injected link (no real
    # prose) must NOT become its own chunk -- it's noise, not content, and
    # short noisy text can score deceptively high in embedding similarity.
    noisy = training_extraction("advice")
    noisy["cleaned_text"] = "## chest & back. _Links on this page:_ [TRAINING LOG](https://example.com/x)"
    _, noisy_leaves = build_chunks("doc3", "training_plan", generic, noisy, section_index=0)
    assert noisy_leaves == []

    print("chunk self-check passed")


if __name__ == "__main__":
    _demo()
