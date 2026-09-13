"""Split merged per-page markdown into sections (Day / Meal-block), and build
the parent/leaf chunk records described in the ingestion plan.
"""
import re

_HEADING = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)

# PaddleOCR-VL repeats running headers per page, so heading-splitting alone
# produces many near-empty sections (just a repeated title, no body) -- each
# one would otherwise cost a full, wasted Gemini call. Anything under this
# length gets folded into the section that follows it instead of being sent
# to the LLM on its own.
_MIN_SECTION_LEN = 30

# For the "whole section" fallback chunk (see build_chunks): strip markdown
# headings, links, and our own injected "Links on this page:" boilerplate
# before measuring content -- a section that's just a heading fragment plus a
# link (no real prose) must not become its own embeddable chunk. Short,
# near-empty text can score deceptively high on cosine similarity against
# unrelated queries, actively hurting retrieval instead of just adding noise.
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


def _merge_tiny_sections(sections):
    if not sections:
        return sections
    merged = []
    carry = ""
    for s in sections:
        text = f"{carry}\n\n{s['text']}".strip() if carry else s["text"]
        carry = ""
        if len(text) < _MIN_SECTION_LEN:
            carry = text
            continue
        merged.append({**s, "text": text})
    if carry:
        if merged:
            merged[-1] = {**merged[-1], "text": f"{merged[-1]['text']}\n\n{carry}"}
        else:
            merged.append({"heading": sections[-1]["heading"], "text": carry, "page": sections[-1]["page"]})
    return merged


def split_sections(pages):
    """Concatenate all page markdown, split on '#'/'##'/'###' headings.

    Returns a list of {"heading": str, "text": str, "page": int} in document order.
    The first page a heading appears on is recorded for provenance.
    """
    full_text = ""
    page_at_offset = []  # (char_offset, page_index)
    for page in pages:
        page_at_offset.append((len(full_text), page["page_index"]))
        full_text += page["markdown"] + "\n"

    matches = list(_HEADING.finditer(full_text))
    if not matches:
        return [{"heading": "Document", "text": full_text.strip(), "page": pages[0]["page_index"] if pages else 0}]

    def page_for_offset(offset):
        page = page_at_offset[0][1]
        for start, p in page_at_offset:
            if start > offset:
                break
            page = p
        return page

    sections = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        sections.append({
            "heading": m.group(1).strip(),
            "text": full_text[start:end].strip(),
            "page": page_for_offset(start),
        })
    return _merge_tiny_sections(sections)


def _slug(text):
    # Arabic (or any non-Latin-script) text always produces an empty slug here,
    # since only ASCII a-z0-9 survive -- "section" was a bad fallback for that:
    # it collided with the "00-section" fixed suffix used for whole-section
    # fallback chunks below and once caused a cleanup script to delete real
    # Arabic-named entities by mistake. "unnamed" doesn't look like anything
    # else in this file.
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "unnamed"


def build_chunks(doc_id, document_type, section, extraction, section_index):
    """Build one parent record (full section) + one leaf chunk per extracted entity.

    extraction = {"cleaned_text": str, "entities": [dict, ...], "needs_review": bool, "confidence": float}
    entities' key entry names differ by document_type (see normalize_extract.py schemas).

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

    name_key = "exercise" if document_type == "training_plan" else "meal"
    leaves = []
    for idx, entity in enumerate(extraction["entities"]):
        name = entity.get(name_key) or f"item-{idx}"
        chunk_id = f"{parent_chunk_id}:{idx:02d}-{_slug(name)}"
        detail_bits = [f"{k}: {v}" for k, v in entity.items() if k not in (name_key, "video_url") and v not in (None, "")]
        embed_text = f"{section['heading']} > {name}: " + ", ".join(detail_bits)
        leaves.append({
            "chunk_id": chunk_id,
            "parent_chunk_id": parent_chunk_id,
            "document_id": doc_id,
            "document_type": document_type,
            "page": section["page"],
            "heading": section["heading"],
            "entity": entity,
            "embed_text": embed_text,
            "confidence": extraction.get("confidence", 1.0),
            "needs_review": extraction["needs_review"],
        })

    if not leaves and _substantive_length(extraction["cleaned_text"]) >= _MIN_SUBSTANTIVE_LEN:
        # Explanatory/definitional sections (RIR, SUPERSETS, FAQS, ...) have no
        # extractable exercise/meal entities, so the loop above produces
        # nothing -- without this, the section is saved as a parent in Mongo
        # but never embedded, making it permanently unretrievable even though
        # it's exactly what answers "what does the coach mean by RIR?". But a
        # section that's just a heading fragment + a link (no real prose)
        # must NOT become its own chunk -- see _substantive_length above.
        chunk_id = f"{parent_chunk_id}:00-section"
        embed_text = f"{section['heading']}: {_strip_links(extraction['cleaned_text'])}"
        leaves.append({
            "chunk_id": chunk_id,
            "parent_chunk_id": parent_chunk_id,
            "document_id": doc_id,
            "document_type": document_type,
            "page": section["page"],
            "heading": section["heading"],
            "entity": None,
            "embed_text": embed_text,
            "confidence": extraction.get("confidence", 1.0),
            "needs_review": extraction["needs_review"],
        })

    return parent, leaves


def _demo():
    pages = [
        {"page_index": 0, "markdown": "## Day 1\n" + "text a " * 10},
        {"page_index": 1, "markdown": "## Day 2\n" + "text b " * 10},
    ]
    sections = split_sections(pages)
    assert [s["heading"] for s in sections] == ["Day 1", "Day 2"]
    assert sections[0]["page"] == 0 and sections[1]["page"] == 1

    # A tiny, heading-only "section" (running header with no real body) must
    # get folded into the next section instead of becoming its own LLM call.
    tiny_pages = [{"page_index": 0, "markdown": "## REST DAY\n\n## TRAINING LOG\n" + "real content " * 10}]
    tiny_sections = split_sections(tiny_pages)
    assert len(tiny_sections) == 1
    assert tiny_sections[0]["heading"] == "TRAINING LOG"
    assert "REST DAY" in tiny_sections[0]["text"]

    extraction = {"cleaned_text": "Day 2 cleaned", "entities": [{"exercise": "Lat Pulldown", "sets": 3}], "needs_review": False, "confidence": 0.97}
    parent, leaves = build_chunks("doc1", "training_plan", sections[1], extraction, section_index=1)
    assert parent["parent_chunk_id"] == "doc1:0001-day-2"
    assert leaves[0]["chunk_id"] == "doc1:0001-day-2:00-lat-pulldown"

    # Two sections with the identical (generic/repeated) heading must not collide.
    generic = {"heading": "Section", "text": "x", "page": 0}
    real_prose = "how many reps you have got left in the tank before hitting failure, basically"
    p0, leaves0 = build_chunks("doc2", "recipe_book", generic, {"cleaned_text": real_prose, "entities": [], "needs_review": False, "confidence": 1.0}, section_index=0)
    p1, _ = build_chunks("doc2", "recipe_book", generic, {"cleaned_text": real_prose, "entities": [], "needs_review": False, "confidence": 1.0}, section_index=1)
    assert p0["parent_chunk_id"] != p1["parent_chunk_id"]
    assert "Lat Pulldown" in leaves[0]["embed_text"]

    # A section with zero extractable entities but real prose (e.g. "RIR")
    # must still produce one embeddable leaf -- otherwise it's saved as a
    # parent in Mongo but never searchable in Qdrant.
    assert len(leaves0) == 1
    assert leaves0[0]["entity"] is None
    assert real_prose in leaves0[0]["embed_text"]

    # A section that's just a heading fragment + an injected link (no real
    # prose) must NOT become its own chunk -- it's noise, not content, and
    # short noisy text can score deceptively high in embedding similarity.
    noisy = {"cleaned_text": "## chest & back. _Links on this page:_ [TRAINING LOG](https://example.com/x)",
             "entities": [], "needs_review": False, "confidence": 1.0}
    _, noisy_leaves = build_chunks("doc3", "training_plan", generic, noisy, section_index=0)
    assert noisy_leaves == []

    print("chunk self-check passed")


if __name__ == "__main__":
    _demo()
