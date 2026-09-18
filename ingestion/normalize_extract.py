"""Normalize + extract, one Gemini call per section (or batch of sections)."""
import os
import re
from typing import List, Literal, Optional

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

load_dotenv()

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
_model = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.0)

# digit run, allowing "1,234"/"3.5" as one token without swallowing trailing punctuation
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_URL = re.compile(r"https?://\S+")


class NutritionEntity(BaseModel):
    meal: str
    food: Optional[str] = None
    quantity: Optional[str] = None
    unit: Optional[str] = None
    calories: Optional[str] = None
    macros: Optional[str] = None
    substitutions: Optional[str] = None


class NutritionSectionResult(BaseModel):
    cleaned_text: str
    entities: List[NutritionEntity]


class NutritionBatchSection(BaseModel):
    heading: str
    cleaned_text: str
    entities: List[NutritionEntity]


class NutritionBatchResult(BaseModel):
    sections: List[NutritionBatchSection]


# --- Training-plan section categories -------------------------------------
# A training plan mixes several distinct kinds of content on different pages
# (exercise tables, a day/split overview, a glossary of coaching terms,
# general advice, cardio, marketing) -- one generic "exercise" schema for the
# whole document either force-fits everything into exercise fields (cardio,
# glossary) or throws it all into an untyped fallback text blob. Classifying
# each section and extracting into the matching typed list keeps embeddings
# specific (e.g. a glossary leaf is just "RIR: how many reps...", not buried
# in a whole-section paragraph) instead of generic.

SectionCategory = Literal[
    "weight_exercise", "cardio", "glossary", "advice", "program_structure", "marketing", "other"
]


class WeightExerciseEntity(BaseModel):
    exercise: str
    working_sets: Optional[str] = None
    reps: Optional[str] = None
    warmup_sets: Optional[str] = None
    rir: Optional[str] = None
    rest: Optional[str] = None
    instructions: Optional[str] = None
    video_url: Optional[str] = None


class CardioEntity(BaseModel):
    activity: str
    duration: Optional[str] = None
    speed: Optional[str] = None
    incline: Optional[str] = None
    timing: Optional[str] = None  # e.g. "post-workout"
    instructions: Optional[str] = None
    video_url: Optional[str] = None


class GlossaryTermEntity(BaseModel):
    term: str
    explanation: str


class AdviceEntity(BaseModel):
    topic: Optional[str] = None
    text: str


class ProgramDayEntity(BaseModel):
    day_number: int
    target: str  # muscle group(s) targeted, or "rest"


class TrainingSectionResult(BaseModel):
    category: SectionCategory
    cleaned_text: str
    weight_exercises: List[WeightExerciseEntity] = Field(default_factory=list)
    cardio_entries: List[CardioEntity] = Field(default_factory=list)
    glossary_terms: List[GlossaryTermEntity] = Field(default_factory=list)
    advice_items: List[AdviceEntity] = Field(default_factory=list)
    program_days: List[ProgramDayEntity] = Field(default_factory=list)


class TrainingBatchSection(TrainingSectionResult):
    heading: str


class TrainingBatchResult(BaseModel):
    sections: List[TrainingBatchSection]


_TRAINING_CATEGORIES = """Classify this section into exactly one category, and extract
entries into ONLY the list matching that category (leave every other list empty):
- "weight_exercise": a table/list of gym exercises with sets/reps/rest.
- "cardio": cardio activities (treadmill, running, cycling, ...) with duration/speed/incline.
- "glossary": defines fitness terms/abbreviations (e.g. RIR, W.U Sets, Supersets, Partials).
- "advice": general coaching advice, rules, mistakes to avoid, or motivation -- not tied
  to a specific exercise or cardio session.
- "program_structure": maps training days to a muscle group/split, or marks a day as rest.
- "marketing": promo codes, discounts, "order now" calls to action -- not coaching content.
- "other": anything that doesn't fit the above (e.g. a bare cover title with no body)."""

_PROMPT = """You are cleaning up raw OCR output from a fitness coach's meal plan.

Raw section text (heading: "{heading}"):
---
{raw_text}
---

1. Reconstruct fragmented OCR lines into clean, readable Markdown prose/headings.
2. Extract each meal entry as a structured entity.

Rules: NEVER change any number, unit, set/rep count, RIR/RPE value, calorie/macro
value, or URL from the raw text. Preserve exercise/food names, coach-specific
terminology, and Arabic text as-is (do not translate). Only fix OCR fragmentation,
typos, and formatting.
"""

_TRAINING_PROMPT = """You are cleaning up raw OCR output from a fitness coach's training plan.

Raw section text (heading: "{heading}"):
---
{raw_text}
---

1. Reconstruct fragmented OCR lines into clean, readable Markdown prose/headings.
2. """ + _TRAINING_CATEGORIES + """

Rules: NEVER change any number, unit, set/rep count, RIR/RPE value, or URL from the raw
text. Preserve exercise/food names, coach-specific terminology, and Arabic text as-is
(do not translate). Only fix OCR fragmentation, typos, and formatting.
"""

_BATCH_PROMPT = """You are cleaning up raw OCR output from a fitness coach's meal plan.

Below are {n} separate sections from the same document. Process each one
INDEPENDENTLY -- never move content, numbers, or entities from one section
into another, even if two sections look similar.

{sections_block}

For each section:
1. Reconstruct fragmented OCR lines into clean, readable Markdown prose/headings.
2. Extract each meal entry as a structured entity.
3. Return its ORIGINAL heading exactly as given above, in a "heading" field, so
   it can be matched back to the section it came from.

Rules: NEVER change any number, unit, set/rep count, RIR/RPE value, calorie/macro
value, or URL from that section's raw text. Preserve exercise/food names, coach-specific
terminology, and Arabic text as-is (do not translate). Only fix OCR fragmentation,
typos, and formatting. Return exactly {n} sections, in the same order given above.
"""

_TRAINING_BATCH_PROMPT = """You are cleaning up raw OCR output from a fitness coach's training plan.

Below are {n} separate sections from the same document. Process each one
INDEPENDENTLY -- never move content, numbers, or entities from one section
into another, even if two sections look similar.

{sections_block}

For each section:
1. Reconstruct fragmented OCR lines into clean, readable Markdown prose/headings.
2. """ + _TRAINING_CATEGORIES + """
3. Return its ORIGINAL heading exactly as given above, in a "heading" field, so
   it can be matched back to the section it came from.

Rules: NEVER change any number, unit, set/rep count, RIR/RPE value, or URL from that
section's raw text. Preserve exercise/food names, coach-specific terminology, and
Arabic text as-is (do not translate). Only fix OCR fragmentation, typos, and
formatting. Return exactly {n} sections, in the same order given above.
"""

_TRAINING_LIST_FIELDS = ("weight_exercises", "cardio_entries", "glossary_terms", "advice_items", "program_days")


def _training_item(result):
    item = {"category": result.category, "cleaned_text": result.cleaned_text}
    for field in _TRAINING_LIST_FIELDS:
        item[field] = [e.model_dump() for e in getattr(result, field)]
    return item


def _sections_block(sections):
    return "\n\n".join(
        f'--- Section {i + 1} (heading: "{s["heading"]}") ---\n{s["text"]}' for i, s in enumerate(sections)
    )


def _numbers_and_urls(text):
    """Distinct tokens as sets, not multisets -- repeat counts can legitimately change."""
    return set(_NUMBER.findall(text)), set(_URL.findall(text))


def _build_result(heading, raw_text, item):
    # heading is included since the model reproduces it inside cleaned_text too
    raw_numbers, raw_urls = _numbers_and_urls(f"{heading}\n{raw_text}")
    clean_numbers, clean_urls = _numbers_and_urls(item["cleaned_text"])
    needs_review = raw_numbers != clean_numbers or raw_urls != clean_urls

    return {
        **item,
        "cleaned_text": raw_text if needs_review else item["cleaned_text"],
        "needs_review": needs_review,
        "confidence": 1.0,
    }


async def normalize_and_extract(document_type, heading, raw_text):
    if document_type == "training_plan":
        structured_model = _model.with_structured_output(TrainingSectionResult)
        result = await structured_model.ainvoke(_TRAINING_PROMPT.format(heading=heading, raw_text=raw_text))
        item = _training_item(result)
    else:
        structured_model = _model.with_structured_output(NutritionSectionResult)
        result = await structured_model.ainvoke(_PROMPT.format(heading=heading, raw_text=raw_text))
        item = {"cleaned_text": result.cleaned_text, "entities": [e.model_dump() for e in result.entities]}
    return _build_result(heading, raw_text, item)


def _match_returned_item(returned, index, heading):
    """Match by position, falling back to heading lookup if order didn't hold."""
    if index < len(returned) and returned[index].get("heading") == heading:
        return returned[index]
    return next((r for r in returned if r.get("heading") == heading), None)


async def normalize_and_extract_batch(document_type, sections):
    """Like normalize_and_extract but for a list of {heading, text} sections in one call."""
    sections_block = _sections_block(sections)

    if document_type == "training_plan":
        structured_model = _model.with_structured_output(TrainingBatchResult)
        result = await structured_model.ainvoke(
            _TRAINING_BATCH_PROMPT.format(n=len(sections), sections_block=sections_block)
        )
        returned = [{"heading": s.heading, **_training_item(s)} for s in result.sections]
    else:
        structured_model = _model.with_structured_output(NutritionBatchResult)
        result = await structured_model.ainvoke(
            _BATCH_PROMPT.format(n=len(sections), sections_block=sections_block)
        )
        returned = [
            {"heading": s.heading, "cleaned_text": s.cleaned_text, "entities": [e.model_dump() for e in s.entities]}
            for s in result.sections
        ]

    results = []
    for i, s in enumerate(sections):
        item = _match_returned_item(returned, i, s["heading"])
        if item is None:
            fallback = {"cleaned_text": s["text"]}
            if document_type == "training_plan":
                fallback["category"] = "other"
                fallback.update({field: [] for field in _TRAINING_LIST_FIELDS})
            else:
                fallback["entities"] = []
            results.append({**fallback, "needs_review": True, "confidence": 1.0})
            continue
        results.append(_build_result(s["heading"], s["text"], item))
    return results


def _demo():
    raw_numbers, raw_urls = _numbers_and_urls("3x8-12 RIR 1-2 https://x.com")
    assert raw_numbers == {"3", "8", "12", "1", "2"}
    assert raw_urls == {"https://x.com"}

    same_numbers, same_urls = _numbers_and_urls("Sets 3, reps 8-12, reps 8 to 12, RIR 1-2, see https://x.com")
    assert same_numbers == raw_numbers and same_urls == raw_urls

    changed_numbers, _ = _numbers_and_urls("3x8-10 RIR 1-2 https://x.com")
    assert changed_numbers != raw_numbers

    returned = [{"heading": "A", "cleaned_text": "a"}, {"heading": "B", "cleaned_text": "b"}]
    assert _match_returned_item(returned, 0, "A") == returned[0]
    reordered = [{"heading": "B", "cleaned_text": "b"}, {"heading": "A", "cleaned_text": "a"}]
    assert _match_returned_item(reordered, 0, "A") == reordered[1]
    assert _match_returned_item(reordered, 0, "C") is None

    # _build_result must pass through category-specific fields untouched, not
    # just the old fixed "entities" key -- every training category depends on this.
    item = {"category": "glossary", "cleaned_text": "RIR means reps in reserve.",
            "glossary_terms": [{"term": "RIR", "explanation": "reps in reserve"}]}
    built = _build_result("Glossary", "RIR means reps in reserve.", item)
    assert built["category"] == "glossary"
    assert built["glossary_terms"] == item["glossary_terms"]
    assert built["needs_review"] is False

    print("normalize_extract self-check passed (no network call)")


if __name__ == "__main__":
    _demo()
