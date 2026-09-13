"""Normalize + extract, one Gemini call per section (or batch of sections)."""
import os
import re
from typing import List, Optional

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

load_dotenv()

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
_model = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.0)

# digit run, allowing "1,234"/"3.5" as one token without swallowing trailing punctuation
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_URL = re.compile(r"https?://\S+")


class TrainingEntity(BaseModel):
    exercise: str
    sets: Optional[str] = None
    reps: Optional[str] = None
    rir: Optional[str] = None
    rest: Optional[str] = None
    instructions: Optional[str] = None
    video_url: Optional[str] = None


class NutritionEntity(BaseModel):
    meal: str
    food: Optional[str] = None
    quantity: Optional[str] = None
    unit: Optional[str] = None
    calories: Optional[str] = None
    macros: Optional[str] = None
    substitutions: Optional[str] = None


class TrainingSectionResult(BaseModel):
    cleaned_text: str
    entities: List[TrainingEntity]


class NutritionSectionResult(BaseModel):
    cleaned_text: str
    entities: List[NutritionEntity]


class TrainingBatchSection(BaseModel):
    heading: str
    cleaned_text: str
    entities: List[TrainingEntity]


class NutritionBatchSection(BaseModel):
    heading: str
    cleaned_text: str
    entities: List[NutritionEntity]


class TrainingBatchResult(BaseModel):
    sections: List[TrainingBatchSection]


class NutritionBatchResult(BaseModel):
    sections: List[NutritionBatchSection]


def _single_model_for(document_type):
    return TrainingSectionResult if document_type == "training_plan" else NutritionSectionResult


def _batch_model_for(document_type):
    return TrainingBatchResult if document_type == "training_plan" else NutritionBatchResult


_PROMPT = """You are cleaning up raw OCR output from a fitness coach's {kind} plan.

Raw section text (heading: "{heading}"):
---
{raw_text}
---

1. Reconstruct fragmented OCR lines into clean, readable Markdown prose/headings.
2. Extract each {kind} entry as a structured entity.

Rules: NEVER change any number, unit, set/rep count, RIR/RPE value, calorie/macro
value, or URL from the raw text. Preserve exercise/food names, coach-specific
terminology, and Arabic text as-is (do not translate). Only fix OCR fragmentation,
typos, and formatting.
"""

_BATCH_PROMPT = """You are cleaning up raw OCR output from a fitness coach's {kind} plan.

Below are {n} separate sections from the same document. Process each one
INDEPENDENTLY -- never move content, numbers, or entities from one section
into another, even if two sections look similar.

{sections_block}

For each section:
1. Reconstruct fragmented OCR lines into clean, readable Markdown prose/headings.
2. Extract each {kind} entry as a structured entity.
3. Return its ORIGINAL heading exactly as given above, in a "heading" field, so
   it can be matched back to the section it came from.

Rules: NEVER change any number, unit, set/rep count, RIR/RPE value, calorie/macro
value, or URL from that section's raw text. Preserve exercise/food names, coach-specific
terminology, and Arabic text as-is (do not translate). Only fix OCR fragmentation,
typos, and formatting. Return exactly {n} sections, in the same order given above.
"""


def _numbers_and_urls(text):
    """Distinct tokens as sets, not multisets -- repeat counts can legitimately change."""
    return set(_NUMBER.findall(text)), set(_URL.findall(text))


def _build_result(heading, raw_text, item):
    # heading is included since the model reproduces it inside cleaned_text too
    raw_numbers, raw_urls = _numbers_and_urls(f"{heading}\n{raw_text}")
    clean_numbers, clean_urls = _numbers_and_urls(item["cleaned_text"])
    needs_review = raw_numbers != clean_numbers or raw_urls != clean_urls

    return {
        "cleaned_text": raw_text if needs_review else item["cleaned_text"],
        "entities": item["entities"],
        "needs_review": needs_review,
        "confidence": 1.0,
    }


async def normalize_and_extract(document_type, heading, raw_text):
    kind = "workout" if document_type == "training_plan" else "meal"
    structured_model = _model.with_structured_output(_single_model_for(document_type))
    result = await structured_model.ainvoke(_PROMPT.format(kind=kind, heading=heading, raw_text=raw_text))
    item = {"cleaned_text": result.cleaned_text, "entities": [e.model_dump() for e in result.entities]}
    return _build_result(heading, raw_text, item)


def _match_returned_item(returned, index, heading):
    """Match by position, falling back to heading lookup if order didn't hold."""
    if index < len(returned) and returned[index].get("heading") == heading:
        return returned[index]
    return next((r for r in returned if r.get("heading") == heading), None)


async def normalize_and_extract_batch(document_type, sections):
    """Like normalize_and_extract but for a list of {heading, text} sections in one call."""
    kind = "workout" if document_type == "training_plan" else "meal"
    sections_block = "\n\n".join(
        f'--- Section {i + 1} (heading: "{s["heading"]}") ---\n{s["text"]}' for i, s in enumerate(sections)
    )
    structured_model = _model.with_structured_output(_batch_model_for(document_type))
    result = await structured_model.ainvoke(
        _BATCH_PROMPT.format(kind=kind, n=len(sections), sections_block=sections_block)
    )
    returned = [
        {"heading": s.heading, "cleaned_text": s.cleaned_text, "entities": [e.model_dump() for e in s.entities]}
        for s in result.sections
    ]

    results = []
    for i, s in enumerate(sections):
        item = _match_returned_item(returned, i, s["heading"])
        if item is None:
            results.append({"cleaned_text": s["text"], "entities": [], "needs_review": True, "confidence": 1.0})
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

    print("normalize_extract self-check passed (no network call)")


if __name__ == "__main__":
    _demo()
