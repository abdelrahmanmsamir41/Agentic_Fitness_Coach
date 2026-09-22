"""Normalize + extract: one Gemini call per batch of pages. No markdown-heading
dependency -- each page's topic/label is inferred by the model itself (see
models.TrainingSectionResult.page_topic).
"""
import asyncio
import os
import re

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

import exercise_reference
from . import prompts
from .models import DayAssignmentResult, NutritionSectionResult, TrainingBatchResult, TRAINING_LIST_FIELDS

load_dotenv()

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
LLM_CONCURRENCY = 4  # cap on concurrent Gemini calls, to avoid rate-limit errors
PAGES_PER_LLM_CALL = 4  # pages batched into one extraction call, instead of one call per page

# digit run, allowing "1,234"/"3.5" as one token without swallowing trailing punctuation
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_URL = re.compile(r"https?://\S+")

_model = None
_llm_semaphore = asyncio.Semaphore(LLM_CONCURRENCY)


def _get_model():
    # lazy -- avoid a live credential requirement at import time.
    global _model
    if _model is None:
        _model = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.1)
    return _model


def _training_item(result):
    item = {"category": result.category, "page_topic": result.page_topic, "cleaned_text": result.cleaned_text}
    for field in TRAINING_LIST_FIELDS:
        item[field] = [e.model_dump() for e in getattr(result, field)]
    return item


def _numbers_and_urls(text):
    """Distinct tokens as sets, not multisets -- repeat counts can legitimately change.
    Numbers are counted outside of URLs only -- a URL's own digits (video ids, index=15,
    ...) aren't "data" the needs_review check should require cleaned_text to reproduce."""
    urls = set(_URL.findall(text))
    numbers = set(_NUMBER.findall(_URL.sub(" ", text)))
    return numbers, urls


def _build_result(raw_text, item):
    raw_numbers, raw_urls = _numbers_and_urls(raw_text)
    clean_numbers, clean_urls = _numbers_and_urls(item["cleaned_text"])
    needs_review = raw_numbers != clean_numbers or raw_urls != clean_urls

    return {
        **item,
        "cleaned_text": raw_text if needs_review else item["cleaned_text"],
        "needs_review": needs_review,
    }


def _match_batch_results(raw_text_by_page, page_results):
    """Map the model's per-page results back onto the pages that were sent,
    by page_index rather than list position -- the model isn't guaranteed to
    preserve order or return every page."""
    results = {}
    for page_result in page_results:
        item = _training_item(page_result)
        raw_text = raw_text_by_page.get(page_result.page_index, "")
        results[page_result.page_index] = _build_result(raw_text, item)

    missing = set(raw_text_by_page) - set(results)
    if missing:
        raise ValueError(f"batch extraction missing pages {sorted(missing)}")

    flagged = sorted(p for p, r in results.items() if r["needs_review"])
    if flagged:
        print(f"🚩 pages {flagged} flagged needs_review (cleaned text lost numbers/URLs)")
    return results


async def normalize_and_extract_batch(document_type, pages):
    """pages: list of {"page_index": int, "markdown": str}. Returns {page: result}."""
    if document_type != "training_plan":
        raise NotImplementedError(f"batched extraction not implemented for {document_type!r}")

    structured_model = _get_model().with_structured_output(TrainingBatchResult)
    prompt = prompts.build_training_batch_prompt([(p["page_index"], p["markdown"].strip()) for p in pages])
    print(f"💬 calling {MODEL_NAME} for pages {[p['page_index'] for p in pages]}...")
    batch_result = await structured_model.ainvoke(prompt)

    raw_text_by_page = {p["page_index"]: p["markdown"].strip() for p in pages}
    return _match_batch_results(raw_text_by_page, batch_result.pages)


async def assign_day_numbers(day_entries, page_topics):
    """day_entries: iterable of (day_number, target), from the split-table page's
    program_days. page_topics: iterable of (page_index, page_topic), for every
    weight_exercise/cardio page in the document. Returns {page_index: day_number},
    omitting pages the model couldn't place in any listed day.

    One classification call over page_topics only (not full page text) -- cheap,
    and independent of the extraction batching, so it doesn't need the split-table
    page's batch to run before any other page's batch.
    """
    day_entries, page_topics = list(day_entries), list(page_topics)
    if not day_entries or not page_topics:
        return {}

    structured_model = _get_model().with_structured_output(DayAssignmentResult)
    prompt = prompts.build_day_assignment_prompt(day_entries, page_topics)
    print(f"📅 assigning day numbers for {len(page_topics)} page(s) against {len(day_entries)} day(s)...")
    result = await structured_model.ainvoke(prompt)

    valid_days = {d for d, _ in day_entries}
    return {a.page_index: a.day_number for a in result.assignments if a.day_number in valid_days}


def _batch(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


async def _extract_one_batch(doc_id, document_type, batch_num, total_batches, pages):
    async with _llm_semaphore:
        print(f"🤖 [{doc_id}] extracting batch {batch_num}/{total_batches} ({len(pages)} pages)...")
        extraction_by_page = await normalize_and_extract_batch(document_type, pages)
    print(f"✅ [{doc_id}] batch {batch_num}/{total_batches} done")
    return extraction_by_page


async def _resolve_day_numbers(doc_id, all_extractions):
    """Stamps day_number onto every weight_exercise/cardio page's extraction,
    resolved from whichever page(s) got category=="program_structure"."""
    day_entries = sorted({
        (day["day_number"], day["target"])
        for extraction in all_extractions.values() if extraction["category"] == "program_structure"
        for day in extraction["program_days"]
    })
    page_topics = [
        (page, extraction["page_topic"])
        for page, extraction in all_extractions.items() if extraction["category"] in ("weight_exercise", "cardio")
    ]
    day_numbers = await assign_day_numbers(day_entries, page_topics)
    if day_numbers:
        print(f"📅 [{doc_id}] assigned day numbers to {len(day_numbers)}/{len(page_topics)} exercise/cardio page(s)")
    for page, day_number in day_numbers.items():
        all_extractions[page]["day_number"] = day_number


def _resolve_exercise_reference(doc_id, all_extractions):
    """Stamps ref_type/ref_bodypart/ref_equipment/ref_level/ref_match_score onto
    every weight_exercise entity, matched against the static megaGymDataset
    reference set (exercise_reference/ package) -- no LLM involved, purely local."""
    exercises = [
        entity
        for extraction in all_extractions.values() if extraction["category"] == "weight_exercise"
        for entity in extraction["weight_exercises"]
    ]
    if not exercises:
        return

    names = [e.get("exercise", "") for e in exercises]
    matches = exercise_reference.match_exercises(names)
    matched = 0
    for entity, match in zip(exercises, matches):
        if match:
            entity.update(match)
            matched += 1
    print(f"🏋️ [{doc_id}] matched {matched}/{len(exercises)} exercise(s) against the reference dataset")


async def extract_document(doc_id, document_type, pages):
    """Runs normalize_and_extract_batch concurrently across PAGES_PER_LLM_CALL-sized
    batches (capped at LLM_CONCURRENCY in flight), then resolves day_number for
    every weight_exercise/cardio page against whatever program_structure page
    turned up. pages: list of {"page_index": int, "markdown": str}. Returns {page: extraction}.
    """
    batches = _batch(pages, PAGES_PER_LLM_CALL)
    print(f"🤖 [{doc_id}] running LLM extraction on {len(batches)} batches ({LLM_CONCURRENCY} concurrent)...")
    results = await asyncio.gather(*(
        _extract_one_batch(doc_id, document_type, i + 1, len(batches), batch)
        for i, batch in enumerate(batches)
    ))

    all_extractions = {}
    for extraction_by_page in results:
        all_extractions.update(extraction_by_page)

    await _resolve_day_numbers(doc_id, all_extractions)
    _resolve_exercise_reference(doc_id, all_extractions)
    return all_extractions


if __name__ == "__main__":
    print("ok")
