"""Normalize + extract: one Gemini call per batch of pages. No markdown-heading
dependency -- each page's topic/label is inferred by the model itself (see
models.TrainingSectionResult.page_topic).
"""
import os
import re

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from . import prompts
from .models import NutritionSectionResult, TrainingBatchResult, TRAINING_LIST_FIELDS

load_dotenv()

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# digit run, allowing "1,234"/"3.5" as one token without swallowing trailing punctuation
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_URL = re.compile(r"https?://\S+")

_model = None


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


async def normalize_and_extract_batch(document_type, sections):
    """sections: list of {"page": int, "text": str}. Returns {page: result}."""
    if document_type != "training_plan":
        raise NotImplementedError(f"batched extraction not implemented for {document_type!r}")

    structured_model = _get_model().with_structured_output(TrainingBatchResult)
    prompt = prompts.build_training_batch_prompt([(s["page"], s["text"]) for s in sections])
    print(f"💬 calling {MODEL_NAME} for pages {[s['page'] for s in sections]}...")
    batch_result = await structured_model.ainvoke(prompt)

    raw_text_by_page = {s["page"]: s["text"] for s in sections}
    return _match_batch_results(raw_text_by_page, batch_result.pages)



if __name__ == "__main__":
    print("ok")
