"""Batched Gemini extraction: one call per PAGES_PER_LLM_CALL pages, each page tagged with
page_index so results match back regardless of model ordering. Cross-page enrichment is in enrich.py.
"""
import asyncio
import os
from itertools import batched

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from lazy import lazy

from . import prompts
from .models import NutritionSectionResult, TrainingBatchResult, TRAINING_LIST_FIELDS

load_dotenv()

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
LLM_CONCURRENCY = 4  # higher hits Gemini rate limits
PAGES_PER_LLM_CALL = 4

_llm_semaphore = asyncio.Semaphore(LLM_CONCURRENCY)

get_model = lazy(lambda: ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.1))


def _to_training_item(result, raw_text):
    item = {"category": result.category, "page_topic": result.page_topic, "page_text": raw_text}
    for field in TRAINING_LIST_FIELDS:
        item[field] = [entity.model_dump() for entity in getattr(result, field)]
    return item


def match_batch_results(raw_text_by_page, page_results):
    """Map the model's per-page results back onto the pages that were sent,
    by page_index rather than list position -- the model isn't guaranteed to
    preserve order or return every page."""
    results = {
        page_result.page_index: _to_training_item(page_result, raw_text_by_page.get(page_result.page_index, ""))
        for page_result in page_results
    }

    missing = set(raw_text_by_page) - set(results)
    if missing:
        raise ValueError(f"batch extraction missing pages {sorted(missing)}")

    return results


async def extract_batch(document_type, pages):
    """pages: list of {"page_index": int, "markdown": str}. Returns {page: result}."""
    if document_type != "training_plan":
        raise NotImplementedError(f"batched extraction not implemented for {document_type!r}")

    structured_model = get_model().with_structured_output(TrainingBatchResult)
    prompt = prompts.build_training_batch_prompt([(page["page_index"], page["markdown"].strip()) for page in pages])
    print(f"💬 calling {MODEL_NAME} for pages {[page['page_index'] for page in pages]}...")
    batch_result = await structured_model.ainvoke(prompt)

    raw_text_by_page = {page["page_index"]: page["markdown"].strip() for page in pages}
    return match_batch_results(raw_text_by_page, batch_result.pages)


async def _extract_one_batch(doc_id, document_type, batch_num, total_batches, pages):
    async with _llm_semaphore:
        print(f"🤖 [{doc_id}] extracting batch {batch_num}/{total_batches} ({len(pages)} pages)...")
        extraction_by_page = await extract_batch(document_type, pages)
    print(f"✅ [{doc_id}] batch {batch_num}/{total_batches} done")
    return extraction_by_page


async def extract_document(doc_id, document_type, pages):
    """Runs extract_batch over PAGES_PER_LLM_CALL-sized batches, at most LLM_CONCURRENCY in flight.
    pages: list of {"page_index": int, "markdown": str}. Returns {page: extraction}."""
    batches = list(batched(pages, PAGES_PER_LLM_CALL))
    print(f"🤖 [{doc_id}] running LLM extraction on {len(batches)} batches ({LLM_CONCURRENCY} concurrent)...")
    results = await asyncio.gather(*(
        _extract_one_batch(doc_id, document_type, batch_number, len(batches), batch)
        for batch_number, batch in enumerate(batches, start=1)
    ))

    all_extractions = {}
    for extraction_by_page in results:
        all_extractions.update(extraction_by_page)
    return all_extractions


if __name__ == "__main__":
    print("ok")
