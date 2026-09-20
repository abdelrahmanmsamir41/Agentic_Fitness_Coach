"""Normalize + extract: one Gemini call per page. No markdown-heading
dependency -- each page's topic/label is inferred by the model itself (see
models.TrainingSectionResult.page_topic).
"""
import os
import re

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from . import prompts
from .models import NutritionSectionResult, TrainingSectionResult, TRAINING_LIST_FIELDS

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
        _model = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.0)
    return _model


def _training_item(result):
    item = {"category": result.category, "page_topic": result.page_topic, "cleaned_text": result.cleaned_text}
    for field in TRAINING_LIST_FIELDS:
        item[field] = [e.model_dump() for e in getattr(result, field)]
    return item


def _numbers_and_urls(text):
    """Distinct tokens as sets, not multisets -- repeat counts can legitimately change."""
    return set(_NUMBER.findall(text)), set(_URL.findall(text))


def _build_result(raw_text, item):
    raw_numbers, raw_urls = _numbers_and_urls(raw_text)
    clean_numbers, clean_urls = _numbers_and_urls(item["cleaned_text"])
    needs_review = raw_numbers != clean_numbers or raw_urls != clean_urls

    return {
        **item,
        "cleaned_text": raw_text if needs_review else item["cleaned_text"],
        "needs_review": needs_review,
        "confidence": 1.0,
    }


async def normalize_and_extract(document_type, raw_text):
    if document_type == "training_plan":
        structured_model = _get_model().with_structured_output(TrainingSectionResult)
        result = await structured_model.ainvoke(prompts.TRAINING_PROMPT.format(raw_text=raw_text))
        item = _training_item(result)
        
    # else:
    #     structured_model = _get_model().with_structured_output(NutritionSectionResult)
    #     result = await structured_model.ainvoke(prompts.PROMPT.format(raw_text=raw_text))
    #     item = {"page_topic": result.page_topic, "cleaned_text": result.cleaned_text,
    #             "entities": [e.model_dump() for e in result.entities]}
    return _build_result(raw_text, item)
