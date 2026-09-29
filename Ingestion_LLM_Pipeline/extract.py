"""PDF -> Gemini extraction: the whole PDF goes to the model in one call (split only past
PDF_PAGES_PER_CALL pages), prompted with prompts.PDF_TRAINING_PROMPT plus the PDF's hyperlinks.
"""
import asyncio
import base64

import pymupdf as fitz
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

import links
from extraction.extract import MODEL_NAME, _match_batch_results
from extraction.models import TrainingBatchResult
from lazy import lazy
from prompts import PDF_TRAINING_PROMPT

PDF_PAGES_PER_CALL = 40  # a typical plan fits one call, as benchmarked; ~80 pages would hit the 65k output cap
THINKING_BUDGET = 1024  # benchmarked: same accuracy as default thinking at ~1/3 the cost and ~2.5x faster

get_model = lazy(lambda: ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.1, thinking_budget=THINKING_BUDGET))


def _pdf_block(pdf_path, pages):
    # ponytail: inline base64, request limit ~20MB; switch to the Gemini File API if huge PDFs come in
    with fitz.open(pdf_path) as src, fitz.open() as sub:
        sub.insert_pdf(src, from_page=pages[0]["page_index"], to_page=pages[-1]["page_index"])
        data = base64.b64encode(sub.tobytes()).decode()
    return {"type": "file", "base64": data, "mime_type": "application/pdf"}


def _prompt(pdf_path, pages):
    # prompt carries page numbering + links only; the page text layer stays out (Gemini reads the PDF itself)
    by_page = links.extract_links(pdf_path)
    blocks = [f"PAGE {p['page_index']}:\n" + "\n".join(
                  f"- {e['url']}" if e["anchor_text"] == "None" else f"- [{e['anchor_text']}] {e['url']}"
                  for e in by_page[p["page_index"]])
              for p in pages if p["page_index"] in by_page]
    first = pages[0]["page_index"]
    return PDF_TRAINING_PROMPT.format(first=first, second=first + 1, links_block="\n".join(blocks) or "(none)")


async def _extract_pdf_part(pdf_path, pages):
    structured_model = get_model().with_structured_output(TrainingBatchResult)
    prompt = _prompt(pdf_path, pages)
    print(f"💬 calling {MODEL_NAME} for pages {pages[0]['page_index']}-{pages[-1]['page_index']}...")
    result = await structured_model.ainvoke([HumanMessage(content=[_pdf_block(pdf_path, pages), {"type": "text", "text": prompt}])])
    return _match_batch_results({p["page_index"]: p["markdown"].strip() for p in pages}, result.pages)


async def extract_document(doc_id, document_type, pages, pdf_path):
    """pages: list of {"page_index": int, "markdown": str} (markdown becomes page_text). Returns {page: extraction}."""
    if document_type != "training_plan":
        raise NotImplementedError(f"PDF extraction not implemented for {document_type!r}")

    parts = [pages[i:i + PDF_PAGES_PER_CALL] for i in range(0, len(pages), PDF_PAGES_PER_CALL)]
    print(f"🤖 [{doc_id}] sending {len(pages)} PDF pages to {MODEL_NAME} in {len(parts)} call(s)...")
    all_extractions = {}
    for extraction_by_page in await asyncio.gather(*(_extract_pdf_part(pdf_path, part) for part in parts)):
        all_extractions.update(extraction_by_page)
    return all_extractions
