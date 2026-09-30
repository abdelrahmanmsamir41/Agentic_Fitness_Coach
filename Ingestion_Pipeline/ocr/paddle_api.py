"""PaddleOCR-VL cloud job API: submit a PDF, poll, and return per-page markdown.
Image references stay as the API's remote URLs; only text is embedded.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

from console import print_step

from . import preprocess

load_dotenv()

JOB_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
OCR_MODEL = "PaddleOCR-VL-1.6"
POLL_INTERVAL_SECONDS = 5
POLL_TIMEOUT_SECONDS = 1800  # a job stuck "running" this long is dead, not slow


def _get_token():
    token = os.environ.get("PADDLEOCR_TOKEN")
    if not token:
        raise RuntimeError("PADDLEOCR_TOKEN not set in .env")
    return token


async def _request(client, method, *args, **kwargs):
    try:
        response = await client.request(method, *args, **kwargs)
        response.raise_for_status()
        return response
    except httpx.ConnectError as error:
        raise RuntimeError(f"PaddleOCR-VL connection error -- check internet or API URL: {error}") from error
    except httpx.TimeoutException as error:
        raise RuntimeError(f"PaddleOCR-VL request timed out: {error}") from error
    except httpx.HTTPStatusError as error:
        raise RuntimeError(f"PaddleOCR-VL HTTP error {error.response.status_code}: {error.response.text}") from error
    except httpx.RequestError as error:
        raise RuntimeError(f"PaddleOCR-VL request failed: {error}") from error


async def _submit_job(client, pdf_path):
    headers = {"Authorization": f"bearer {_get_token()}"}
    form_data = {
        "model": OCR_MODEL,
        "optionalPayload": json.dumps({"useDocOrientationClassify": False, "useDocUnwarping": False, "useChartRecognition": False}),
    }
    with open(pdf_path, "rb") as pdf_file:
        print_step(f"🐌 Processing file: {pdf_path}")
        files = {"file": (Path(pdf_path).name, pdf_file, "application/pdf")}
        response = await _request(client, "POST", JOB_URL, headers=headers, data=form_data, files=files)
    job_id = response.json()["data"]["jobId"]
    print_step(f"✅ OCR request submitted, job id: {job_id}")
    return job_id


async def _poll_job(client, job_id):
    headers = {"Authorization": f"bearer {_get_token()}"}
    started_at = asyncio.get_event_loop().time()
    while True:
        response = await _request(client, "GET", f"{JOB_URL}/{job_id}", headers=headers)
        job = response.json()["data"]
        state = job["state"]
        elapsed = int(asyncio.get_event_loop().time() - started_at)
        if elapsed > POLL_TIMEOUT_SECONDS:
            raise RuntimeError(f"PaddleOCR-VL job {job_id} stuck in '{state}' after {elapsed}s, giving up")
        if state == "done":
            print_step(f"✅ job {job_id} done, extracted {job['extractProgress']['extractedPages']} pages ({elapsed}s)")
            return job["resultUrl"]["jsonUrl"]
        if state == "failed":
            raise RuntimeError(f"PaddleOCR-VL job failed: {job.get('errorMsg')}")
        if state == "running" and "extractProgress" in job:
            progress = job["extractProgress"]
            print(f"... running, page {progress['extractedPages']}/{progress['totalPages']} ({elapsed}s)")
        else:
            print(f"... {state} ({elapsed}s)")
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


async def parse_pdf(pdf_path):
    """Submit pdf_path to PaddleOCR-VL and return a list of per-page dicts:
    [{"page_index": int, "markdown": str}, ...] (0-based, matches PyMuPDF page.number).
    """
    async with httpx.AsyncClient(timeout=120) as client:
        job_id = await _submit_job(client, pdf_path)
        jsonl_url = await _poll_job(client, job_id)
        jsonl_response = await _request(client, "GET", jsonl_url)

    jsonl_text = jsonl_response.content.decode("utf-8")

    pages = []
    for line in jsonl_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        line_result = json.loads(line)["result"]
        for layout_result in line_result["layoutParsingResults"]:
            text = layout_result.get("markdown", {}).get("text", "")
            pages.append({"page_index": len(pages), "markdown": preprocess.strip_noise(text)})
            print_step(f"📦 parsed page {len(pages)}")
    return pages


if __name__ == "__main__":
    if len(sys.argv) > 1:
        parsed_pages = asyncio.run(parse_pdf(sys.argv[1]))
        print(f"Parsed {len(parsed_pages)} pages")
