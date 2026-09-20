"""PaddleOCR-VL cloud job API: submit a PDF, poll, and return per-page markdown.

Refactored from the working ad-hoc script in test-paddle-ocr.ipynb (kept only
for experimentation now). Image references in the returned markdown point at
the API's remote image URLs directly -- we don't download the bytes for MVP,
text is what gets embedded/queried.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

from . import preprocess

load_dotenv()

JOB_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
MODEL = "PaddleOCR-VL-1.6"
POLL_INTERVAL_SECONDS = 5


def _token():
    token = os.environ.get("PADDLEOCR_TOKEN")
    if not token:
        raise RuntimeError("PADDLEOCR_TOKEN not set in .env")
    return token


async def _request(client, method, *args, **kwargs):
    try:
        resp = await client.request(method, *args, **kwargs)
        resp.raise_for_status()
        return resp
    except httpx.ConnectError as e:
        raise RuntimeError(f"PaddleOCR-VL connection error -- check internet or API URL: {e}") from e
    except httpx.TimeoutException as e:
        raise RuntimeError(f"PaddleOCR-VL request timed out: {e}") from e
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"PaddleOCR-VL HTTP error {e.response.status_code}: {e.response.text}") from e
    except httpx.RequestError as e:
        raise RuntimeError(f"PaddleOCR-VL request failed: {e}") from e


async def _submit(client, pdf_path):
    headers = {"Authorization": f"bearer {_token()}"}
    data = {
        "model": MODEL,
        "optionalPayload": json.dumps({"useDocOrientationClassify": False, "useDocUnwarping": False, "useChartRecognition": False}),
    }
    with open(pdf_path, "rb") as f:
        print(f"🐌 Processing file: {pdf_path}")
        files = {"file": (Path(pdf_path).name, f, "application/pdf")}
        resp = await _request(client, "POST", JOB_URL, headers=headers, data=data, files=files)
    job_id = resp.json()["data"]["jobId"]
    print(f"✅ OCR request submitted, job id: {job_id}")
    return job_id


async def _poll(client, job_id):
    headers = {"Authorization": f"bearer {_token()}"}
    start = asyncio.get_event_loop().time()
    while True:
        resp = await _request(client, "GET", f"{JOB_URL}/{job_id}", headers=headers)
        data = resp.json()["data"]
        state = data["state"]
        elapsed = int(asyncio.get_event_loop().time() - start)
        if state == "done":
            print(f"✅ job {job_id} done, extracted {data['extractProgress']['extractedPages']} pages ({elapsed}s)")
            return data["resultUrl"]["jsonUrl"]
        if state == "failed":
            raise RuntimeError(f"PaddleOCR-VL job failed: {data.get('errorMsg')}")
        if state == "running" and "extractProgress" in data:
            print(f"... running, page {data['extractProgress']['extractedPages']}/{data['extractProgress']['totalPages']} ({elapsed}s)")
        else:
            print(f"... {state} ({elapsed}s)")
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


async def parse_pdf(pdf_path):
    """Submit pdf_path to PaddleOCR-VL and return a list of per-page dicts:
    [{"page_index": int, "markdown": str}, ...] (0-based, matches PyMuPDF page.number).
    """
    async with httpx.AsyncClient(timeout=120) as client:
        job_id = await _submit(client, pdf_path)
        jsonl_url = await _poll(client, job_id)
        jsonl_resp = await _request(client, "GET", jsonl_url)

    jsonl_text = jsonl_resp.content.decode("utf-8")

    pages = []
    for line in jsonl_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        result = json.loads(line)["result"]
        for res in result["layoutParsingResults"]:
            text = res.get("markdown", {}).get("text", "")
            pages.append({"page_index": len(pages), "markdown": preprocess.strip_noise(text)})
            print(f"📦 parsed page {len(pages)}")
    return pages


if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = asyncio.run(parse_pdf(sys.argv[1]))
        print(f"Parsed {len(result)} pages")
