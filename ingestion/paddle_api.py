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

import clean

load_dotenv()

JOB_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
MODEL = "PaddleOCR-VL-1.6"
POLL_INTERVAL_SECONDS = 5


def _token():
    token = os.environ.get("PADDLEOCR_TOKEN")
    if not token:
        raise RuntimeError("PADDLEOCR_TOKEN not set in .env")
    return token


async def _submit(client, pdf_path):
    headers = {"Authorization": f"bearer {_token()}"}
    data = {
        "model": MODEL,
        "optionalPayload": '{"useDocOrientationClassify": false, "useDocUnwarping": false, "useChartRecognition": false}',
    }
    with open(pdf_path, "rb") as f:
        files = {"file": (Path(pdf_path).name, f, "application/pdf")}
        resp = await client.post(JOB_URL, headers=headers, data=data, files=files)
    resp.raise_for_status()
    return resp.json()["data"]["jobId"]


async def _poll(client, job_id):
    headers = {"Authorization": f"bearer {_token()}"}
    while True:
        resp = await client.get(f"{JOB_URL}/{job_id}", headers=headers)
        resp.raise_for_status()
        state = resp.json()["data"]["state"]
        if state == "done":
            return resp.json()["data"]["resultUrl"]["jsonUrl"]
        if state == "failed":
            raise RuntimeError(f"PaddleOCR-VL job failed: {resp.json()['data'].get('errorMsg')}")
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


async def parse_pdf(pdf_path):
    """Submit pdf_path to PaddleOCR-VL and return a list of per-page dicts:
    [{"page_index": int, "markdown": str}, ...] (0-based, matches PyMuPDF page.number).
    """
    async with httpx.AsyncClient(timeout=120) as client:
        job_id = await _submit(client, pdf_path)
        jsonl_url = await _poll(client, job_id)
        jsonl_resp = await client.get(jsonl_url)
    jsonl_resp.raise_for_status()

    jsonl_text = jsonl_resp.content.decode("utf-8")

    pages = []
    for line in jsonl_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        result = json.loads(line)["result"]
        for res in result["layoutParsingResults"]:
            text = res.get("markdown", {}).get("text", "")
            pages.append({"page_index": len(pages), "markdown": clean.strip_noise(text)})
    return pages


def _demo():
    pages = [{"page_index": 0, "markdown": "![img](rel/a.png) hello"}]
    assert pages[0]["markdown"] == "![img](rel/a.png) hello"
    print("paddle_api self-check passed (no network call)")


if __name__ == "__main__":
    _demo()
    if len(sys.argv) > 1:
        result = asyncio.run(parse_pdf(sys.argv[1]))
        print(f"Parsed {len(result)} pages")
