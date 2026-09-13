"""Recall@5 eval: for each question in eval_questions.json, check whether a
chunk matching `expected_contains` (a substring checked against the chunk's
embed_text) shows up in the top-5 Qdrant hits.

Populate eval_questions.json AFTER the first `python ingestion/ingest.py`
run -- expected content depends on what's actually in the real PDFs.
"""
import asyncio
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ingestion"))
import store_qdrant  # noqa: E402

EVAL_FILE = Path(__file__).resolve().parent / "eval_questions.json"


async def run():
    questions = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    if not questions:
        print("eval_questions.json is empty -- populate it after running ingestion once.")
        return

    hits = 0
    for q in questions:
        results = await store_qdrant.search(q["question"], top_k=5)
        found = any(q["expected_contains"].lower() in r.payload.get("embed_text", "").lower() for r in results)
        hits += found
        print(f"{'OK  ' if found else 'MISS'} {q['question']!r}")

    print(f"\nRecall@5: {hits}/{len(questions)} = {hits / len(questions):.2f}")


if __name__ == "__main__":
    asyncio.run(run())
