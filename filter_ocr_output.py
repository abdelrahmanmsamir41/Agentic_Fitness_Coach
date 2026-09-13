"""Filter PaddleOCR's per-page JSON results down to high-confidence text only."""
import json
import re
import sys
from pathlib import Path

MIN_SCORE = 0.15


def page_index(path):
    m = re.search(r"_(\d+)_res\.json$", path.name)
    return int(m.group(1)) if m else 0


def filter_page(json_path, min_score):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    texts = data["rec_texts"]
    scores = data["rec_scores"]
    return [t for t, s in zip(texts, scores) if s >= min_score and t.strip()]


def main(json_dir, min_score=MIN_SCORE):
    json_dir = Path(json_dir)
    paths = sorted(json_dir.glob("*_res.json"), key=page_index)
    for path in paths:
        lines = filter_page(path, min_score)
        print(f"--- page {page_index(path)} ({path.name}) ---")
        for line in lines:
            print(line)
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\apder\Choose-Your-Trainer\Splits\shady_splits\output")
