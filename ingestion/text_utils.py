"""Small shared regex helpers -- previously each hand-written independently
in more than one module (a duplicated NUMBER pattern in extraction/extract.py
and ocr/native_text.py; a duplicated word-boundary-alternation builder in
ocr/preprocess.py, reimplemented per-keyword instead of precompiled in
exercise_reference/rules.py).
"""
import re

# digit run, allowing "1,234"/"3.5" as one token without swallowing trailing punctuation
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def compile_word_boundary_pattern(keywords):
    """One compiled regex matching any of `keywords` as a whole word,
    case-insensitively -- longest keywords first, so e.g. "smith machine"
    is tried before the shorter "smith" would otherwise win instead."""
    alternation = "|".join(re.escape(k) for k in sorted(keywords, key=len, reverse=True))
    return re.compile(rf"\b({alternation})\b", re.IGNORECASE)


def _demo():
    assert NUMBER.findall("3x8-12, 90s") == ["3", "8", "12", "90"]

    pattern = compile_word_boundary_pattern(["smith", "smith machine"])
    assert pattern.search("Smith Machine Squat").group(0).lower() == "smith machine"
    assert pattern.search("Cable crossover") is None, "must not match unrelated substrings"
    assert pattern.search("flat bench") is None, "\\b must not match 'lat' inside 'flat'"


if __name__ == "__main__":
    _demo()
    print("ok")
