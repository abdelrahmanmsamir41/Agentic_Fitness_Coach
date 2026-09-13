"""Strip low-value noise from parsed markdown before it reaches the LLM, to
cut token usage. Adapted from the image/table-attribute cleanup steps in
C:\\Users\\apder\\Data-Entry\\pipelines\\ocr\\Cleaning\\clean.py -- only the
parts relevant here (image tags, verbose HTML table styling); the
currency/duration normalization in that file doesn't apply to fitness plans.
"""
import re

_IMG = re.compile(r"<div[^>]*>\s*<img[^>]*>\s*</div>|<img[^>]*>|!\[[^\]]*\]\([^)]*\)", re.DOTALL)
_STYLE_BORDER_ATTR = re.compile(r"\s*(style|border)\s*=\s*(\"[^\"]*\"|'[^']*'|\S+)", re.DOTALL)
_WHITESPACE = re.compile(r"[ \t]+")


def strip_noise(text):
    text = _IMG.sub("", text)
    text = _STYLE_BORDER_ATTR.sub("", text)
    text = _WHITESPACE.sub(" ", text)
    return text


def _demo():
    html = "<table border=1 style=\"margin:0\"><tr><td>Day 1</td></tr></table>"
    
    cleaned = strip_noise(html)
    assert "style=" not in cleaned and "border=" not in cleaned
    assert "Day 1" in cleaned

    md_img = "![photo](https://example.com/a.png) some text"
    assert strip_noise(md_img).strip() == "some text"
    print("clean self-check passed")


if __name__ == "__main__":
    _demo()
