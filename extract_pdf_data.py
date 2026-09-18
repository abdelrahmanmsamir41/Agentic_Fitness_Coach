"""Extract text, tables, and hyperlinks from a PDF into a Markdown file, page by page."""
import re
import sys
import pymupdf as fitz

PDF_PATH = r"Splits\shady_splits\W-1401-260906-0202 - Abdelrhman Mohamed Samer.pdf"
OUT_PATH = r"C:\Users\apder\Choose-Your-Trainer\shadad split 2.0.md"

# PyMuPDF pulls Arabic glyph runs in visual (already-mirrored) order, so pure-Arabic
# spans come out character-reversed. Reversing each contiguous Arabic-script run
# (letters + internal spaces) undoes it; Latin/digits outside the run are untouched.
_ARABIC_RUN = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF][\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\s]*[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]|[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")


def fix_rtl(s):
    if not s or not _ARABIC_RUN.search(s):
        return s
    return _ARABIC_RUN.sub(lambda m: m.group(0)[::-1], s)


def extract(pdf_path, out_path):
    doc = fitz.open(pdf_path)
    lines = [f"# {doc.metadata.get('title') or 'Extracted PDF Data'}\n"]
    lines.append(f"Source: `{pdf_path}`  \nPages: {doc.page_count}\n")

    for page in doc:
        pno = page.number + 1
        links = [l for l in page.get_links() if l.get("uri")]
        if links:
            lines.append(f"\n### Links (page {pno})\n")
            for l in links:
                rect = l["from"]
                anchor_text = " ".join(page.get_textbox(rect).split()) or "(no visible text)"
                lines.append(f"- [{anchor_text}]({l['uri']})")
            lines.append("")

    doc.close()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote {out_path}")


def _demo():
    assert fix_rtl("نيرمتلل ت") == "ت للتمرين"
    assert fix_rtl("Smith High Incline Press") == "Smith High Incline Press"
    assert fix_rtl("1~2 نيرمت 4~6") == "1~2 تمرين 4~6"
    print("fix_rtl self-check passed")


if __name__ == "__main__":
    _demo()
    pdf = PDF_PATH
    out = OUT_PATH
    extract(pdf, out)
