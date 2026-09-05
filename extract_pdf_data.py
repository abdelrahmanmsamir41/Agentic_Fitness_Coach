"""Extract text, tables, and hyperlinks from a PDF into a Markdown file, page by page."""
import re
import sys
import fitz  # PyMuPDF

from recombine_split_pdf import content_pairs, merged_lines, recombine

PDF_PATH = r"C:\Users\apder\Music\938632224-BelghamdiSplit-Upper-Lower-4-Days.pdf"
OUT_PATH = r"C:\Users\apder\Choose-Your-Trainer\BelghamdiSplit-Upper-Lower-4-Days.md"

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
    orig_doc, pairs = None, content_pairs(doc)
    if pairs is not None:
        orig_doc = doc  # keep the pre-merge pages around for line-accurate text
        fixed_path = recombine(pdf_path)
        print(f"Detected a left/right-split PDF; recombined to {fixed_path}")
        pdf_path = fixed_path
        doc = fitz.open(pdf_path)
    lines = [f"# {doc.metadata.get('title') or 'Extracted PDF Data'}\n"]
    lines.append(f"Source: `{pdf_path}`  \nPages: {doc.page_count}\n")

    for page in doc:
        pno = page.number + 1
        lines.append(f"\n---\n\n## Page {pno}\n")

        page_lines = None
        if orig_doc is not None:
            left_pno, right_pno = pairs[page.number]
            page_lines = merged_lines(orig_doc[left_pno], orig_doc[right_pno])
            text = "\n".join(t for _, t in page_lines).strip()
        else:
            text = page.get_text("text").strip()
        if text:
            lines.append("### Text\n")
            lines.append("```\n" + text + "\n```\n")
        else:
            lines.append("_No text on this page._\n")

        tables = page.find_tables()
        for i, table in enumerate(tables.tables, start=1):
            rows = table.extract()
            if not rows:
                continue
            lines.append(f"\n### Table {i} (page {pno})\n")
            header, *body = rows
            header = [fix_rtl(str(c)) if c is not None else "" for c in header]
            lines.append("| " + " | ".join(header) + " |")
            lines.append("|" + "|".join(["---"] * len(header)) + "|")
            for row in body:
                row = [fix_rtl(str(c)) if c is not None else "" for c in row]
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

        links = [l for l in page.get_links() if l.get("uri")]
        if links:
            lines.append(f"\n### Links (page {pno})\n")
            for l in links:
                rect = l["from"]
                anchor_text = " ".join(page.get_textbox(rect).split()) or "(no visible text)"
                lines.append(f"- [{anchor_text}]({l['uri']})")
            lines.append("")

    doc.close()
    if orig_doc is not None:
        orig_doc.close()

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
    pdf = r"Splits\hossam_mansour_splits\Agonist antagonist advanced(2).pdf"
    out = r"splits_extracted\Agonist antagonist advanced(2).md"
    extract(pdf, out)
