"""Undo a 'left-half pages, then right-half pages, plus blank padding' PDF split.

Some exported PDFs take N real pages, crop each into a left half and a right
half (same page size, content anchored to one side), then emit: N left
halves, N right halves, and some blank filler pages -- so a 5-page doc becomes
a 15-page file. Blank pages mark the boundary between independent chunks of
this pattern: each maximal run of consecutive non-blank pages is treated as
its own segment, and within a segment the first half must be left crops and
the second half the matching right crops. This matters because a real file
can contain more than one such run back to back (e.g. a filled-in example
copy of a program followed by a blank template copy of the same program,
divided by a single blank page) -- pooling every content page in the whole
document together would wrongly pair a page from one run with a page from
another. This rebuilds each run's original N pages by placing each left/right
pair side by side, unscaled, on one double-width page.
"""
import re
import sys
import fitz  # PyMuPDF

_ARABIC_CHAR = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")


def is_blank(page):
    # Images alone don't count: a shared background/logo image can appear on
    # every page, blank or not, so text presence is the reliable signal.
    return not page.get_text().strip()


def _segments(doc):
    """Maximal runs of consecutive non-blank pages; any blank page ends a run."""
    segments, current = [], []
    for p in doc:
        if is_blank(p):
            if current:
                segments.append(current)
                current = []
        else:
            current.append(p)
    if current:
        segments.append(current)
    return segments


def _segment_pairs(segment):
    """(left_pno, right_pno) pairs within one contiguous run, or None if it doesn't look split."""
    n = len(segment)
    if n < 2 or n % 2 != 0:
        return None
    widths = {round(p.rect.width) for p in segment}
    heights = {round(p.rect.height) for p in segment}
    if len(widths) != 1 or len(heights) != 1:
        return None  # left/right pages must share one page size
    half = n // 2
    return [(segment[i].number, segment[i + half].number) for i in range(half)]


def looks_split(doc):
    """Return the total pair count across every blank-delimited run that looks
    like an even set of same-size left/right crops, else None."""
    total = sum(len(_segment_pairs(seg) or []) for seg in _segments(doc))
    return total or None


def content_pairs(doc):
    """All (left_page_number, right_page_number) pairs across every splittable
    run, in document order. A run that doesn't look split is left alone (its
    pages contribute no pairs)."""
    pairs = []
    for seg in _segments(doc):
        pairs.extend(_segment_pairs(seg) or [])
    return pairs or None


def _lines(page):
    """(y0, text) per non-empty line, in the page's own (already-correct) reading order."""
    out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"])
            if text.strip():
                out.append((line["bbox"][1], text))
    return out


def merged_lines(left_page, right_page, y_tol=2.0):
    """(y, text) pairs for a left/right page pair, one merged line at a time.

    The crop boundary sometimes falls mid-word, so this can't be perfect --
    but matching lines by their (near-identical) y-position and joining each
    pair in the right direction (right-then-left for Arabic/RTL lines,
    left-then-right otherwise) turns two disconnected column dumps into one
    coherent, line-aligned page.
    """
    left_lines = sorted(_lines(left_page))
    right_lines = sorted(_lines(right_page))
    used_right = [False] * len(right_lines)
    merged = []

    for ly, lt in left_lines:
        match = next((i for i, (ry, _) in enumerate(right_lines)
                       if not used_right[i] and abs(ry - ly) <= y_tol), None)
        if match is None:
            merged.append((ly, lt))
            continue
        used_right[match] = True
        ry, rt = right_lines[match]
        merged.append((min(ly, ry), rt + lt if _ARABIC_CHAR.search(rt + lt) else lt + rt))

    for i, (ry, rt) in enumerate(right_lines):
        if not used_right[i]:
            merged.append((ry, rt))

    merged.sort(key=lambda pair: pair[0])
    return merged


def merge_text(left_page, right_page, y_tol=2.0):
    """Stitch a left/right page pair's text into one reading-order block."""
    return "\n".join(text for _, text in merged_lines(left_page, right_page, y_tol))


def recombine(pdf_path, out_path=None):
    """If pdf_path looks split, write the rebuilt PDF to out_path and return out_path. Else return None."""
    doc = fitz.open(pdf_path)
    pairs = content_pairs(doc)
    if pairs is None:
        doc.close()
        return None

    w, h = doc[pairs[0][0]].rect.width, doc[pairs[0][0]].rect.height
    out = fitz.open()
    for left_pno, right_pno in pairs:
        page = out.new_page(width=2 * w, height=h)
        page.show_pdf_page(fitz.Rect(0, 0, w, h), doc, left_pno)
        page.show_pdf_page(fitz.Rect(w, 0, 2 * w, h), doc, right_pno)

        # Left and right halves were cropped from the same original page, so
        # each carries its own full copy of every link annotation (same uri,
        # rect merely shifted) -- keep one rect per uri instead of doubling up.
        by_uri = {}
        for src_pno, x_off in ((left_pno, 0), (right_pno, w)):
            for link in doc[src_pno].get_links():
                if link.get("uri"):
                    by_uri[link["uri"]] = link["from"] + (x_off, 0, x_off, 0)
        for uri, rect in by_uri.items():
            page.insert_link({"kind": fitz.LINK_URI, "from": rect, "uri": uri})

    out_path = out_path or pdf_path.rsplit(".", 1)[0] + "-recombined.pdf"
    out.save(out_path)
    out.close()
    doc.close()
    return out_path


def _demo():
    doc = fitz.open()
    for _ in range(2):
        p = doc.new_page(width=100, height=50)
        p.insert_text((10, 10), "x")
    for _ in range(2):
        p = doc.new_page(width=100, height=50)
        p.insert_text((10, 10), "y")
    doc.new_page(width=100, height=50)  # trailing blank
    assert looks_split(doc) == 2
    doc.close()

    # Two independent split segments back to back, each divided from its
    # neighbor by a blank page (mirrors a filled example + blank template of
    # the same program, exported one after the other):
    # [blank, L1, R1, blank, L2, R2, blank]
    doc4 = fitz.open()
    for _ in range(7):
        doc4.new_page(width=100, height=50)
    doc4[1].insert_text((10, 10), "L1")
    doc4[2].insert_text((10, 10), "R1")
    doc4[4].insert_text((10, 10), "L2")
    doc4[5].insert_text((10, 10), "R2")
    assert looks_split(doc4) == 2
    assert content_pairs(doc4) == [(1, 2), (4, 5)]  # not (1,4)/(2,5) -- segments don't mix
    doc4.close()

    doc2 = fitz.open()
    doc2.new_page(width=100, height=50)
    doc2.new_page(width=100, height=50)
    assert looks_split(doc2) is None  # all blank, no content
    doc2.close()

    doc3 = fitz.open()
    doc3.new_page(width=100, height=50)
    doc3.new_page(width=100, height=50)
    doc3[0].insert_text((10, 10), "hello")
    doc3[1].insert_text((10, 10), "world")
    assert merge_text(doc3[0], doc3[1]) == "helloworld"  # LTR: left-then-right
    doc3.close()
    print("recombine self-check passed")


if __name__ == "__main__":
    _demo()
    src = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\apder\Choose-Your-Trainer\belghamdi_splits\BelghamdiSplit-Upper-Lower-4-Days.pdf"
    result = recombine(src)
    print(result or f"{src} does not look split; nothing written")
