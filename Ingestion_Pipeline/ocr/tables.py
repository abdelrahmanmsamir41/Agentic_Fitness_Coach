"""Parse raw HTML <table> markup (as PaddleOCR-VL emits it) into a compact,
LLM-friendly text block instead of passing hundreds of <table>/<tr>/<td> tags
straight through. Uses BeautifulSoup (already a transitive dependency) since
a hand-rolled parser would need to deal with colspan/rowspan and ragged rows
anyway -- no reason to redo what a real HTML parser already handles.
"""
from __future__ import annotations

import re

from bs4 import BeautifulSoup

_TABLE_BLOCK = re.compile(r"<table\b.*?</table>", re.DOTALL | re.IGNORECASE)


def _expand_grid(table_tag):
    """<tr>/<td> rows -> a rectangular grid of strings, expanding colspan/rowspan
    by repeating the spanned cell's value into every cell it covers."""
    grid = []
    rowspan_carry = {}  # col_index -> [rows_remaining, value]

    for row_tag in table_tag.find_all("tr"):
        row = []
        col_index = 0

        def fill_carried():
            nonlocal col_index
            while col_index in rowspan_carry:
                carry = rowspan_carry[col_index]
                row.append(carry[1])
                carry[0] -= 1
                if carry[0] <= 0:
                    del rowspan_carry[col_index]
                col_index += 1

        fill_carried()
        for cell in row_tag.find_all(["td", "th"], recursive=False):
            fill_carried()
            text = cell.get_text(" ", strip=True)
            colspan = int(cell.get("colspan", 1) or 1)
            rowspan = int(cell.get("rowspan", 1) or 1)
            for _ in range(colspan):
                row.append(text)
                if rowspan > 1:
                    rowspan_carry[col_index] = [rowspan - 1, text]
                col_index += 1
            fill_carried()
        grid.append(row)

    width = max((len(row) for row in grid), default=0)
    return [row + [""] * (width - len(row)) for row in grid]


def _slugify_header(header, normalize):
    text = normalize(header) if normalize else header
    text = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return text or "col"


def _dedupe_keys(keys):
    seen_counts = {}
    unique_keys = []
    for key in keys:
        if key not in seen_counts:
            seen_counts[key] = 0
            unique_keys.append(key)
        else:
            seen_counts[key] += 1
            unique_keys.append(f"{key}_{seen_counts[key]}")
    return unique_keys


def _render_grid(grid, normalize):
    if not grid or not grid[0]:
        return ""
    width = len(grid[0])

    if width == 2:
        lines = ["TABLE[type=mapping]"]
        for label, value in grid:
            label = normalize(label) if normalize else label
            lines.append(f"{label}: {value}")
        return "\n".join(lines)

    header, *rows = grid
    keys = _dedupe_keys([_slugify_header(column, normalize) for column in header])
    lines = ["TABLE[type=data]", "columns: " + ", ".join(keys), ""]
    for row in rows:
        cells = [normalize(cell) if normalize else cell for cell in row]
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def render_tables(text, normalize=None):
    """Replace every raw HTML <table> in `text` with a compact TABLE[...] block.

    `normalize` is an optional str->str callback applied to header/cell text
    (e.g. the same abbreviation-expansion used elsewhere in preprocessing) --
    kept as a parameter instead of importing it, so this module has no
    fitness-specific knowledge of its own.
    """
    def _render_table_match(match):
        table_tag = BeautifulSoup(match.group(0), "lxml").find("table")
        if table_tag is None:
            return match.group(0)
        return _render_grid(_expand_grid(table_tag), normalize)

    return _TABLE_BLOCK.sub(_render_table_match, text)
