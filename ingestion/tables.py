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

    for tr in table_tag.find_all("tr"):
        row = []
        col = 0

        def fill_carried():
            nonlocal col
            while col in rowspan_carry:
                carry = rowspan_carry[col]
                row.append(carry[1])
                carry[0] -= 1
                if carry[0] <= 0:
                    del rowspan_carry[col]
                col += 1

        fill_carried()
        for cell in tr.find_all(["td", "th"], recursive=False):
            fill_carried()
            text = cell.get_text(" ", strip=True)
            colspan = int(cell.get("colspan", 1) or 1)
            rowspan = int(cell.get("rowspan", 1) or 1)
            for _ in range(colspan):
                row.append(text)
                if rowspan > 1:
                    rowspan_carry[col] = [rowspan - 1, text]
                col += 1
            fill_carried()
        grid.append(row)

    width = max((len(r) for r in grid), default=0)
    return [r + [""] * (width - len(r)) for r in grid]


def _slugify(header, normalize):
    text = normalize(header) if normalize else header
    text = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return text or "col"


def _dedupe(keys):
    seen = {}
    out = []
    for key in keys:
        if key not in seen:
            seen[key] = 0
            out.append(key)
        else:
            seen[key] += 1
            out.append(f"{key}_{seen[key]}")
    return out


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
    keys = _dedupe([_slugify(h, normalize) for h in header])
    lines = ["TABLE[type=data]", "columns: " + ", ".join(keys), ""]
    for row in rows:
        cells = [normalize(c) if normalize else c for c in row]
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def render_tables(text, normalize=None):
    """Replace every raw HTML <table> in `text` with a compact TABLE[...] block.

    `normalize` is an optional str->str callback applied to header/cell text
    (e.g. the same abbreviation-expansion used elsewhere in preprocessing) --
    kept as a parameter instead of importing it, so this module has no
    fitness-specific knowledge of its own.
    """
    def _replace(match):
        table_tag = BeautifulSoup(match.group(0), "lxml").find("table")
        if table_tag is None:
            return match.group(0)
        return _render_grid(_expand_grid(table_tag), normalize)

    return _TABLE_BLOCK.sub(_replace, text)


def _demo():
    simple = "<table><tr><td>Name</td><td>Reps</td><td>Rest</td></tr><tr><td>Squat</td><td>10</td><td>2 min</td></tr><tr><td>Row</td><td>12</td><td>90s</td></tr></table>"
    out = render_tables(simple)
    assert out == "TABLE[type=data]\ncolumns: name, reps, rest\n\nSquat | 10 | 2 min\nRow | 12 | 90s"

    mapping = "<table><tr><td>chest & back</td><td>DAY 1</td></tr><tr><td>LEGS</td><td>DAY 2</td></tr></table>"
    out = render_tables(mapping)
    assert out == "TABLE[type=mapping]\nchest & back: DAY 1\nLEGS: DAY 2"

    colspan = "<table><tr><td colspan=\"2\">Group</td><td>Reps</td></tr><tr><td>a</td><td>b</td><td>c</td></tr><tr><td>d</td><td>e</td><td>f</td></tr></table>"
    out = render_tables(colspan)
    assert out == "TABLE[type=data]\ncolumns: group, group_1, reps\n\na | b | c\nd | e | f"

    rowspan = (
        "<table><tr><td>Group</td><td>Exercise</td><td>Reps</td></tr>"
        "<tr><td rowspan=\"2\">Push</td><td>OHP</td><td>10</td></tr>"
        "<tr><td>Bench</td><td>8</td></tr>"
        "<tr><td>Pull</td><td>Row</td><td>12</td></tr></table>"
    )
    out = render_tables(rowspan)
    assert "Push | OHP | 10" in out
    assert "Push | Bench | 8" in out  # carried-down rowspan value
    assert "Pull | Row | 12" in out

    ragged = "<table><tr><td>A</td><td>B</td><td>C</td></tr><tr><td>1</td><td>2</td></tr></table>"
    out = render_tables(ragged)
    assert "1 | 2 | " in out  # padded to header width

    normalized = render_tables(simple, normalize=str.upper)
    assert "SQUAT | 10" in normalized

    around = f"before\n\n{simple}\n\nafter"
    assert render_tables(around).startswith("before\n\nTABLE[type=data]")
    assert render_tables(around).endswith("after")

    print("tables self-check passed")


if __name__ == "__main__":
    _demo()
