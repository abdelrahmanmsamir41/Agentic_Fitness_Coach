"""Strip low-value noise from parsed markdown before it reaches the LLM, to cut token usage.
"""
import html
import re

from text_utils import compile_word_boundary_pattern

from . import tables

_IMAGE = re.compile(r"<div[^>]*>\s*<img[^>]*>\s*</div>|<img[^>]*>|!\[[^\]]*\]\([^)]*\)", re.DOTALL)
_STYLE_BORDER_ATTR = re.compile(r"\s*(style|border)\s*=\s*(\"[^\"]*\"|'[^']*'|\S+)", re.DOTALL)
_WHITESPACE = re.compile(r"[ \t ]+")
_BLANK_LINES = re.compile(r"\n{3,}")
# Fitness abbreviations, split into layers rather than one dict, since each
# layer has different ambiguity risk. Only terms confirmed low-collision for
# workout-plan text are here; terms like SS/BP/DL/PR/TM/RP/ISO/MV/AF collide
# with common words or other fitness programs/brands and are deliberately
# left unexpanded.
_EXERCISE_TERMS = {
    "DB": "Dumbbell",
    "KB": "Kettlebell",
    "OHP": "Overhead Press",
    "RDL": "Romanian Deadlift",
    "BSS": "Bulgarian Split Squat",
    "RFESS": "Rear-Foot-Elevated Split Squat",
    "GHR": "Glute-Ham Raise",
    "GHD": "Glute-Ham Developer",
}
_PRESCRIPTION_TERMS = {
    "RIR": "Reps in Reserve",
    "RPE": "Rate of Perceived Exertion",
    "TUT": "Time Under Tension",
    "1RM": "One-Rep Max",
}
_METHOD_TERMS = {
    "AMRAP": "As Many Reps As Possible",
    "EMOM": "Every Minute on the Minute",
}
_PROGRAM_TERMS = {
    "PPLUL": "Push/Pull/Legs + Upper/Lower",
    "PPL": "Push/Pull/Legs",
    "MEV": "Minimum Effective Volume",
    "MRV": "Maximum Recoverable Volume",
    "MAV": "Maximum Adaptive Volume",
    "SBD": "Squat/Bench/Deadlift",
}
# "WU"/"W.U" alone is too short to safely blind-expand, but paired with
# "set(s)" it's unambiguous -- the only set-type term worth an automated rule.
_WU_SET = re.compile(r"\bW\.?U\.?\b(?=\s+sets?\b)", re.IGNORECASE)


def _build_expander(mapping):
    pattern = compile_word_boundary_pattern(mapping)

    def _expand(match):
        expansion = mapping[match.group(0).upper()]
        # digit-led tokens like "1RM" have no case of their own -- default to
        # the capitalized form, only lowercase when the match itself is lowercase.
        return expansion.lower() if match.group(0)[0].islower() else expansion

    return pattern, _expand


_EXERCISE_RE, _expand_exercise = _build_expander(_EXERCISE_TERMS)
_PRESCRIPTION_RE, _expand_prescription = _build_expander(_PRESCRIPTION_TERMS)
_METHOD_RE, _expand_method = _build_expander(_METHOD_TERMS)
_PROGRAM_RE, _expand_program = _build_expander(_PROGRAM_TERMS)


def _expand_terms(text):
    text = _EXERCISE_RE.sub(_expand_exercise, text)
    text = _PRESCRIPTION_RE.sub(_expand_prescription, text)
    text = _METHOD_RE.sub(_expand_method, text)
    text = _PROGRAM_RE.sub(_expand_program, text)
    text = _WU_SET.sub(lambda match: "Warm-up" if match.group(0)[0].isupper() else "warm-up", text)
    return text


def strip_noise(text):
    # tables first, while their HTML is still intact
    text = tables.render_tables(text, normalize=_expand_terms)
    text = _IMAGE.sub("", text)
    text = _STYLE_BORDER_ATTR.sub("", text)
    text = _expand_terms(text)
    text = html.unescape(text)
    text = _WHITESPACE.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text


def _demo():
    styled_div = "<div style=\"margin:0\" border=1>text</div>"
    assert "style=" not in strip_noise(styled_div) and "border=" not in strip_noise(styled_div)

    raw_table = "<table><tr><td>Exercise</td><td>Reps</td><td>Rest</td></tr><tr><td>DB Squat</td><td>10</td><td>2 min</td></tr></table>"
    cleaned = strip_noise(raw_table)
    assert cleaned == "TABLE[type=data]\ncolumns: exercise, reps, rest\n\nDumbbell Squat | 10 | 2 min"

    md_img = "![photo](https://example.com/a.png) some text"
    assert strip_noise(md_img).strip() == "some text"

    assert strip_noise("DB shoulder press") == "Dumbbell shoulder press"
    assert strip_noise("3x10 db chest press") == "3x10 dumbbell chest press"
    assert strip_noise("DBMS is unrelated") == "DBMS is unrelated"

    assert strip_noise("chest &amp; back") == "chest & back"
    assert strip_noise("Day&nbsp;1&nbsp;&mdash;&nbsp;Legs") == "Day 1 — Legs"
    assert strip_noise("a\n\n\n\nb") == "a\n\nb"

    assert strip_noise("KB swings") == "Kettlebell swings"
    assert strip_noise("3 sets ohp") == "3 sets overhead press"
    assert strip_noise("RDL and BSS") == "Romanian Deadlift and Bulgarian Split Squat"
    assert strip_noise("RIR 2, RPE 8") == "Reps in Reserve 2, Rate of Perceived Exertion 8"
    assert strip_noise("test your 1RM") == "test your One-Rep Max"
    assert strip_noise("finish with AMROM") == "finish with AMROM"  # not a real term, must not partial-match AMRAP
    assert strip_noise("last set is AMRAP") == "last set is As Many Reps As Possible"
    assert strip_noise("Day 5 is PPLUL, others are PPL") == "Day 5 is Push/Pull/Legs + Upper/Lower, others are Push/Pull/Legs"
    assert strip_noise("stay near your MEV") == "stay near your Minimum Effective Volume"
    assert strip_noise("W.U Sets don't count") == "Warm-up Sets don't count"
    assert strip_noise("do 1 wu set first") == "do 1 warm-up set first"
    # deliberately NOT expanded -- see the layer comment above
    assert strip_noise("do a SS with BB rows") == "do a SS with BB rows"
    print("preprocess self-check passed")


if __name__ == "__main__":
    _demo()
    print("ok")
