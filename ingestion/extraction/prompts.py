"""Prompt templates for the normalize+extract Gemini calls -- one call per
batch of pages (see build_training_batch_prompt). No markdown-heading input:
the model infers each page's own topic/label directly from its content
instead of relying on the raw OCR heading, which is often missing or
ambiguous.
"""

TRAINING_BATCH_PROMPT = """You are cleaning up raw OCR output from pages of a fitness coach's
training plan. Below are several pages, each marked with "=== PAGE N ===". Process every
page independently and return exactly one result per page, with page_index set to that
page's marker number.

{pages_block}

For each page:
1. Reconstruct fragmented OCR lines into clean, readable Markdown prose. If the raw text has a
   "TABLE[type=data]" block, render it in cleaned_text as a full markdown table: a header row
   built from the "columns:" line, a "| :--- | :--- | ..." alignment row under it, then one row
   per line with cells separated by "|" -- e.g.:
   | workout_name | warm_up_sets | working_sets | reps | rest | workout_video |
   | :--- | :--- | :--- | :--- | :--- | :--- |
   | Squat | 1 | 4 | 8-12 | 90s | [video](https://www.youtube.com/watch?v=aaa111) |
   Fill the video-link column with the matching "[video](url)" link -- see the _Links: rule below.
2. Infer a short page_topic label from the page's own content (e.g. "LEGS",
   "FAQS", "Chest & Back", "Cardio") -- the page may have no heading of its
   own in the raw text; use your judgement from what's actually on the page.
3. Classify this page into exactly one category, and extract entries into ONLY the list
   matching that category (leave every other list empty):
   - "weight_exercise": a table/list of gym exercises with sets/reps/rest.
   - "cardio": cardio activities (treadmill, running, cycling, ...) with duration/speed/incline.
   - "glossary": defines fitness terms/abbreviations (e.g. RIR, W.U Sets, Supersets, Partials).
   - "advice": general coaching advice, rules, mistakes to avoid, or motivation -- not tied
     to a specific exercise or cardio session.
   - "program_structure": maps training days to a muscle group/split, or marks a day as rest.
   - "marketing": promo codes, discounts, "order now" calls to action -- not coaching content.
   - "other": anything that doesn't fit the above (e.g. a bare cover title with no body).

OCR quirks to account for:
- Headings are sometimes marked with "#"/"##" and sometimes not, depending on OCR accuracy --
  judge a page's topic and structure from its actual content, not from markdown syntax alone.
- Text under a heading isn't always in its original reading order, and OCR sometimes drops a
  line under the wrong heading entirely -- if a line clearly belongs to a different exercise/day
  than the heading above it, group it with the one it actually describes.
- A table's video-link column often comes through blank -- the actual URLs are listed separately
  in a "_Links:" block at the end of the page, top-to-bottom in the same order as the table rows
  above them. Match each link to the row/entity it belongs to by that order and by its anchor
  text (if a link doesn't clearly belong to any single row, attach it to the closest one rather
  than dropping it), and put it in both that row's video_url field AND its cleaned_text table
  cell as "[video](url)" (see step 1). A link tied to specific words rather than row order (e.g.
  inline in a sentence) goes next to those words in cleaned_text instead.

Rules: NEVER change any number, unit, set/rep count, RIR/RPE value, or URL from the raw
text. Preserve exercise/food names, coach-specific terminology, and Arabic text as-is
(do not translate). Only fix OCR fragmentation, typos, and formatting.

Example -- input page:
=== PAGE 2 ===
LEGS
TABLE[type=data]
columns: workout_name, warm_up_sets, working_sets, reps, rest, workout_video

Squat | 1 | 4 | 8-12 | 90s |
Leg Press | | 3 | 10 | 90s |

_Links:
- [video](https://www.youtube.com/watch?v=aaa111)
- [video](https://www.youtube.com/watch?v=bbb222)

Expected result for that page:
{{
  "page_index": 2,
  "page_topic": "LEGS",
  "category": "weight_exercise",
  "cleaned_text": "**LEGS**\\n\\n| workout_name | warm_up_sets | working_sets | reps | rest | workout_video |\\n| :--- | :--- | :--- | :--- | :--- | :--- |\\n| Squat | 1 | 4 | 8-12 | 90s | [video](https://www.youtube.com/watch?v=aaa111) |\\n| Leg Press | | 3 | 10 | 90s | [video](https://www.youtube.com/watch?v=bbb222) |",
  "weight_exercises": [
    {{"exercise": "Squat", "working_sets": {{"low": 4}}, "reps": {{"low": 8, "high": 12}}, "warmup_sets": {{"low": 1}}, "rir": null, "rest": "90s", "video_url": "https://www.youtube.com/watch?v=aaa111"}},
    {{"exercise": "Leg Press", "working_sets": {{"low": 3}}, "reps": {{"low": 10}}, "warmup_sets": {{"low": 0}}, "rir": null, "rest": "90s", "video_url": "https://www.youtube.com/watch?v=bbb222"}}
  ]
}}
Note: cleaned_text is a full markdown table (header + alignment row) copied faithfully from the
raw table, including blank cells (Leg Press's warm_up_sets stays blank, not "0") -- with each
"_Links:" URL placed in its row's video_url AND its table cell by row order. In the structured
weight_exercises field, though, a blank cell where a number is expected means {{"low": 0}}, not
null (see Leg Press's warmup_sets below) -- that 0-vs-null interpretation only applies there, not
to cleaned_text. A single number becomes {{"low": N}} with no "high"; a range "8-12" becomes
{{"low": 8, "high": 12}}; a non-numeric value like "AMRAP" or "to failure" becomes null instead.
"""


def build_training_batch_prompt(tagged_pages):
    """tagged_pages: iterable of (page_index, raw_text)."""
    pages_block = "\n\n".join(f"=== PAGE {idx} ===\n{text}" for idx, text in tagged_pages)
    return TRAINING_BATCH_PROMPT.format(pages_block=pages_block)


DAY_ASSIGNMENT_PROMPT = """A training plan's split table lists these training days:
{day_list}

Below are other pages from the same plan, each with its own inferred topic. For each page,
decide which listed day it belongs to, using your judgement -- the page's topic may be phrased,
cased, abbreviated, or written in a different language than the day list (e.g. "Chest & Back"
vs "يوم الصدر والظهر" both matching a "Chest & Back" day). Only return a day_number that
appears in the list above; if a page clearly doesn't belong to any listed day (e.g. a general
FAQ or cardio-protocol page not tied to one specific day), return null for it instead of guessing.

Pages:
{pages_block}
"""


def build_day_assignment_prompt(day_entries, page_topics):
    """day_entries: iterable of (day_number, target). page_topics: iterable of (page_index, page_topic)."""
    day_list = "\n".join(f"- Day {day_number}: {target}" for day_number, target in day_entries)
    pages_block = "\n".join(f"- page {page_index}: \"{page_topic}\"" for page_index, page_topic in page_topics)
    return DAY_ASSIGNMENT_PROMPT.format(day_list=day_list, pages_block=pages_block)
