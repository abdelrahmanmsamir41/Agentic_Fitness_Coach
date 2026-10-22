"""Prompt templates for the normalize+extract Gemini calls -- one call per batch of pages (see build_training_batch_prompt). No markdown-heading input:
the model infers each page's own topic/label directly from its content
instead of relying on the raw OCR heading, which is often missing or
ambiguous.
"""

TRAINING_BATCH_PROMPT = """You are extracting structured data from raw OCR output from pages of
a fitness coach's training plan. Below are several pages, each marked with "=== PAGE N ===".
Process every page independently and return exactly one result per page, with page_index set to
that page's marker number.

{pages_block}

For each page:
1. Infer a short page_topic label from the page's own content (e.g. "LEGS",
   "FAQS", "Chest & Back", "Cardio") -- the page may have no heading of its
   own in the raw text; use your judgement from what's actually on the page.
2. Classify this page's dominant/primary content into exactly one category (see the list
   below), but populate EVERY list that actually has matching content on the page, not just
   the one matching category -- a page can genuinely contain more than one kind of content
   (e.g. mostly advice text that also names one specific exercise: category "advice", but
   still list that exercise in weight_exercises too).
   - "weight_exercise": a table/list of gym exercises with sets/reps/rest.
   - "cardio": cardio activities (treadmill, running, cycling, ...) with duration/speed/incline.
   - "glossary": defines fitness terms/abbreviations (e.g. RIR, W.U Sets, Supersets, Partials).
   - "advice": general coaching advice, rules, mistakes to avoid, or motivation -- not tied
     to a specific exercise or cardio session.
   - "program_structure": a table mapping MULTIPLE training days to muscle groups/splits (the
     plan's overall weekly schedule) -- not a page about just one single day.
   - "rest_day": a standalone page marking exactly ONE specific day as a rest day, with no
     table of other days -- e.g. a lone "REST DAY"/"Rest Day" divider page. This is different
     from "program_structure": that's the whole week's table, this is a single day's marker.
   - "marketing": promo codes, discounts, "order now" calls to action -- not coaching content.
   - "other": anything that doesn't fit the above (e.g. a bare cover title with no body).
3. If the plan marks days with inline headings ("DAY 2 – Push + Abs") instead of a split table,
   add one program_days entry per heading on its page, and set each exercise/cardio entry's
   day_number to the nearest DAY heading above it -- the one exception to processing pages
   independently: a page that opens mid-day continues the previous page's day.

OCR quirks to account for:
- Headings are sometimes marked with "#"/"##" and sometimes not, depending on OCR accuracy --
  judge a page's topic and structure from its actual content, not from markdown syntax alone.
- Text under a heading isn't always in its original reading order, and OCR sometimes drops a
  line under the wrong heading entirely -- if a row/line clearly belongs to a different
  exercise/day than the heading it's grouped under, attribute the data (sets, reps, rest, ...)
  to the exercise/day it actually describes, not the one it's positioned near.
- A table's video-link column often comes through blank -- the actual URLs are listed separately
  in a "_Links:" block at the end of the page, top-to-bottom in the same order as the table rows
  above them. Match each link to the row/entity it belongs to by that order and by its anchor
  text (if a link doesn't clearly belong to any single row, attach it to the closest one rather
  than dropping it), and put it in that row's video_url field.

Rules: NEVER change any number, unit, set/rep count, RIR/RPE value, or URL from the raw
text. Preserve exercise/food names, coach-specific terminology, and Arabic text as-is
(do not translate).

Example -- input page:
=== PAGE 2 ===
LEGS
TABLE[type=data]
columns: workout_name, warm_up_sets, working_sets, reps, reps_in_reserve, rest, workout_video

Squat | 1 | 4 | 8-12 | 2 | 90s |
Leg Press | | 3 | 10 | 1-2 | 90s |

_Links:
- [video](https://www.youtube.com/watch?v=aaa111)
- [video](https://www.youtube.com/watch?v=bbb222)

Expected result for that page:
{{
  "page_index": 2,
  "page_topic": "LEGS",
  "category": "weight_exercise",
  "weight_exercises": [
    {{"exercise": "Squat", "working_sets": {{"low": 4}}, "reps": {{"low": 8, "high": 12}}, "warmup_sets": {{"low": 1}}, "rir": {{"low": 2}}, "rest": "90s", "video_url": "https://www.youtube.com/watch?v=aaa111"}},
    {{"exercise": "Leg Press", "working_sets": {{"low": 3}}, "reps": {{"low": 10}}, "warmup_sets": {{"low": 0}}, "rir": {{"low": 1, "high": 2}}, "rest": "90s", "video_url": "https://www.youtube.com/watch?v=bbb222"}}
  ]
}}
Note: a blank cell where a number is expected means {{"low": 0}}, not null (see Leg Press's
warmup_sets above) -- a genuinely non-numeric value like "AMRAP" or "to failure" becomes null
instead. A single number becomes {{"low": N}} with no "high" (Squat's rir: "2"); a range becomes
{{"low", "high"}} (Leg Press's rir: "1-2", and both exercises' reps).
"""


def build_training_batch_prompt(tagged_pages):
    """tagged_pages: iterable of (page_index, raw_text)."""
    pages_block = "\n\n".join(f"=== PAGE {page_index} ===\n{text}" for page_index, text in tagged_pages)
    return TRAINING_BATCH_PROMPT.format(pages_block=pages_block)


DAY_ASSIGNMENT_PROMPT = """A training plan's split table lists these training days:
{day_list}

Below are other pages from the same plan, listed by page_index in the plan's actual document
order (a lower page_index always comes earlier in the PDF), each with its own inferred topic.
For each page, decide which listed day it belongs to, using your judgement -- the page's topic
may be phrased, cased, abbreviated, or written in a different language than the day list (e.g.
"Chest & Back" vs "يوم الصدر والظهر" both matching a "Chest & Back" day). Only return a
day_number that appears in the list above; if a page clearly doesn't belong to any listed day
(e.g. a general FAQ or cardio-protocol page not tied to one specific day), return null for it
instead of guessing.

Two or more days can share the exact same description (e.g. several days all just listed as
"Rest" or "REST DAY") -- when that happens, a page with that same generic topic must still be
resolved to one specific one of those days, not left null just because the wording is
ambiguous. Use document order to disambiguate: pages that share a specific, non-repeated day's
topic (e.g. two pages both about "Legs" back-to-back) can legitimately both map to that one
day, but distinct pages carrying a repeated generic topic (like several separate "Rest Day"
pages) each correspond to a different occurrence of that repeated day -- match them to the
repeated days in the same relative order they appear (the first such page in reading order
gets the earliest of those day_numbers, and so on).

Pages:
{pages_block}
"""


def build_day_assignment_prompt(day_entries, page_topics):
    """day_entries: iterable of (day_number, target). page_topics: iterable of (page_index, page_topic)."""
    day_list = "\n".join(f"- Day {day_number}: {target}" for day_number, target in day_entries)
    pages_block = "\n".join(f"- page {page_index}: \"{page_topic}\"" for page_index, page_topic in page_topics)
    return DAY_ASSIGNMENT_PROMPT.format(day_list=day_list, pages_block=pages_block)
