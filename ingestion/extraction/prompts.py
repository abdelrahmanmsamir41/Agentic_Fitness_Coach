"""Prompt templates for the normalize+extract Gemini calls -- one call per
page. No markdown-heading input: the model infers each page's own
topic/label directly from its content instead of relying on the raw OCR
heading, which is often missing or ambiguous.
"""

TRAINING_CATEGORIES = """Classify this page into exactly one category, and extract
entries into ONLY the list matching that category (leave every other list empty):
- "weight_exercise": a table/list of gym exercises with sets/reps/rest.
- "cardio": cardio activities (treadmill, running, cycling, ...) with duration/speed/incline.
- "glossary": defines fitness terms/abbreviations (e.g. RIR, W.U Sets, Supersets, Partials).
- "advice": general coaching advice, rules, mistakes to avoid, or motivation -- not tied
  to a specific exercise or cardio session.
- "program_structure": maps training days to a muscle group/split, or marks a day as rest.
- "marketing": promo codes, discounts, "order now" calls to action -- not coaching content.
- "other": anything that doesn't fit the above (e.g. a bare cover title with no body)."""

PROMPT = """You are cleaning up raw OCR output from a fitness coach's meal plan.

Raw page text:
---
{raw_text}
---

1. Reconstruct fragmented OCR lines into clean, readable Markdown prose.
2. Infer a short page_topic label from the page's own content (e.g.
   "Breakfast", "Post-Workout Shake") -- the page may have no heading of its
   own in the raw text; use your judgement from what's actually on the page.
3. Extract each meal entry as a structured entity.

Rules: NEVER change any number, unit, set/rep count, RIR/RPE value, calorie/macro
value, or URL from the raw text. Preserve exercise/food names, coach-specific
terminology, and Arabic text as-is (do not translate). Only fix OCR fragmentation,
typos, and formatting.
"""

TRAINING_PROMPT = """You are cleaning up raw OCR output from a fitness coach's training plan.

Raw page text:
---
{raw_text}
---

1. Reconstruct fragmented OCR lines into clean, readable Markdown prose.
2. Infer a short page_topic label from the page's own content (e.g. "LEGS",
   "FAQS", "Chest & Back", "Cardio") -- the page may have no heading of its
   own in the raw text; use your judgement from what's actually on the page.
3. """ + TRAINING_CATEGORIES + """

Rules: NEVER change any number, unit, set/rep count, RIR/RPE value, or URL from the raw
text. Preserve exercise/food names, coach-specific terminology, and Arabic text as-is
(do not translate). Only fix OCR fragmentation, typos, and formatting.
"""
