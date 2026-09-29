"""Extraction prompt for the attached-PDF call -- the OCR pipeline's prompt minus its OCR-specific parts."""

PDF_TRAINING_PROMPT = """Extract structured data from the attached PDF: pages of a fitness coach's training plan.
Return exactly one result per PDF page. page_index: the PDF's first page is {first}, the next {second}, and so on.

Per page:
- page_topic: short label inferred from the page's content (e.g. "LEGS", "FAQS", "Chest & Back").
- category: the page's dominant content, one of:
  weight_exercise: gym exercise table/list with sets/reps/rest
  cardio: treadmill/running/cycling... with duration/speed/incline
  glossary: defines fitness terms (e.g. RIR, W.U Sets, Supersets)
  advice: general coaching rules, mistakes to avoid, motivation
  program_structure: table mapping MULTIPLE days to muscle groups (the whole split)
  rest_day: a lone page marking ONE day as rest
  marketing: promo codes, discounts, "order now"
  other: anything else (e.g. a bare cover or title page)
- Fill EVERY list that has matching content on the page, not only the category's.

Values:
- Never change numbers, units or URLs. Keep exercise names, coach terminology and Arabic exactly as written.
- Numbers: single value -> {{"low": N}}; range "8-12" -> {{"low": 8, "high": 12}}; blank cell -> {{"low": 0}};
  non-numeric ("AMRAP", "to failure") -> null.
- video_url: a page's links below are top-to-bottom; assign them to its table rows in the same order.

Links per page (pages not listed have none):
{links_block}
"""
