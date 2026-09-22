"""Post-extraction enrichment passes over one document's all_extractions --
kept separate from extraction/extract.py, whose only job is turning one
page's raw text into structured data. These are two different concerns:
day-number resolution is a second, independent Gemini call over page topics;
exercise-reference matching is local entity-linking with no LLM in it at all.
Called from ingest.py, after extract_document() returns.
"""
import exercise_reference
from extraction import prompts
from extraction.extract import get_model
from extraction.models import DayAssignmentResult


async def assign_day_numbers(day_entries, page_topics):
    """day_entries: iterable of (day_number, target), from the split-table page's
    program_days. page_topics: iterable of (page_index, page_topic), for every
    weight_exercise/cardio page in the document. Returns {page_index: day_number},
    omitting pages the model couldn't place in any listed day.

    One classification call over page_topics only (not full page text) -- cheap,
    and independent of the extraction batching, so it doesn't need the split-table
    page's batch to run before any other page's batch.
    """
    day_entries, page_topics = list(day_entries), list(page_topics)
    if not day_entries or not page_topics:
        return {}

    structured_model = get_model().with_structured_output(DayAssignmentResult)
    prompt = prompts.build_day_assignment_prompt(day_entries, page_topics)
    print(f"📅 assigning day numbers for {len(page_topics)} page(s) against {len(day_entries)} day(s)...")
    result = await structured_model.ainvoke(prompt)

    valid_days = {d for d, _ in day_entries}
    return {a.page_index: a.day_number for a in result.assignments if a.day_number in valid_days}


async def resolve_day_numbers(doc_id, all_extractions):
    """Stamps day_number onto every weight_exercise/cardio page's extraction,
    resolved from whichever page(s) got category=="program_structure"."""
    day_entries = sorted({
        (day["day_number"], day["target"])
        for extraction in all_extractions.values() if extraction["category"] == "program_structure"
        for day in extraction["program_days"]
    })
    page_topics = [
        (page, extraction["page_topic"])
        for page, extraction in all_extractions.items() if extraction["category"] in ("weight_exercise", "cardio")
    ]
    day_numbers = await assign_day_numbers(day_entries, page_topics)
    if day_numbers:
        print(f"📅 [{doc_id}] assigned day numbers to {len(day_numbers)}/{len(page_topics)} exercise/cardio page(s)")
    for page, day_number in day_numbers.items():
        all_extractions[page]["day_number"] = day_number


def resolve_exercise_reference(doc_id, all_extractions):
    """Stamps ref_type/ref_bodypart/ref_equipment/ref_level/ref_match_score onto
    every weight_exercise entity, matched against the static megaGymDataset
    reference set (exercise_reference/ package) -- no LLM involved, purely local."""
    exercises = [
        entity
        for extraction in all_extractions.values() if extraction["category"] == "weight_exercise"
        for entity in extraction["weight_exercises"]
    ]
    if not exercises:
        return

    names = [e.get("exercise", "") for e in exercises]
    matches = exercise_reference.match_exercises(names)
    matched = 0
    for entity, match in zip(exercises, matches):
        if match:
            entity.update(match)
            matched += 1
    print(f"🏋️ [{doc_id}] matched {matched}/{len(exercises)} exercise(s) against the reference dataset")


async def enrich_document(doc_id, all_extractions):
    """Runs both enrichment passes over one document's extractions, in place."""
    await resolve_day_numbers(doc_id, all_extractions)
    resolve_exercise_reference(doc_id, all_extractions)


if __name__ == "__main__":
    print("ok")
