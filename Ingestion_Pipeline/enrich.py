"""Post-extraction enrichment passes over one document's all_extractions --
kept separate from extraction/extract.py, whose only job is turning one
page's raw text into structured data. These are three different concerns:
day-number resolution is a second, independent Gemini call over page topics;
exercise-reference matching is local entity-linking with no LLM in it at all;
build_program_view is a pure reshape of already-enriched data into a
day-by-day view. Called from ingest.py, after extract_document() returns.
"""
import exercise_reference
from extraction import prompts
from extraction.extract import get_model
from extraction.models import DayAssignmentResult


async def assign_day_numbers(day_entries, page_topics):
    """day_entries: iterable of (day_number, target), from the split-table page's
    program_days. page_topics: iterable of (page_index, page_topic), for every
    page that might belong to one specific day (see _is_day_candidate).
    Returns {page_index: day_number}, omitting pages the model couldn't place
    in any listed day.

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

    valid_day_numbers = {day_number for day_number, _ in day_entries}
    return {assignment.page_index: assignment.day_number
            for assignment in result.assignments if assignment.day_number in valid_day_numbers}


def _is_split_table(extraction):
    """The real weekly split table -- "rest_day" is a separate category now
    (a standalone single-day marker)"""
    return extraction["category"] == "program_structure"


def _is_day_candidate(extraction):
    """Any page whose own page_topic might name one specific day: exercise/cardio
    pages (need day_number for filtering), "other" pages (duplicate "X Training
    Log" pages with no structured entities of their own), and "rest_day" pages --
    resolved through the exact same matching-against-the-split-table pipeline as
    everything else here, not a separate mechanism. Pages that clearly don't
    belong to any day get null back from assign_day_numbers rather than a guess,
    so over-including candidates here is safe."""
    return extraction["category"] in ("weight_exercise", "cardio", "other", "rest_day")


def _split_table_day_entries(all_extractions):
    """The plan's canonical (day_number, target) list, pooled from whichever
    page(s) hold the real split table."""
    return sorted({
        (day["day_number"], day["target"])
        for extraction in all_extractions.values() if _is_split_table(extraction)
        for day in extraction["program_days"]
    })


async def resolve_day_numbers(doc_id, all_extractions):
    """Stamps day_number onto every page that isn't itself the plan's split
    table, resolved from whichever page(s) hold that table."""
    day_entries = _split_table_day_entries(all_extractions)
    page_topics = sorted(
        (page, extraction["page_topic"])
        for page, extraction in all_extractions.items() if _is_day_candidate(extraction)
    )  # ascending page_index -- the prompt relies on this being real document order
    day_numbers = await assign_day_numbers(day_entries, page_topics)
    if day_numbers:
        print(f"📅 [{doc_id}] assigned day numbers to {len(day_numbers)}/{len(page_topics)} page(s)")
    for page, day_number in day_numbers.items():
        all_extractions[page]["day_number"] = day_number


def resolve_exercise_reference(doc_id, all_extractions):
    """Stamps ref_type/ref_bodypart/ref_equipment/ref_level/ref_match_score onto
    every weight_exercise entity, matched against the static megaGymDataset
    reference set (exercise_reference/ package) -- no LLM involved, purely local.
    Collected by list contents, not by category=="weight_exercise" -- a page's
    dominant category no longer gates which lists it can populate, so an
    exercise named on e.g. an "advice" page still needs matching."""
    exercises = [
        entity
        for extraction in all_extractions.values()
        for entity in extraction["weight_exercises"]
    ]
    if not exercises:
        return

    names = [exercise.get("exercise", "") for exercise in exercises]
    matches = exercise_reference.match_exercises(names)
    matched_count = 0
    for entity, match in zip(exercises, matches):
        if match:
            entity.update(match)
            matched_count += 1
    print(f"🏋️ [{doc_id}] matched {matched_count}/{len(exercises)} exercise(s) against the reference dataset")


def build_program_view(all_extractions):
    """Pure reshape (no LLM, nothing mutated) of the already-enriched
    all_extractions into a day-by-day view: each split-table day gets the
    weight_exercises/cardio_entries pulled from every page whose resolved
    day_number matches it -- a rest day naturally ends up with empty lists,
    its target already says "rest". Cardio with no resolved day_number (a
    generic "Cardio" page not tied to one day) goes in general_cardio instead
    of being duplicated onto every training day or attached to a rest day.
    """
    day_entries = _split_table_day_entries(all_extractions)
    days = [{"day_number": day_number, "target": target, "weight_exercises": [], "cardio_entries": []}
            for day_number, target in day_entries]
    days_by_number = {day["day_number"]: day for day in days}

    general_cardio = []
    for extraction in all_extractions.values():
        day = days_by_number.get(extraction.get("day_number"))
        if day is not None:
            day["weight_exercises"].extend(extraction["weight_exercises"])
            day["cardio_entries"].extend(extraction["cardio_entries"])
        elif extraction["cardio_entries"]:
            general_cardio.extend(extraction["cardio_entries"])

    return {"days": days, "general_cardio": general_cardio}


async def enrich_document(doc_id, all_extractions):
    """Runs both LLM/local enrichment passes over one document's extractions,
    in place."""
    await resolve_day_numbers(doc_id, all_extractions)
    resolve_exercise_reference(doc_id, all_extractions)


def _demo():
    def other():
        return {"category": "other", "program_days": [], "weight_exercises": [], "cardio_entries": []}

    table = {"category": "program_structure", "program_days": [
        {"day_number": 1, "target": "chest"}, {"day_number": 2, "target": "rest"},
    ], "weight_exercises": [], "cardio_entries": []}
    assert _is_split_table(table) is True
    assert _is_split_table(other()) is False

    rest_marker = {**other(), "category": "rest_day"}
    assert _is_day_candidate(rest_marker) is True
    assert _is_day_candidate(table) is False  # the table itself is never a candidate for its own resolution

    all_extractions = {
        0: table,
        1: {**other(), "category": "weight_exercise", "day_number": 1,
            "weight_exercises": [{"exercise": "Bench"}]},
        2: {**other(), "category": "rest_day", "day_number": 2},
        3: {**other(), "category": "cardio", "day_number": None,
            "cardio_entries": [{"activity": "Treadmill"}]},
    }
    view = build_program_view(all_extractions)
    assert view["days"] == [
        {"day_number": 1, "target": "chest", "weight_exercises": [{"exercise": "Bench"}], "cardio_entries": []},
        {"day_number": 2, "target": "rest", "weight_exercises": [], "cardio_entries": []},
    ]
    assert view["general_cardio"] == [{"activity": "Treadmill"}]


if __name__ == "__main__":
    _demo()
    print("ok")
