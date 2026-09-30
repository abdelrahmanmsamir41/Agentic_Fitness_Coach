from typing import Literal

from langchain.tools import ToolRuntime, tool

from . import matching, plan_store
from .context import TrainerContext

TOP_K = 5

Runtime = ToolRuntime[TrainerContext]


@tool
async def get_program_overview(runtime: Runtime) -> list:
    """Every day number with its target."""
    program = await plan_store.load_program(runtime.context.client_id)
    return [{"day_number": day["day_number"], "target": day["target"]} for day in program["days"]]


@tool(parse_docstring=True)
async def get_day(day: str, runtime: Runtime) -> dict:
    """A full day's exercises and cardio.

    Args:
        day: Day number or target, e.g. "2", "legs".
    """
    days = (await plan_store.load_program(runtime.context.client_id))["days"]
    return matching.make_readable(matching.pick_day(days, day)) or {"error": f"no day matching {day!r}", "days": [program_day["target"] for program_day in days]}


@tool(parse_docstring=True)
async def count_exercises(days: list[str], runtime: Runtime) -> list:
    """Exercise count per day.

    Args:
        days: Day numbers or targets, e.g. ["3", "legs"].
    """
    program_days = (await plan_store.load_program(runtime.context.client_id))["days"]
    counts = []
    for day in days:
        matched_day = matching.pick_day(program_days, day)
        counts.append({"day_number": matched_day["day_number"], "target": matched_day["target"],
                       "exercise_count": len(matched_day["weight_exercises"])}
                      if matched_day else {"error": f"no day matching {day!r}"})
    return counts


@tool(parse_docstring=True)
async def get_exercise(name: str, runtime: Runtime) -> list:
    """One exercise's details, video and day.

    Args:
        name: Exercise name, fuzzy.
    """
    hits = matching.pick_exercises(await plan_store.load_program(runtime.context.client_id), name)
    return matching.make_readable(hits, include_video=True)


@tool(parse_docstring=True)
async def list_exercises(
    runtime: Runtime,
    field: Literal["working_sets", "warmup_sets", "reps", "rir"] | None = None,
    above: float | None = None,
    below: float | None = None,
    only_missing: bool = False,
    equipment: str | None = None,
    bodypart: str | None = None,
    day: str | None = None,
    position: int | None = None,
) -> list:
    """All plan exercises, optionally filtered, grouped by day.

    Args:
        field: Number to filter on.
        above: Strictly above ("more than 2" -> 2).
        below: Strictly below ("fewer than 10" -> 10).
        only_missing: Only where field is blank.
        equipment: e.g. "dumbbell".
        bodypart: e.g. "chest". Only when the client asks for a muscle, not because a day's name mentions one.
        day: Only this day number or target, e.g. "2", "legs". "Leg day", "chest day" etc. are days, not bodyparts.
        position: Place in the day's plan order: 1 = first, 2 = second, -1 = last.
    """
    program = await plan_store.load_program(runtime.context.client_id)
    matched_day = None
    if day:
        matched_day = matching.pick_day(program["days"], day)
        if not matched_day:
            return [{"error": f"no day matching {day!r}", "days": [program_day["target"] for program_day in program["days"]]}]
        program = {**program, "days": [matched_day]}
    hits = matching.filter_exercises(program, field, above, below, only_missing, equipment, bodypart, position)
    if not hits and matched_day:
        return [_why_no_match(matched_day)]
    return matching.make_readable(matching.group_by_day(hits))


def _why_no_match(day):
    """What the model needs to explain an empty result instead of just saying "none"."""
    return {"day_number": day["day_number"], "target": day["target"], "rest_day": matching.is_rest_day(day),
            "exercises_that_day": len(day["weight_exercises"]), "matches": 0}


@tool(parse_docstring=True)
async def plan_value_counts(field: Literal["rest", "reps", "working_sets", "warmup_sets", "rir"], runtime: Runtime) -> dict:
    """How many exercises have each value of one field.

    Args:
        field: Field to count.
    """
    return matching.value_counts(await plan_store.load_program(runtime.context.client_id), field)


@tool
async def get_cardio(runtime: Runtime) -> dict:
    """The plan's cardio: general (every training day) and day-specific."""
    program = await plan_store.load_program(runtime.context.client_id)
    general = program["general_cardio"]
    specific = [{"day_number": day["day_number"], "target": day["target"], "cardio": own_cardio}
                for day in program["days"]
                if (own_cardio := [entry for entry in day["cardio_entries"] if entry not in general])]
    return matching.make_readable({
        "general_cardio_every_training_day": general,
        "rest_days": [day["day_number"] for day in program["days"] if matching.is_rest_day(day)],
        "day_specific": specific,
    })


@tool(parse_docstring=True)
async def search_coach_notes(query: str, runtime: Runtime) -> list:
    """Search coach's prose notes; low score = not covered.

    Args:
        query: Short English search query; translate Arabic, keep fitness terms like RIR.
    """
    return await plan_store.search_notes(runtime.context.client_id, query, TOP_K)


@tool(parse_docstring=True)
async def find_alternatives(exercise: str, runtime: Runtime, avoid_equipment: str | None = None) -> list:
    """Substitutes for the same body part.

    Args:
        exercise: Exercise to replace.
        avoid_equipment: Equipment to exclude, e.g. "Barbell".
    """
    hits = matching.pick_exercises(await plan_store.load_program(runtime.context.client_id), exercise)
    bodypart = (hits and hits[0].get("ref_bodypart")) or await plan_store.lookup_reference_bodypart(exercise)
    if not bodypart:
        return []
    reference, ref_vectors = plan_store.get_reference_set()
    return matching.rank_alternatives(
        reference, ref_vectors, await plan_store.embed_exercise(exercise), bodypart,
        exclude_titles=[exercise, *(hit["exercise"] for hit in hits)], avoid_equipment=avoid_equipment, limit=TOP_K,
    )


TOOLS =[
         get_program_overview, 
         get_day,
         count_exercises,
         get_exercise,
         list_exercises, 
         plan_value_counts,
         get_cardio, 
         search_coach_notes, 
         find_alternatives
        ]


def _demo():
    for trainer_tool in TOOLS:
        assert "runtime" not in trainer_tool.args, f"{trainer_tool.name} leaks runtime into the LLM schema"
    assert "Day number" in get_day.args["day"]["description"]


if __name__ == "__main__":
    _demo()
    print("ok")
