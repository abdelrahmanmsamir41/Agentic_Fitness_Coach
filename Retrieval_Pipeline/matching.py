"""Pure lookup logic over a program view and the exercise reference set -- no I/O."""
import re
import sys
from collections import Counter
from pathlib import Path

from rapidfuzz import fuzz, process, utils

sys.path.append(str(Path(__file__).resolve().parent.parent / "Ingestion_Pipeline"))
from exercise_reference.rules import rule_based_fields  # noqa: E402
from text_utils import format_range  # noqa: E402

MATCH_CUTOFF = 60
MAX_EXERCISE_HITS = 3
RESOLVE_CUTOFF = 85  # typos/plurals: "dumbells" -> Dumbbell; unrelated values stay well below
_REST = re.compile(r"\brest\b", re.IGNORECASE)
# reference-matching internals: used for filtering, noise (and sometimes wrong) in answers
_INTERNAL_FIELDS = {"ref_type", "ref_level", "ref_match_score", "ref_equipment"}
# group words rules.py can't map to a single bodypart
_BODYPART_GROUPS = {
    "legs": {"Quadriceps", "Hamstrings", "Glutes", "Calves", "Adductors", "Abductors"},
    "leg": {"Quadriceps", "Hamstrings", "Glutes", "Calves", "Adductors", "Abductors"},
    "back": {"Lats", "Middle Back", "Lower Back", "Traps"},
    "arms": {"Biceps", "Triceps", "Forearms"},
    "arm": {"Biceps", "Triceps", "Forearms"},
    "core": {"Abdominals"},
}


def is_rest_day(day):
    return bool(_REST.search(day["target"]))


def attach_general_cardio(program):
    """General cardio applies to every training day; rest days keep only their own cardio."""
    general = program["general_cardio"]
    days = [day if is_rest_day(day) else {**day, "cardio_entries": day["cardio_entries"] + general}
            for day in program["days"]]
    return {**program, "days": days}


def _fuzzy_match(query, choices, **kwargs):
    return process.extract(query, choices, scorer=fuzz.WRatio, processor=utils.default_process,
                           score_cutoff=MATCH_CUTOFF, **kwargs)


def pick_day(days, query):
    """query: a day number ("2") or a target ("legs"). Returns the day dict or None."""
    if query.strip().isdigit():
        return next((day for day in days if day["day_number"] == int(query)), None)
    hits = _fuzzy_match(query, [day["target"] for day in days], limit=1)
    return days[hits[0][2]] if hits else None


def _exercises_with_day(program):
    """Every plan exercise, tagged with the day it belongs to."""
    return [{"day_number": day["day_number"], "day_target": day["target"], **exercise}
            for day in program["days"] for exercise in day["weight_exercises"]]


def pick_exercises(program, name):
    """Best plan exercises matching name, each tagged with its day."""
    exercises = _exercises_with_day(program)
    hits = _fuzzy_match(name, [exercise["exercise"] for exercise in exercises], limit=MAX_EXERCISE_HITS)
    return [exercises[index] for _, _, index in hits]


def make_readable(value, include_video=False):
    """Shape a result for the model: {low, high} ranges -> "3" / "8-12" (so high=None never reads as open-ended),
    and blanks, internal matching fields and (unless wanted) video links dropped."""
    hidden_keys = _INTERNAL_FIELDS | (set() if include_video else {"video_url"})
    if isinstance(value, dict):
        if set(value) == {"low", "high"}:
            return format_range(value["low"], value["high"])
        return {key: make_readable(item, include_video) for key, item in value.items()
                if item is not None and key not in hidden_keys}
    if isinstance(value, list):
        return [make_readable(item, include_video) for item in value]
    return value


def _resolve_filter_values(query, key, known_values):
    """Canonical values a loose term means: group word, rules.py shorthand ("db"), or a close spelling."""
    if query.strip().lower() in _BODYPART_GROUPS and key == "ref_bodypart":
        return _BODYPART_GROUPS[query.strip().lower()]
    if ruled_value := rule_based_fields(query).get(key):
        return {ruled_value}  # rules win: fuzzy would also match "bb" inside "Dumbbell"
    hits = _fuzzy_match(query, known_values, limit=None)
    return {known_values[index] for _, score, index in hits if score >= RESOLVE_CUTOFF}


def filter_exercises(program, field=None, above=None, below=None, only_missing=False, equipment=None, bodypart=None):
    """Every plan exercise tagged with its day; optionally only those whose field range reaches
    above/below the bounds, or (only_missing=True) whose field is blank, and/or with that equipment/bodypart."""
    exercises = _exercises_with_day(program)
    allowed_values_by_key = {
        key: _resolve_filter_values(term, key, sorted({exercise.get(key) for exercise in exercises} - {None}))
        for key, term in (("ref_equipment", equipment), ("ref_bodypart", bodypart)) if term
    }
    matches = []
    for exercise in exercises:
        if any(exercise.get(key) not in allowed for key, allowed in allowed_values_by_key.items()):
            continue
        field_value = exercise.get(field) if field else None
        if field and bool(field_value) == only_missing:
            continue
        if field_value:
            low = field_value["low"]
            high = field_value["high"] if field_value["high"] is not None else low
            if (above is not None and high <= above) or (below is not None and low >= below):
                continue
        matches.append(exercise)
    return matches


def group_by_day(exercises):
    """Day-tagged exercises -> one entry per day, so day number/target aren't repeated on every exercise."""
    days_by_number = {}
    for exercise in exercises:
        day = days_by_number.setdefault(
            exercise["day_number"], {"day_number": exercise["day_number"], "target": exercise["day_target"], "exercises": []})
        day["exercises"].append({key: value for key, value in exercise.items() if key not in ("day_number", "day_target")})
    return list(days_by_number.values())


def value_counts(program, field):
    """How many plan exercises have each value of field, most common first; blanks count as "not listed"."""
    values = [make_readable(exercise.get(field)) or "not listed" for exercise in _exercises_with_day(program)]
    return dict(Counter(values).most_common())


def rank_alternatives(reference, ref_vectors, query_vector, bodypart, exclude_titles, avoid_equipment, limit):
    """Reference rows for bodypart, minus excluded titles/equipment, most similar to query_vector first."""
    excluded_titles = {title.lower() for title in exclude_titles}
    avoided_equipment = (avoid_equipment or "").lower()
    candidate_indexes = [index for index, row in enumerate(reference)
                         if row["bodypart"] == bodypart and row["title"].lower() not in excluded_titles
                         and not (avoided_equipment and (row["equipment"] or "").lower() == avoided_equipment)]
    candidate_indexes.sort(key=lambda index: -float(ref_vectors[index] @ query_vector))
    return [reference[index] for index in candidate_indexes[:limit]]


def _demo():
    import numpy as np

    days = [{"day_number": 1, "target": "chest & back", "weight_exercises": [{"exercise": "Hack squat machine"}]},
            {"day_number": 3, "target": "legs", "weight_exercises": [{"exercise": "Incline chest press"}]}]
    assert pick_day(days, "3")["target"] == "legs"
    assert pick_day(days, "Legs")["day_number"] == 3
    assert pick_day(days, "9") is None
    assert pick_exercises({"days": days}, "hack squat")[0]["day_number"] == 1

    sets = {"days": [{"day_number": 1, "target": "x", "weight_exercises": [
        {"exercise": "A", "working_sets": {"low": 2.0, "high": None}},
        {"exercise": "B", "working_sets": {"low": 3.0, "high": None}},
        {"exercise": "C", "working_sets": {"low": 2.0, "high": 4.0}},
        {"exercise": "D", "working_sets": None},
    ]}]}
    assert [exercise["exercise"] for exercise in filter_exercises(sets, "working_sets", above=2)] == ["B", "C"]
    assert [exercise["exercise"] for exercise in filter_exercises(sets, "working_sets", below=3)] == ["A", "C"]
    assert len(filter_exercises(sets)) == 4
    assert [exercise["exercise"] for exercise in filter_exercises(sets, "working_sets", only_missing=True)] == ["D"]
    tagged = {"days": [{"day_number": 2, "target": "arms", "weight_exercises": [
        {"exercise": "DB shoulder press", "ref_equipment": "Dumbbell", "ref_bodypart": "Shoulders"},
        {"exercise": "Rope pushdown", "ref_equipment": "Rope", "ref_bodypart": "Triceps"},
    ]}]}
    assert [exercise["exercise"] for exercise in filter_exercises(tagged, equipment="dumbbell")] == ["DB shoulder press"]
    assert [exercise["exercise"] for exercise in filter_exercises(tagged, bodypart="Triceps")] == ["Rope pushdown"]
    assert [exercise["exercise"] for exercise in filter_exercises(tagged, equipment="db")] == ["DB shoulder press"]
    assert [exercise["exercise"] for exercise in filter_exercises(tagged, bodypart="arms")] == ["Rope pushdown"]
    assert filter_exercises(tagged, equipment="bb") == [], "bb is Barbell, not a fuzzy hit on Dumbbell"
    assert filter_exercises(tagged, equipment="zzz") == []
    assert make_readable({"sets": {"low": 3.0, "high": None}, "reps": [{"low": 8.0, "high": 12.0}]}) == {"sets": "3", "reps": ["8-12"]}
    assert value_counts(sets, "working_sets") == {"2": 1, "3": 1, "2-4": 1, "not listed": 1}
    raw = {"exercise": "A", "rest": "2 min", "instructions": None, "video_url": "u", "ref_bodypart": "Chest", "ref_match_score": 0.9}
    assert make_readable(raw) == {"exercise": "A", "rest": "2 min", "ref_bodypart": "Chest"}
    assert make_readable([raw], include_video=True)[0]["video_url"] == "u"
    assert make_readable({"working_sets": {"low": 2.0, "high": None}}) == {"working_sets": "2"}, "range formatted before blanks dropped"
    grouped = group_by_day([{"day_number": 1, "day_target": "x", "exercise": "A"}, {"day_number": 1, "day_target": "x", "exercise": "B"}])
    assert grouped == [{"day_number": 1, "target": "x", "exercises": [{"exercise": "A"}, {"exercise": "B"}]}]
    rests = {"days": [{"day_number": 1, "target": "x", "weight_exercises": [{"rest": "2-3 min"}, {"rest": "2-3 min"}, {"rest": "90 s"}]}]}
    assert value_counts(rests, "rest") == {"2-3 min": 2, "90 s": 1}

    treadmill = {"activity": "Treadmill"}
    program = attach_general_cardio({"general_cardio": [treadmill], "days": [
        {"day_number": 3, "target": "LEGS", "cardio_entries": []},
        {"day_number": 4, "target": "REST DAY", "cardio_entries": []},
    ]})
    assert program["days"][0]["cardio_entries"] == [treadmill]
    assert program["days"][1]["cardio_entries"] == []
    assert is_rest_day({"target": "Rest"})
    assert not is_rest_day({"target": "interest"})

    reference = [
        {"title": "Barbell Bench Press", "bodypart": "Chest", "equipment": "Barbell"},
        {"title": "Dumbbell Bench Press", "bodypart": "Chest", "equipment": "Dumbbell"},
        {"title": "Push-up", "bodypart": "Chest", "equipment": None},
        {"title": "Squat", "bodypart": "Quadriceps", "equipment": "Barbell"},
    ]
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.5, 0.5], [0.0, 1.0]])
    ranked = rank_alternatives(reference, vectors, np.array([1.0, 0.0]), "Chest", ["Barbell Bench Press"], "barbell", 5)
    assert [row["title"] for row in ranked] == ["Dumbbell Bench Press", "Push-up"]
    assert len(rank_alternatives(reference, vectors, np.array([1.0, 0.0]), "Chest", [], None, 5)) == 3


if __name__ == "__main__":
    _demo()
    print("ok")
