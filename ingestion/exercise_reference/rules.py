"""Gazetteer keyword rules: near-certain precision whenever an exercise name
spells out its own equipment/bodypart/type ("Cable lat pullover" -> Cable,
Lats), checked with word-boundary regex so e.g. "Cable" never fires a bare
"ab" bodypart rule. These always win over whatever matching.py's similarity
match would have said for the same field -- see exercise_reference/__init__.py.
"""
import re

# Ordered most-specific first -- first keyword hit wins, so e.g. "smith machine"
# must be checked before the bare "machine" fallback at the bottom.
_EQUIPMENT_RULES = [
    (("e-z", "ez", "ez-bar", "ez bar", "ez curl"), "E-Z Curl Bar"),
    (("smith machine", "smith"), "Machine"),
    (("leverage machine",), "Machine"),
    (("kettlebell", "kettlebells"), "Kettlebells"),
    (("medicine ball",), "Medicine Ball"),
    (("exercise ball", "stability ball", "swiss ball"), "Exercise Ball"),
    (("foam roll", "foam roller"), "Foam Roll"),
    (("band", "bands", "banded"), "Bands"),
    (("cable",), "Cable"),
    (("dumbbell", "dumbbells", "dumbell", "dumbells"), "Dumbbell"),
    (("landmine", "trap bar", "barbell", "barbells"), "Barbell"),
    (("bodyweight", "body weight", "body-weight"), "Body Only"),
    (("rope",), "Rope"),
    (("machine",), "Machine"),
]

_BODYPART_RULES = [
    (("hamstring", "hamstrings"), "Hamstrings"),
    (("quad", "quads", "quadriceps", "thigh", "thighs"), "Quadriceps"),
    (("calf", "calves"), "Calves"),
    (("glute", "glutes"), "Glutes"),
    (("adductor", "adductors", "groin"), "Adductors"),
    (("abductor", "abductors"), "Abductors"),
    (("bicep", "biceps"), "Biceps"),
    (("tricep", "triceps"), "Triceps"),
    (("forearm", "forearms", "wrist", "grip"), "Forearms"),
    (("shoulder", "shoulders", "delt", "delts", "deltoid"), "Shoulders"),
    (("chest", "pec", "pecs", "pectoral"), "Chest"),
    (("shrug", "shrugs"), "Traps"),
    (("lat", "lats", "pulldown", "pullover"), "Lats"),
    (("neck",), "Neck"),
    (("lower back",), "Lower Back"),
    (("ab", "abs", "abdominal", "abdominals", "crunch", "sit-up", "situp", "plank"), "Abdominals"),
]

# Deliberately tiny: a name essentially never signals anything else about type,
# and never signals difficulty ("level") at all -- both are left to matching.py.
_TYPE_RULES = [
    (("stretch", "stretching"), "Stretching"),
]


def _first_rule_match(name, rules):
    for keywords, value in rules:
        for keyword in keywords:
            if re.search(rf"\b{re.escape(keyword)}\b", name, re.IGNORECASE):
                return value
    return None


def rule_based_fields(name):
    """Keyword-rule overrides only -- whichever of ref_equipment/ref_bodypart/
    ref_type the name itself spells out. Missing keys mean "no rule fired,
    fall back to the similarity match" for that field."""
    fields = {}
    equipment = _first_rule_match(name, _EQUIPMENT_RULES)
    if equipment:
        fields["ref_equipment"] = equipment
    bodypart = _first_rule_match(name, _BODYPART_RULES)
    if bodypart:
        fields["ref_bodypart"] = bodypart
    ex_type = _first_rule_match(name, _TYPE_RULES)
    if ex_type:
        fields["ref_type"] = ex_type
    return fields


def _demo():
    assert rule_based_fields("Cable lat pullover") == {"ref_equipment": "Cable", "ref_bodypart": "Lats"}
    assert rule_based_fields("Smith standing calf raises") == {"ref_equipment": "Machine", "ref_bodypart": "Calves"}
    assert rule_based_fields("EZ-bar skullcrusher") == {"ref_equipment": "E-Z Curl Bar"}
    assert rule_based_fields("Trap bar deadlift") == {"ref_equipment": "Barbell"}, "trap bar != Traps bodypart"
    # word-boundary check: "Cable" must never fire the bare "ab" bodypart rule
    assert rule_based_fields("Cable crossover") == {"ref_equipment": "Cable"}
    assert rule_based_fields("Bodyweight squat") == {"ref_equipment": "Body Only"}
    assert rule_based_fields("Nordic hamstring curl") == {"ref_bodypart": "Hamstrings"}
    assert rule_based_fields("Completely unrelated made-up thing") == {}


if __name__ == "__main__":
    _demo()
    print("ok")
