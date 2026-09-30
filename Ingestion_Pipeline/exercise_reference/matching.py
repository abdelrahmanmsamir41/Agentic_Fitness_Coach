"""Resolves exercise names to reference rows: rules.py first, then BGE-m3 cosine top-K
re-ranked by RapidFuzz, since short, near-identical names are where lexical overlap helps most.
"""
import numpy as np
from rapidfuzz import fuzz

from . import embeddings
from .rules import rule_based_fields

TOP_K = 5
FUZZY_WEIGHT = 0.3
THRESHOLD = 0.6


def _find_similarity_match(query, query_vector, ref_vectors, reference):
    scores = ref_vectors @ query_vector  # both L2-normalized -> dot product is cosine similarity
    candidate_count = min(TOP_K, len(scores))
    top_indexes = np.argpartition(scores, -candidate_count)[-candidate_count:]

    query_lower = query.lower()
    best_index, best_score = None, -1.0
    for index in top_indexes:
        # token_sort_ratio is case-sensitive
        fuzzy_score = fuzz.token_sort_ratio(query_lower, reference[index]["title"].lower()) / 100
        combined_score = (1 - FUZZY_WEIGHT) * float(scores[index]) + FUZZY_WEIGHT * fuzzy_score
        if combined_score > best_score:
            best_index, best_score = index, combined_score

    if best_score < THRESHOLD:
        return None
    match = reference[best_index]
    return {
        "ref_type": match["type"],
        "ref_bodypart": match["bodypart"],
        "ref_equipment": match["equipment"],
        "ref_level": match["level"],
        "ref_match_score": round(best_score, 3),
    }


def _resolve_exercise(name, query_vector, ref_vectors, reference):
    rule_fields = rule_based_fields(name)
    match = _find_similarity_match(name, query_vector, ref_vectors, reference)

    if not rule_fields and match is None:
        return None

    result = match or {"ref_type": None, "ref_bodypart": None,
                        "ref_equipment": None, "ref_level": None, "ref_match_score": None}
    result.update(rule_fields)
    return result


def match_exercises(names):
    """names: list of exercise-name strings. Returns a same-length list, each
    entry either an enrichment dict (ref_type/ref_bodypart/ref_equipment/
    ref_level/ref_match_score) or None when neither a keyword rule nor the
    similarity matcher found anything usable. Rule-derived fields always win
    over the matched row's own fields for the same key."""
    if not names:
        return []

    reference = embeddings.get_reference()
    ref_vectors = embeddings.get_reference_embeddings()
    query_vectors = embeddings.embed_queries(names)

    return [_resolve_exercise(name, query_vector, ref_vectors, reference) for name, query_vector in zip(names, query_vectors)]


def _to_unit_vector(vector):
    return vector / np.linalg.norm(vector)


def _demo():
    reference = [
        {"title": "Incline Bench Press", "type": "Strength", "bodypart": "Chest", "equipment": "Barbell", "level": "Intermediate"},
        {"title": "Hack Squat", "type": "Strength", "bodypart": "Quadriceps", "equipment": "Machine", "level": "Intermediate"},
    ]
    ref_vectors = embeddings.normalize(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))

    # rule beats a confident match that disagrees with it
    close_vector = _to_unit_vector(np.array([0.95, 0.05], dtype=np.float32))
    match = _resolve_exercise("Cable incline chest press", close_vector, ref_vectors, reference)
    assert match["ref_equipment"] == "Cable", "keyword rule should override the matched row's equipment"
    assert match["ref_bodypart"] == "Chest", "unmatched-by-rule fields still come from the similarity match"

    # rule fires even with no confident similarity match at all
    far_vector = _to_unit_vector(np.array([-1.0, -0.2], dtype=np.float32))
    rule_only = _resolve_exercise("Barbell some made-up lift nobody has heard of", far_vector, ref_vectors, reference)
    assert rule_only == {"ref_type": None, "ref_bodypart": None,
                          "ref_equipment": "Barbell", "ref_level": None, "ref_match_score": None}

    # neither a rule nor a confident match -> nothing to add
    no_match = _resolve_exercise("Completely unrelated made-up thing", far_vector, ref_vectors, reference)
    assert no_match is None

    # tied cosine, so only the fuzzy term decides; this real pairing lost to the wrong title case-sensitively
    tied_vector = _to_unit_vector(np.array([1.0, 1.0], dtype=np.float32))
    case_reference = [
        {"title": "Seated Leg Curl", "type": "Strength", "bodypart": "Hamstrings", "equipment": "Machine", "level": "Beginner"},
        {"title": "Seated finger curl", "type": "Strength", "bodypart": "Forearms", "equipment": "Barbell", "level": "Beginner"},
    ]
    case_vectors = np.array([tied_vector, tied_vector])
    match = _resolve_exercise("Seated leg curl", tied_vector, case_vectors, case_reference)
    assert match["ref_bodypart"] == "Hamstrings", "case difference must not lose to a same-case wrong title"


if __name__ == "__main__":
    _demo()
    print("ok")
