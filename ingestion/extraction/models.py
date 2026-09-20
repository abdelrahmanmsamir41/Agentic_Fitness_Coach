"""Pydantic schemas for normalize+extract Gemini calls."""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class NutritionEntity(BaseModel):
    meal: str
    food: Optional[str] = None
    quantity: Optional[str] = None
    unit: Optional[str] = None
    calories: Optional[str] = None
    macros: Optional[str] = None
    substitutions: Optional[str] = None


class NutritionSectionResult(BaseModel):
    # page_topic is model-inferred (e.g. "Breakfast", "Post-Workout Shake"),
    # not lifted from a markdown heading -- a page's own OCR heading is
    # unreliable (missing, or matching the wrong line among several), so the
    # model infers it directly from the page's content instead.
    page_topic: str
    cleaned_text: str
    entities: List[NutritionEntity]


# --- Training-plan section categories -------------------------------------
# A training plan mixes several distinct kinds of content on different pages
# (exercise tables, a day/split overview, a glossary of coaching terms,general advice, cardio, marketing)
# -- one generic "exercise" schema for the whole document either force-fits everything into exercise fields
# (cardio, glossary) or throws it all into an untyped fallback text blob. Classifying each section and extracting
# into the matching typed list keeps embeddings specific (e.g. a glossary leaf is just "RIR: how many reps...",
# not buried in a whole-section paragraph) instead of generic.

SectionCategory = Literal[
    "weight_exercise", "cardio", "glossary", "advice", "program_structure", "marketing", "other"
]


class WeightExerciseEntity(BaseModel):
    exercise: str
    working_sets: Optional[str] = None
    reps: Optional[str] = None
    warmup_sets: Optional[str] = None
    rir: Optional[str] = None
    rest: Optional[str] = None
    instructions: Optional[str] = None
    video_url: Optional[str] = None


class CardioEntity(BaseModel):
    activity: str
    duration: Optional[str] = None
    speed: Optional[str] = None
    incline: Optional[str] = None
    timing: Optional[str] = None  # e.g. "post-workout"
    instructions: Optional[str] = None
    video_url: Optional[str] = None


class GlossaryTermEntity(BaseModel):
    term: str
    explanation: str


class AdviceEntity(BaseModel):
    topic: Optional[str] = None
    text: str


class ProgramDayEntity(BaseModel):
    day_number: int
    target: str  # muscle group(s) targeted, or "rest"


class TrainingSectionResult(BaseModel):
    # Same rationale as NutritionSectionResult.page_topic above -- e.g. "LEGS",
    # "FAQS", "Chest & Back" -- inferred by the model from the page's own
    # content, not lifted from a markdown heading line.
    page_topic: str
    category: SectionCategory
    cleaned_text: str
    weight_exercises: List[WeightExerciseEntity] = Field(default_factory=list)
    cardio_entries: List[CardioEntity] = Field(default_factory=list)
    glossary_terms: List[GlossaryTermEntity] = Field(default_factory=list)
    advice_items: List[AdviceEntity] = Field(default_factory=list)
    program_days: List[ProgramDayEntity] = Field(default_factory=list)


TRAINING_LIST_FIELDS = ("weight_exercises", "cardio_entries", "glossary_terms", "advice_items", "program_days")
