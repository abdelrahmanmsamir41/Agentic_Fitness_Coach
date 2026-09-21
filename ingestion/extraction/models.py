"""Pydantic schemas for normalize+extract Gemini calls."""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class NutritionEntity(BaseModel):
    meal: str = Field(description="Meal name, e.g. \"Breakfast\"")
    food: Optional[str] = Field(default=None, description="Food item name")
    quantity: Optional[str] = Field(default=None, description="Amount, e.g. \"200\"")
    unit: Optional[str] = Field(default=None, description="Unit for quantity, e.g. \"g\", \"cup\"")
    calories: Optional[str] = Field(default=None, description="Calorie count")
    macros: Optional[str] = Field(default=None, description="Macro breakdown, e.g. \"P40/C50/F10\"")
    substitutions: Optional[str] = Field(default=None, description="Allowed substitute foods")


class NutritionSectionResult(BaseModel):
    page_topic: str = Field(description="Short label inferred from the page's own content, e.g. \"Breakfast\", \"Post-Workout Shake\" -- the page may have no heading of its own, so infer it from what's actually there")
    cleaned_text: str = Field(description="Raw OCR text reconstructed into clean, readable Markdown prose")
    entities: List[NutritionEntity] = Field(description="Meal entries extracted from the page")


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


class NumberRange(BaseModel):
    low: float = Field(description="The number itself, or the lower bound of a range")
    high: Optional[float] = Field(default=None, description="Upper bound if this is a range like \"8-12\"; omit for a single number")


class WeightExerciseEntity(BaseModel):
    exercise: str = Field(description="Exercise name")
    working_sets: Optional[NumberRange] = Field(default=None, description="Number of working sets")
    reps: Optional[NumberRange] = Field(default=None, description="Rep count or range per set")
    warmup_sets: Optional[NumberRange] = Field(default=None, description="Number of warmup sets")
    rir: Optional[NumberRange] = Field(default=None, description="Reps in reserve")
    rest: Optional[str] = Field(default=None, description="Rest duration between sets, e.g. \"60-90s\"")
    instructions: Optional[str] = Field(default=None, description="Coaching notes or technique cues for this exercise")
    video_url: Optional[str] = Field(default=None, description="Link to a demo video, if present")


class CardioEntity(BaseModel):
    activity: str = Field(description="Cardio activity, e.g. \"Treadmill\", \"Cycling\"")
    duration: Optional[str] = Field(default=None, description="Duration, e.g. \"20-30 min\"")
    speed: Optional[NumberRange] = Field(default=None, description="Speed, e.g. km/h")
    incline: Optional[NumberRange] = Field(default=None, description="Incline percentage")
    timing: Optional[str] = Field(default=None, description="When to do it, e.g. \"post-workout\"")
    instructions: Optional[str] = Field(default=None, description="Coaching notes for this activity")
    video_url: Optional[str] = Field(default=None, description="Link to a demo video, if present")


class GlossaryTermEntity(BaseModel):
    term: str = Field(description="Fitness term or abbreviation, e.g. \"RIR\"")
    explanation: str = Field(description="Plain-language definition of the term")


class AdviceEntity(BaseModel):
    text: str = Field(description="The advice, rule, mistake to avoid, or motivation text itself")


class ProgramDayEntity(BaseModel):
    day_number: int = Field(description="Day number within the program")
    target: str = Field(description="Muscle group(s) targeted that day, or \"rest\"")


class TrainingSectionResult(BaseModel):
    page_index: int = Field(description="The page marker number (from \"=== PAGE N ===\") this result corresponds to")
    page_topic: str = Field(description="Short label inferred from the page's own content, e.g. \"LEGS\", \"FAQS\", \"Chest & Back\" -- the page may have no heading of its own, so infer it from what's actually there")
    category: SectionCategory = Field(description="Which kind of content this page contains")
    cleaned_text: str = Field(description="Raw OCR text reconstructed into clean, readable Markdown prose")
    weight_exercises: List[WeightExerciseEntity] = Field(default_factory=list, description="Exercise entries, only when category is \"weight_exercise\"")
    cardio_entries: List[CardioEntity] = Field(default_factory=list, description="Cardio entries, only when category is \"cardio\"")
    glossary_terms: List[GlossaryTermEntity] = Field(default_factory=list, description="Term definitions, only when category is \"glossary\"")
    advice_items: List[AdviceEntity] = Field(default_factory=list, description="Advice entries, only when category is \"advice\"")
    program_days: List[ProgramDayEntity] = Field(default_factory=list, description="Day/split entries, only when category is \"program_structure\"")


class TrainingBatchResult(BaseModel):
    pages: List[TrainingSectionResult] = Field(description="One result per input page")


TRAINING_LIST_FIELDS = ("weight_exercises", "cardio_entries", "glossary_terms", "advice_items", "program_days")
