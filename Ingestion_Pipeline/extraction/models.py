"""Pydantic schemas for normalize+extract Gemini calls."""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


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
    entities: List[NutritionEntity] = Field(description="Meal entries extracted from the page")


# One typed list per content kind keeps each leaf's embedding specific instead of one generic schema.

SectionCategory = Literal[
    "weight_exercise", "cardio", "glossary", "advice", "program_structure", "rest_day", "marketing", "other"
]


class NumberRange(BaseModel):
    low: float = Field(description="The number itself, or the lower bound of a range")
    high: Optional[float] = Field(default=None, description="Upper bound if this is a range like \"8-12\"; omit for a single number")

    @model_validator(mode="after")
    def _swap_reversed_bounds(self):
        # coaches write ranges either way ("1-0" RIR); the model copies them as written
        if self.high is not None and self.high < self.low:
            self.low, self.high = self.high, self.low
        return self


# plans without a split table mark days inline ("DAY 2 - Push"), and a day can run across a page break
_ENTITY_DAY_NUMBER_DESCRIPTION = (
    "N of the nearest \"DAY N\" heading above this entry in reading order -- including one on an earlier page "
    "when this page continues that day; null if the plan has no such inline day headings"
)


class WeightExerciseEntity(BaseModel):
    exercise: str = Field(description="Exercise name")
    working_sets: Optional[NumberRange] = Field(default=None, description="Number of working sets")
    reps: Optional[NumberRange] = Field(default=None, description="Rep count or range per set")
    warmup_sets: Optional[NumberRange] = Field(default=None, description="Number of warmup sets")
    rir: Optional[NumberRange] = Field(default=None, description="Reps in reserve")
    rest: Optional[str] = Field(default=None, description="Rest duration between sets, e.g. \"60-90s\"")
    instructions: Optional[str] = Field(default=None, description="Coaching notes or technique cues for this exercise")
    video_url: Optional[str] = Field(default=None, description="Link to a demo video, if present")
    day_number: Optional[int] = Field(default=None, description=_ENTITY_DAY_NUMBER_DESCRIPTION)


class CardioEntity(BaseModel):
    activity: str = Field(description="Cardio activity, e.g. \"Treadmill\", \"Cycling\"")
    duration: Optional[str] = Field(default=None, description="Duration, e.g. \"20-30 min\"")
    speed: Optional[NumberRange] = Field(default=None, description="Speed, e.g. km/h")
    incline: Optional[NumberRange] = Field(default=None, description="Incline percentage")
    timing: Optional[str] = Field(default=None, description="When to do it, e.g. \"post-workout\"")
    instructions: Optional[str] = Field(default=None, description="Coaching notes for this activity")
    video_url: Optional[str] = Field(default=None, description="Link to a demo video, if present")
    day_number: Optional[int] = Field(default=None, description=_ENTITY_DAY_NUMBER_DESCRIPTION)


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
    category: SectionCategory = Field(description="The page's dominant/primary kind of content -- not exclusive, see the list fields below")
    weight_exercises: List[WeightExerciseEntity] = Field(default_factory=list, description="Exercise entries -- populate whenever the page has any, regardless of category")
    cardio_entries: List[CardioEntity] = Field(default_factory=list, description="Cardio entries -- populate whenever the page has any, regardless of category")
    glossary_terms: List[GlossaryTermEntity] = Field(default_factory=list, description="Term definitions -- populate whenever the page has any, regardless of category")
    advice_items: List[AdviceEntity] = Field(default_factory=list, description="Advice entries -- populate whenever the page has any, regardless of category")
    program_days: List[ProgramDayEntity] = Field(default_factory=list, description="Day/split entries: every row of the plan's multi-day split table (category \"program_structure\"), or one entry per inline \"DAY N - <target>\" heading that starts a day on this page")


class TrainingBatchResult(BaseModel):
    pages: List[TrainingSectionResult] = Field(description="One result per input page")


TRAINING_LIST_FIELDS = ("weight_exercises", "cardio_entries", "glossary_terms", "advice_items", "program_days")


class DayAssignment(BaseModel):
    page_index: int = Field(description="The page marker number (from \"=== PAGE N ===\") this assignment is for")
    day_number: Optional[int] = Field(default=None, description="The matching day_number from the day list, or null if this page doesn't clearly belong to any of them")


class DayAssignmentResult(BaseModel):
    assignments: List[DayAssignment] = Field(description="One assignment per input page")
