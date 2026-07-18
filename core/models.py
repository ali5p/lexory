from datetime import datetime, timezone
from typing import Optional, List, Dict

from pydantic import BaseModel, ConfigDict, Field

from core.exercises import ExerciseAnswerRequest, ExerciseAnswerResponse, ExercisePayload


class UserText(BaseModel):
    text: str
    user_id: str
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MistakeInstance(BaseModel):
    text: str
    mistake_type: str
    correction: str
    user_id: str
    detected_at: datetime = Field(default_factory=datetime.now)


class LessonArtifact(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(alias="artifact_id")
    user_id: str
    session_id: Optional[str]
    topic: str = ""
    explanation: str = ""
    approach_type: str = ""
    mistake_type: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SessionContext(BaseModel):
    user_id: str
    session_id: str
    context_data: dict[str, str]
    created_at: datetime = Field(default_factory=datetime.now)


class SubmitRequest(BaseModel):
    text: str = ""
    user_id: str


class DetectedMistakeExample(BaseModel):
    """Primary mistake surfaced in submit response context (for API / debugging)."""

    mistake_id: Optional[str] = None
    rule_id: str = Field(
        default="",
        description="LanguageTool rule id after normalization.",
    )
    mistake_type: str = Field(
        default="",
        description="Internal Lexory category from languagetool_to_mistaketype mapping.",
    )
    description: str = ""
    examples: List[str] = Field(default_factory=list)
    rule_message: str = ""


class ContextAssembly(BaseModel):
    """Internal context for lesson generation (not exposed on /submit)."""

    detected_mistake_examples: List[DetectedMistakeExample]
    similar_past_examples: List[Dict] = Field(
        default_factory=list,
        description=(
            "User's own prior sentences with the same mistake_type (text + "
            "rule_message), by sentence similarity. Fuel for the inductive "
            "(example_based) approach; ignored by deductive approaches."
        ),
    )


class LessonResponse(BaseModel):
    """Internal lesson result including generation metadata."""

    topic: str
    explanation: str
    approach_type: str


class LessonContent(BaseModel):
    """User-facing lesson payload (no generation metadata)."""

    topic: str
    explanation: str
    exercises: list[ExercisePayload] = Field(default_factory=list)


class LessonTarget(BaseModel):
    mistake_id: str
    rule_id: str
    mistake_type: str
    text: str
    rule_message: str = ""


class DetectedMistakeItem(BaseModel):
    mistake_id: str
    rule_id: str
    mistake_type: str
    text: str
    rule_message: str = ""
    selected_for_lesson: bool


class LessonItemResponse(BaseModel):
    lesson_artifact_id: Optional[str] = None
    target: Optional[LessonTarget] = None
    lesson: LessonContent


class SubmitResponse(BaseModel):
    user_text_id: Optional[str] = None
    session_id: str
    detected_mistakes: List[DetectedMistakeItem]
    lesson_items: List[LessonItemResponse]


class QueryResponse(BaseModel):
    lesson: LessonResponse
    context: ContextAssembly
