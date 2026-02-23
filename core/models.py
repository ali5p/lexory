from datetime import datetime, timezone
from typing import Optional, List, Dict

from pydantic import BaseModel, Field, ConfigDict


class UserText(BaseModel):
    text: str
    user_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MistakeInstance(BaseModel):
    text: str
    mistake_type: str
    correction: str
    user_id: str
    timestamp: datetime = Field(default_factory=datetime.now)


class LearningSummary(BaseModel):
    summary_id: str
    user_id: str
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LessonArtifact(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(alias="artifact_id")
    user_id: str
    session_id: Optional[str]
    mistake_types_covered: List[str]
    pedagogy_tags: List[str]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SessionContext(BaseModel):
    user_id: str
    session_id: str
    context_data: dict[str, str]
    created_at: datetime = Field(default_factory=datetime.now)


class SubmitRequest(BaseModel):
    """Combined ingest + lesson: text (optional for fallback), user_id."""

    text: str = ""
    user_id: str


class SubmitResponse(BaseModel):
    """Full system output for testing; production UI shows only lesson."""

    user_text_id: Optional[str] = None
    session_id: str
    lesson_artifact_id: str
    lesson: "LessonResponse"
    context: "ContextAssembly"
    status: str


class ContextAssembly(BaseModel):
    detected_mistake_examples: List[Dict]
    long_term_dynamics: List[Dict]
    recently_used_explanations: List[Dict]


class LessonResponse(BaseModel):
    topic: str
    explanation: str
    exercises: list[str]
    approach_type: str


class QueryResponse(BaseModel):
    lesson: LessonResponse
    context: ContextAssembly


class ExerciseAttempt(BaseModel):
    text: str
    user_id: str
    lesson_artifact_id: str  # Required: links attempt to the lesson it exercises
