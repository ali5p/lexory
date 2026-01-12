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


class MistakePattern(BaseModel):
    pattern_id: str
    description: str
    examples: List[str]
    user_id: str
    last_session_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


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
    patterns_covered: List[str]
    pedagogy_tags: List[str]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SessionContext(BaseModel):
    user_id: str
    session_id: str
    context_data: dict[str, str]
    created_at: datetime = Field(default_factory=datetime.now)


class DocumentRequest(BaseModel):
    text: str
    user_id: str
    session_id: Optional[str] = None


class DocumentResponse(BaseModel):
    document_id: str
    user_text_id: str
    status: str


class QueryRequest(BaseModel):
    user_id: str
    query: Optional[str] = None
    session_id: Optional[str] = None


class ContextAssembly(BaseModel):
    detected_patterns: List[Dict]
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

