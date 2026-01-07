from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class UserText(BaseModel):
    text: str
    user_id: str
    timestamp: datetime = Field(default_factory=datetime.now)


class MistakeInstance(BaseModel):
    text: str
    mistake_type: str
    correction: str
    user_id: str
    timestamp: datetime = Field(default_factory=datetime.now)


class MistakePattern(BaseModel):
    pattern_id: str
    description: str
    examples: list[str]
    user_id: str
    created_at: datetime = Field(default_factory=datetime.now)


class LearningSummary(BaseModel):
    summary_id: str
    user_id: str
    content: str
    created_at: datetime = Field(default_factory=datetime.now)


class LessonArtifact(BaseModel):
    artifact_id: str
    user_id: str
    content: str
    lesson_type: str
    created_at: datetime = Field(default_factory=datetime.now)


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
    detected_patterns: list[dict]
    long_term_dynamics: list[dict]
    recently_used_explanations: list[dict]


class LessonResponse(BaseModel):
    topic: str
    explanation: str
    exercises: list[str]
    approach_type: str


class QueryResponse(BaseModel):
    lesson: LessonResponse
    context: ContextAssembly

