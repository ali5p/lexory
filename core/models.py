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


class DocumentResponse(BaseModel):
    document_id: str
    status: str


class QueryRequest(BaseModel):
    user_id: str
    query: Optional[str] = None


class QueryResponse(BaseModel):
    lesson_content: str
    relevant_patterns: list[str]
    relevant_summaries: list[str]
    relevant_artifacts: list[str]

