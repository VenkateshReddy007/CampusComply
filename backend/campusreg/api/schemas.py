"""Pydantic request and response schemas for the CampusReg API."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ErrorSchema(BaseModel):
    code: str
    message: str


class SourceSchema(BaseModel):
    document_id: str
    document_name: str
    page: int = Field(ge=1)
    chunk_id: str
    citation: str
    text: str
    retrieval_method: str
    score: float
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class KnowledgeGraphNodeSchema(BaseModel):
    id: str
    type: Literal["document", "page", "passage"]
    label: str
    document_id: str
    document_name: str
    page: int | None = Field(default=None, ge=1)
    chunk_id: str | None = None
    citation: str | None = None
    text: str | None = None


class KnowledgeGraphEdgeSchema(BaseModel):
    source: str
    target: str
    relation: Literal["contains_page", "contains_passage"]


class KnowledgeGraphSchema(BaseModel):
    nodes: list[KnowledgeGraphNodeSchema]
    edges: list[KnowledgeGraphEdgeSchema]
    truncated: bool
    scope_note: str


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=100)

    @field_validator("question")
    @classmethod
    def require_non_whitespace_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("question must contain non-whitespace text")
        return normalized


class QuestionResponse(BaseModel):
    status: str
    answer: str | None
    citations: list[str]
    supporting_passages: list[SourceSchema]
    abstained: bool
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    conflict_check_status: str = "not_requested"
    conversation_id: str | None = None
    retrieval_query: str | None = None
    error: ErrorSchema | None = None


class DocumentSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    content_hash: str
    source_url: str | None
    effective_date: date | None
    document_type: str
    upload_status: str
    ingestion_status: str
    size_bytes: int
    page_count: int
    chunk_count: int
    embedding_model: str | None
    embedding_dimension: int | None
    last_error_code: str | None
    created_at: str
    updated_at: str


class UploadItemSchema(BaseModel):
    filename: str
    status: str
    duplicate: bool = False
    document: DocumentSchema | None = None
    error: ErrorSchema | None = None


class UploadBatchResponse(BaseModel):
    items: list[UploadItemSchema]


class ConversationCreateResponse(BaseModel):
    conversation_id: str


class ConversationMessageSchema(BaseModel):
    id: int
    conversation_id: str
    role: str
    content: str
    created_at: str
    original_question: str | None = None
    retrieval_query: str | None = None
    answer_status: str | None = None
    sources: list[dict[str, Any]] = Field(default_factory=list)


class ConversationHistoryResponse(BaseModel):
    conversation_id: str
    messages: list[ConversationMessageSchema]


class EvaluationJobRequest(BaseModel):
    split: str = Field(pattern="^(development|validation|test)$")
    force: bool = False


class EvaluationJobSchema(BaseModel):
    id: str
    status: str
    split: str
    result: dict[str, Any] | None = None
    error: ErrorSchema | None = None
