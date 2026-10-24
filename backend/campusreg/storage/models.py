"""Persistent document metadata exposed by storage adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any


class UploadStatus(StrEnum):
    STORED = "stored"


class IngestionStatus(StrEnum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    REINDEXING = "reindexing"
    INDEXED = "indexed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    id: str
    filename: str
    content_hash: str
    source_url: str | None
    effective_date: date | None
    document_type: str
    upload_status: UploadStatus
    ingestion_status: IngestionStatus
    size_bytes: int
    page_count: int
    chunk_count: int
    embedding_model: str | None
    embedding_dimension: int | None
    last_error_code: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class IngestionResult:
    document: DocumentRecord
    duplicate: bool


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    id: int
    conversation_id: str
    role: str
    content: str
    created_at: str
    original_question: str | None = None
    retrieval_query: str | None = None
    answer_status: str | None = None
    sources: tuple[dict[str, Any], ...] = field(default_factory=tuple)
