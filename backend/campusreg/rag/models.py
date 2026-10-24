"""Typed data structures exchanged by the shared RAG core."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class DocumentPage:
    """Text extracted from one physical, one-based PDF page."""

    page: int
    text: str
    source_name: str
    document_id: str = ""

    @property
    def page_number(self) -> int:
        """Physical page number in the original PDF (not a chunk position)."""

        return self.page


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A word chunk retaining stable source identity and physical PDF page."""

    page: int
    text: str
    source_name: str
    document_id: str = ""
    chunk_id: str = ""

    @property
    def page_number(self) -> int:
        return self.page


@dataclass(frozen=True, slots=True)
class RetrievedSource:
    """Evidence passage with a method-specific retrieval score and diagnostics."""

    page: int
    text: str
    source_name: str
    score: float
    document_id: str = ""
    chunk_id: str = ""
    retrieval_method: str = "dense"
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def page_number(self) -> int:
        return self.page

    @property
    def citation(self) -> str:
        if self.document_id:
            return f"[{self.source_name}, document {self.document_id}, p. {self.page}]"
        return f"[{self.source_name}, p. {self.page}]"


@dataclass(frozen=True, slots=True)
class KnowledgeBase:
    """Dense vectors aligned by row with immutable source chunks."""

    name: str
    chunks: tuple[TextChunk, ...]
    vectors: np.ndarray


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"
    GENERATION_ERROR = "generation_error"
    INVALID_RESPONSE = "invalid_response"


@dataclass(frozen=True, slots=True)
class ErrorInfo:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class AnswerResult:
    """A generated/abstained/error result with the actual retrieved evidence."""

    status: AnswerStatus
    answer: str | None
    sources: tuple[RetrievedSource, ...]
    error: ErrorInfo | None = None
    gate_decision: Mapping[str, Any] | None = None
    retrieval_query: str | None = None

    def __post_init__(self) -> None:
        if self.status in (AnswerStatus.GENERATION_ERROR, AnswerStatus.INVALID_RESPONSE) and self.answer is not None:
            raise ValueError("failed responses cannot contain a normal answer")
        if self.status is AnswerStatus.ANSWERED and not self.answer:
            raise ValueError("answered results must contain answer text")
        if self.status in (AnswerStatus.GENERATION_ERROR, AnswerStatus.INVALID_RESPONSE) and self.error is None:
            raise ValueError("failed responses must include structured error information")


@dataclass(frozen=True, slots=True)
class KnowledgeBaseSummary:
    name: str
    page_count: int
    chunk_count: int
