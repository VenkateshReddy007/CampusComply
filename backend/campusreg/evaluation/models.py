"""Typed records used by the reproducible evaluation runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

EVALUATION_CATEGORIES = (
    "direct_factual",
    "multiple_passages",
    "multiple_documents",
    "exceptions_or_effective_dates",
    "unanswerable",
    "follow_up",
    "potential_conflict",
)
EVALUATION_SPLITS = ("development", "validation", "test")


@dataclass(frozen=True, slots=True)
class ExpectedPage:
    document_id: str
    page_number: int


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class EvaluationQuestion:
    id: str
    question: str
    category: str
    expected_answer: str | None
    answerable: bool
    expected_document_ids: tuple[str, ...]
    expected_pages: tuple[ExpectedPage, ...]
    expected_conflict: bool
    split: str
    conversation_context: tuple[ConversationTurn, ...] = ()
    expected_claims: tuple[dict[str, Any], ...] = ()

    @property
    def query_text(self) -> str:
        if not self.conversation_context:
            return self.question
        history = "\n".join(
            f"{turn.role.title()}: {turn.content}" for turn in self.conversation_context
        )
        return f"Conversation context:\n{history}\n\nCurrent question: {self.question}"


@dataclass(slots=True)
class EvaluationPrediction:
    """Raw model output and separately reviewable evaluation labels."""

    question_id: str
    question: str
    category: str
    split: str
    cache_key: str
    config_fingerprint: str
    status: str
    answer: str | None
    retrieved_sources: list[dict[str, Any]]
    latency_ms: float
    request_failed: bool
    error_code: str | None
    citation_exists: bool | None
    citation_valid: bool | None
    conflict_detected: bool | None
    conflict_detection_failed: bool
    human_labels: dict[str, Any] = field(default_factory=dict)
    evidence_gate: dict[str, Any] | None = None
    retrieval_query: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "category": self.category,
            "split": self.split,
            "cache_key": self.cache_key,
            "config_fingerprint": self.config_fingerprint,
            "status": self.status,
            "answer": self.answer,
            "retrieved_sources": self.retrieved_sources,
            "latency_ms": self.latency_ms,
            "request_failed": self.request_failed,
            "error_code": self.error_code,
            "citation_exists": self.citation_exists,
            "citation_valid": self.citation_valid,
            "conflict_detected": self.conflict_detected,
            "conflict_detection_failed": self.conflict_detection_failed,
            "human_labels": self.human_labels,
            "evidence_gate": self.evidence_gate,
            "retrieval_query": self.retrieval_query,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvaluationPrediction":
        required = {
            "question_id",
            "question",
            "category",
            "split",
            "cache_key",
            "config_fingerprint",
            "status",
            "answer",
            "retrieved_sources",
            "latency_ms",
            "request_failed",
        }
        if not required.issubset(value):
            missing = ", ".join(sorted(required - set(value)))
            raise ValueError(f"cached prediction is missing fields: {missing}")
        return cls(
            question_id=value["question_id"],
            question=value["question"],
            category=value["category"],
            split=value["split"],
            cache_key=value["cache_key"],
            config_fingerprint=value["config_fingerprint"],
            status=value["status"],
            answer=value["answer"],
            retrieved_sources=value["retrieved_sources"],
            latency_ms=float(value["latency_ms"]),
            request_failed=bool(value["request_failed"]),
            error_code=value.get("error_code"),
            citation_exists=value.get("citation_exists"),
            citation_valid=value.get("citation_valid"),
            conflict_detected=value.get("conflict_detected"),
            conflict_detection_failed=bool(value.get("conflict_detection_failed", False)),
            human_labels=value.get("human_labels") or {},
            evidence_gate=value.get("evidence_gate"),
            retrieval_query=value.get("retrieval_query"),
        )
