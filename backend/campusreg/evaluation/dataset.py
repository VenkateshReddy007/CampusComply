"""Load only complete, human-reviewed evaluation questions from JSONL."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from campusreg.evaluation.models import (
    EVALUATION_CATEGORIES,
    EVALUATION_SPLITS,
    ConversationTurn,
    EvaluationQuestion,
    ExpectedPage,
)


class EvaluationDatasetError(ValueError):
    """Raised for malformed or unreviewed records in a selected dataset."""


def _require_string(value: Any, field: str, *, line: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationDatasetError(f"line {line}: {field} must be non-empty text")
    return value


def _page(value: Any, *, field: str, line: int) -> ExpectedPage:
    if not isinstance(value, dict):
        raise EvaluationDatasetError(f"line {line}: {field} items must be objects")
    document_id = _require_string(value.get("document_id"), f"{field}.document_id", line=line)
    page_number = value.get("page_number")
    if isinstance(page_number, bool) or not isinstance(page_number, int) or page_number < 1:
        raise EvaluationDatasetError(f"line {line}: {field}.page_number must be a positive integer")
    return ExpectedPage(document_id=document_id, page_number=page_number)


def _ready_record(value: dict[str, Any], *, line: int) -> EvaluationQuestion:
    required_fields = {
        "id", "question", "category", "expected_answer", "answerable",
        "expected_document_ids", "expected_pages", "expected_conflict", "split",
        "conversation_context", "expected_claims", "human_review",
    }
    missing_fields = required_fields - value.keys()
    if missing_fields:
        raise EvaluationDatasetError(
            f"line {line}: ready record is missing fields: {', '.join(sorted(missing_fields))}"
        )
    question_id = _require_string(value.get("id"), "id", line=line)
    question = _require_string(value.get("question"), "question", line=line)
    category = value.get("category")
    if not isinstance(category, str) or category not in EVALUATION_CATEGORIES:
        raise EvaluationDatasetError(f"line {line}: unsupported category {category!r}")
    split = value.get("split")
    if not isinstance(split, str) or split not in EVALUATION_SPLITS:
        raise EvaluationDatasetError(f"line {line}: split must be development, validation, or test")
    answerable = value.get("answerable")
    if not isinstance(answerable, bool):
        raise EvaluationDatasetError(f"line {line}: ready records require a boolean answerable label")
    expected_answer = value.get("expected_answer")
    if answerable:
        expected_answer = _require_string(expected_answer, "expected_answer", line=line)
    elif expected_answer is not None:
        raise EvaluationDatasetError(f"line {line}: unanswerable records must not contain an expected answer")

    expected_document_ids = value.get("expected_document_ids")
    if not isinstance(expected_document_ids, list) or any(
        not isinstance(item, str) or not item.strip() for item in expected_document_ids
    ):
        raise EvaluationDatasetError(f"line {line}: expected_document_ids must be an array of IDs")
    if len(set(expected_document_ids)) != len(expected_document_ids):
        raise EvaluationDatasetError(f"line {line}: expected_document_ids must be unique")
    expected_pages_value = value.get("expected_pages")
    if not isinstance(expected_pages_value, list):
        raise EvaluationDatasetError(f"line {line}: expected_pages must be an array")
    expected_pages = tuple(
        _page(item, field="expected_pages", line=line) for item in expected_pages_value
    )
    if len(set(expected_pages)) != len(expected_pages):
        raise EvaluationDatasetError(f"line {line}: expected_pages must be unique")
    if answerable:
        if not expected_document_ids or not expected_pages:
            raise EvaluationDatasetError(
                f"line {line}: answerable records require expected document IDs and PDF pages"
            )
        page_document_ids = {page.document_id for page in expected_pages}
        if page_document_ids != set(expected_document_ids):
            raise EvaluationDatasetError(
                f"line {line}: expected_document_ids must match the documents referenced by expected_pages"
            )
    elif expected_document_ids or expected_pages:
        raise EvaluationDatasetError(f"line {line}: unanswerable records must have empty expected evidence")
    if category == "unanswerable" and answerable:
        raise EvaluationDatasetError(f"line {line}: unanswerable category requires answerable=false")
    if category == "multiple_passages" and len(expected_pages) < 2:
        raise EvaluationDatasetError(f"line {line}: multiple_passages requires at least two expected pages")
    if category == "multiple_documents" and len(expected_document_ids) < 2:
        raise EvaluationDatasetError(f"line {line}: multiple_documents requires at least two expected documents")

    expected_conflict = value.get("expected_conflict")
    if not isinstance(expected_conflict, bool):
        raise EvaluationDatasetError(f"line {line}: ready records require a boolean expected_conflict label")
    review = value.get("human_review")
    if not isinstance(review, dict) or review.get("status") != "reviewed":
        raise EvaluationDatasetError(f"line {line}: ready records require completed human review")
    if answerable and review.get("source_documents_verified") is not True:
        raise EvaluationDatasetError(f"line {line}: answerable records require verified official sources")
    _require_string(review.get("reviewer"), "human_review.reviewer", line=line)
    reviewed_at = _require_string(review.get("reviewed_at"), "human_review.reviewed_at", line=line)
    if "notes" not in review:
        raise EvaluationDatasetError(f"line {line}: human_review.notes is required")
    try:
        parsed_reviewed_at = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvaluationDatasetError(f"line {line}: human_review.reviewed_at must be an ISO date-time") from exc
    if parsed_reviewed_at.tzinfo is None:
        raise EvaluationDatasetError(f"line {line}: human_review.reviewed_at must include a timezone")

    context_value = value.get("conversation_context", [])
    if not isinstance(context_value, list):
        raise EvaluationDatasetError(f"line {line}: conversation_context must be an array")
    context: list[ConversationTurn] = []
    for turn in context_value:
        if not isinstance(turn, dict) or turn.get("role") not in {"user", "assistant"}:
            raise EvaluationDatasetError(f"line {line}: conversation turns need a user or assistant role")
        context.append(
            ConversationTurn(
                role=turn["role"],
                content=_require_string(turn.get("content"), "conversation_context.content", line=line),
            )
        )
    if category == "follow_up" and not context:
        raise EvaluationDatasetError(f"line {line}: follow_up requires conversation_context")
    claims_value = value.get("expected_claims", [])
    if not isinstance(claims_value, list):
        raise EvaluationDatasetError(f"line {line}: expected_claims must be an array")
    claims: list[dict[str, Any]] = []
    seen_claim_ids: set[str] = set()
    for claim in claims_value:
        if not isinstance(claim, dict):
            raise EvaluationDatasetError(f"line {line}: expected_claims items must be objects")
        claim_id = _require_string(claim.get("claim_id"), "expected_claims.claim_id", line=line)
        if claim_id in seen_claim_ids:
            raise EvaluationDatasetError(f"line {line}: expected_claims.claim_id values must be unique")
        seen_claim_ids.add(claim_id)
        _require_string(claim.get("claim"), "expected_claims.claim", line=line)
        supporting = claim.get("supporting_pages")
        if not isinstance(supporting, list) or not supporting:
            raise EvaluationDatasetError(f"line {line}: expected claims require supporting PDF pages")
        supporting_pages = tuple(
            _page(item, field="expected_claims.supporting_pages", line=line)
            for item in supporting
        )
        if len(set(supporting_pages)) != len(supporting_pages):
            raise EvaluationDatasetError(f"line {line}: claim supporting_pages must be unique")
        if not set(supporting_pages).issubset(set(expected_pages)):
            raise EvaluationDatasetError(
                f"line {line}: expected claim support must reference pages in expected_pages"
            )
        claims.append(
            {
                "claim_id": claim["claim_id"],
                "claim": claim["claim"],
                "supporting_pages": [
                    {"document_id": page.document_id, "page_number": page.page_number}
                    for page in supporting_pages
                ],
            }
        )
    if expected_conflict and len(claims) < 2:
        raise EvaluationDatasetError(
            f"line {line}: a reviewed conflict requires two supporting expected claims"
        )
    return EvaluationQuestion(
        id=question_id,
        question=question,
        category=category,
        expected_answer=expected_answer,
        answerable=answerable,
        expected_document_ids=tuple(expected_document_ids),
        expected_pages=expected_pages,
        expected_conflict=expected_conflict,
        split=split,
        conversation_context=tuple(context),
        expected_claims=tuple(claims),
    )


def load_dataset(path: str | Path, *, split: str | None = None) -> list[EvaluationQuestion]:
    """Read ready records; template/draft records are deliberately skipped."""

    if split is not None and split not in EVALUATION_SPLITS:
        raise ValueError(f"split must be one of: {', '.join(EVALUATION_SPLITS)}")
    questions: list[EvaluationQuestion] = []
    seen_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, raw in enumerate(source, start=1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise EvaluationDatasetError(f"line {line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise EvaluationDatasetError(f"line {line_number}: each JSONL row must be an object")
            record_id = _require_string(value.get("id"), "id", line=line_number)
            if record_id in seen_ids:
                raise EvaluationDatasetError(f"line {line_number}: duplicate question id {record_id!r}")
            seen_ids.add(record_id)
            status = value.get("record_status")
            if not isinstance(status, str):
                raise EvaluationDatasetError(
                    f"line {line_number}: record_status must be ready, draft, or template"
                )
            if status in {"template", "draft"}:
                continue
            if status != "ready":
                raise EvaluationDatasetError(f"line {line_number}: record_status must be ready, draft, or template")
            question = _ready_record(value, line=line_number)
            if split is None or question.split == split:
                questions.append(question)
    return questions


def validate_challenge_dataset(
    path: str | Path,
    *,
    minimum_questions: int = 15,
    require_all_splits: bool = True,
) -> dict[str, Any]:
    """Validate final challenge-set size and split coverage without running models."""

    if minimum_questions < 15:
        raise ValueError("minimum_questions cannot be lower than the challenge minimum of 15")
    questions = load_dataset(path)
    if len(questions) < minimum_questions:
        raise EvaluationDatasetError(
            f"dataset has {len(questions)} reviewed questions; at least {minimum_questions} are required"
        )
    split_counts = {
        split: sum(question.split == split for question in questions)
        for split in EVALUATION_SPLITS
    }
    if require_all_splits:
        missing = [split for split, count in split_counts.items() if count == 0]
        if missing:
            raise EvaluationDatasetError(
                f"dataset must include development, validation, and held-out test questions; empty: {', '.join(missing)}"
            )
    category_counts = {
        category: sum(question.category == category for question in questions)
        for category in EVALUATION_CATEGORIES
    }
    return {
        "question_count": len(questions),
        "split_counts": split_counts,
        "category_counts": category_counts,
    }
