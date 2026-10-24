"""Metric calculations that keep retrieval, answer, and citation labels separate."""

from __future__ import annotations

from math import ceil
from datetime import datetime
from statistics import mean
from typing import Any, Sequence

from campusreg.evaluation.models import EvaluationPrediction, EvaluationQuestion


def _prediction_dict(value: EvaluationPrediction | dict[str, Any]) -> dict[str, Any]:
    return value.to_dict() if isinstance(value, EvaluationPrediction) else value


def _summary(values: Sequence[float | int | bool | None]) -> dict[str, float | int | None]:
    selected = [float(value) for value in values if value is not None]
    return {"value": mean(selected) if selected else None, "n": len(selected)}


def _count(value: int, denominator: int) -> dict[str, int | float | None]:
    return {"value": value, "n": denominator}


def _reviewed_label(prediction: dict[str, Any], name: str) -> bool | None:
    """Use nullable labels only when a reviewer and timestamp are recorded."""

    labels = prediction.get("human_labels") or {}
    value = labels.get(name)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"human_labels.{name} must be boolean or null")
    reviewer = labels.get("reviewer")
    reviewed_at = labels.get("reviewed_at")
    if not isinstance(reviewer, str) or not reviewer.strip() or not isinstance(reviewed_at, str):
        raise ValueError(f"human_labels.{name} requires a reviewer and reviewed_at timestamp")
    try:
        parsed_reviewed_at = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"human_labels.{name} has an invalid reviewed_at timestamp") from exc
    if parsed_reviewed_at.tzinfo is None:
        raise ValueError(f"human_labels.{name} reviewed_at timestamp must include a timezone")
    return value


def _rank_metrics(question: EvaluationQuestion, prediction: dict[str, Any], k: int) -> tuple[float, float] | None:
    if not question.answerable or not question.expected_pages and not question.expected_document_ids:
        return None
    expected_pages = {
        (page.document_id, page.page_number) for page in question.expected_pages
    }
    expected_documents = set(question.expected_document_ids)
    sources = prediction.get("retrieved_sources") or []
    top_sources = sources[:k]

    def relevant(source: dict[str, Any]) -> bool:
        document_id = source.get("document_id")
        if expected_pages:
            return (document_id, source.get("page_number", source.get("page"))) in expected_pages
        return document_id in expected_documents

    hits = {
        (source.get("document_id"), source.get("page_number", source.get("page")))
        if expected_pages
        else source.get("document_id")
        for source in top_sources
        if relevant(source)
    }
    denominator = len(expected_pages) if expected_pages else len(expected_documents)
    recall = len(hits) / denominator if denominator else 0.0
    reciprocal_rank = next(
        (1.0 / rank for rank, source in enumerate(top_sources, start=1) if relevant(source)),
        0.0,
    )
    return recall, reciprocal_rank


def _confusion(expected: Sequence[bool], predicted: Sequence[bool]) -> dict[str, Any]:
    tp = sum(actual and guess for actual, guess in zip(expected, predicted, strict=True))
    tn = sum(not actual and not guess for actual, guess in zip(expected, predicted, strict=True))
    fp = sum(not actual and guess for actual, guess in zip(expected, predicted, strict=True))
    fn = sum(actual and not guess for actual, guess in zip(expected, predicted, strict=True))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    total = tp + tn + fp + fn
    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "precision": {"value": precision, "n": tp + fp},
        "recall": {"value": recall, "n": tp + fn},
        "f1": {"value": f1, "n": total},
        "accuracy": {"value": (tp + tn) / total if total else None, "n": total},
    }


def _nearest_rank(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _compute_subset(
    questions: Sequence[EvaluationQuestion],
    prediction_rows: dict[str, dict[str, Any]],
    retrieval_k: int,
) -> dict[str, Any]:
    selected = [(question, prediction_rows[question.id]) for question in questions if question.id in prediction_rows]
    ranks = [
        rank
        for question, prediction in selected
        if (rank := _rank_metrics(question, prediction, retrieval_k)) is not None
    ]
    answer_correct = [_reviewed_label(prediction, "answer_correct") for _, prediction in selected]
    citation_support = [_reviewed_label(prediction, "citation_support") for _, prediction in selected]
    evidence_support = [_reviewed_label(prediction, "evidence_support") for _, prediction in selected]
    human_refusal = [_reviewed_label(prediction, "appropriate_refusal") for _, prediction in selected]
    answered = [
        prediction
        for _, prediction in selected
        if prediction.get("status") == "answered"
    ]
    refusal_match: list[bool | None] = []
    expected_conflicts: list[bool] = []
    predicted_conflicts: list[bool] = []
    for question, prediction in selected:
        status = prediction.get("status")
        if status in {"answered", "abstained"}:
            refusal_match.append((status == "abstained") == (not question.answerable))
        else:
            refusal_match.append(None)
        conflict = prediction.get("conflict_detected")
        if isinstance(conflict, bool):
            expected_conflicts.append(question.expected_conflict)
            predicted_conflicts.append(conflict)

    latencies = [float(prediction.get("latency_ms", 0.0)) for _, prediction in selected]
    failure_count = sum(bool(prediction.get("request_failed")) for _, prediction in selected)
    conflict_failures = sum(bool(prediction.get("conflict_detection_failed")) for _, prediction in selected)
    count = len(selected)
    return {
        "question_count": count,
        "retrieval": {
            f"recall_at_{retrieval_k}": _summary([rank[0] for rank in ranks]),
            f"mrr_at_{retrieval_k}": _summary([rank[1] for rank in ranks]),
        },
        "answer_quality": {
            "answer_correctness_human": _summary(answer_correct),
            "evidence_support_human": _summary(evidence_support),
        },
        "citations": {
            "citation_existence": _summary([prediction.get("citation_exists") for prediction in answered]),
            "citation_validity": _summary([prediction.get("citation_valid") for prediction in answered]),
            "citation_support_human": _summary(citation_support),
        },
        "refusal": {
            "appropriate_refusal_human": _summary(human_refusal),
            "abstention_behavior_match": _summary(refusal_match),
        },
        "potential_conflict_detection": _confusion(expected_conflicts, predicted_conflicts),
        "operations": {
            "failed_requests": _count(failure_count, count),
            "failed_request_rate": {"value": failure_count / count if count else None, "n": count},
            "conflict_detection_failures": _count(conflict_failures, count),
            "mean_latency_ms": _summary(latencies),
            "p50_latency_ms": {"value": _nearest_rank(latencies, 0.50), "n": len(latencies)},
            "p95_latency_ms": {"value": _nearest_rank(latencies, 0.95), "n": len(latencies)},
        },
    }


def compute_metrics(
    questions: Sequence[EvaluationQuestion],
    predictions: Sequence[EvaluationPrediction | dict[str, Any]],
    *,
    retrieval_k: int = 5,
) -> dict[str, Any]:
    """Compute auditable metrics without using an LLM judge as ground truth."""

    if retrieval_k <= 0:
        raise ValueError("retrieval_k must be positive")
    prediction_rows = {_prediction_dict(item)["question_id"]: _prediction_dict(item) for item in predictions}
    overall = _compute_subset(questions, prediction_rows, retrieval_k)
    by_category = {
        category: _compute_subset(
            [question for question in questions if question.category == category],
            prediction_rows,
            retrieval_k,
        )
        for category in sorted({question.category for question in questions})
    }
    return {"retrieval_k": retrieval_k, "overall": overall, "by_category": by_category}
