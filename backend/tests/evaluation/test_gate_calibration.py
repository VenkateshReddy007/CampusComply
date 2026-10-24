"""Synthetic-only evidence-gate calibration and held-out tests."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from campusreg.evaluation.gate_calibration import (
    GateCalibrationError,
    calibrate_evidence_gates,
    evaluate_gate_on_test,
)
from campusreg.rag.evidence_gate import EvidenceGatePolicy


REVIEWED_AT = "2026-10-24T10:00:00Z"
SOURCE_CONFIG = {
    "embedding_model": "SYNTHETIC TEST FIXTURE embedding model",
    "chunk_size": 80,
    "chunk_overlap": 30,
    "top_k": 3,
    "retrieval_method": "dense",
    "corpus_revision": "SYNTHETIC TEST FIXTURE corpus v1",
    "dataset_fingerprint": "SYNTHETIC TEST FIXTURE dataset fingerprint",
}


def _record(
    question_id: str,
    *,
    split: str,
    answerable: bool,
    score: float,
    model: str = "SYNTHETIC TEST FIXTURE MiniLM",
    status: str | None = None,
    reviewed: bool = True,
) -> dict:
    config = {**SOURCE_CONFIG, "embedding_model": model}
    if status is None:
        status = "answered" if answerable else "abstained"
    labels = {
        "answer_correct": answerable and score >= 0.8,
        "unsupported_claims_present": status == "answered" and score < 0.8,
        "appropriate_refusal": (not answerable) == (status == "abstained"),
    }
    if reviewed:
        labels.update({"reviewer": "synthetic-reviewer", "reviewed_at": REVIEWED_AT})
    return {
        "condition": "rag",
        "split": split,
        "question_id": question_id,
        "question": f"SYNTHETIC TEST FIXTURE {question_id}",
        "answerable": answerable,
        "status": status,
        "configuration": config,
        "retrieved_sources": (
            [{"score": score, "document_id": "synthetic-doc", "page_number": 1}]
            if score is not None
            else []
        ),
        "human_labels": labels,
    }


def _validation(model: str = "SYNTHETIC TEST FIXTURE MiniLM") -> list[dict]:
    return [
        _record("SYNTHETIC-positive-high", split="validation", answerable=True, score=0.90, model=model),
        _record("SYNTHETIC-positive-low", split="validation", answerable=True, score=0.45, model=model),
        _record("SYNTHETIC-negative-high", split="validation", answerable=False, score=0.88, model=model, status="answered"),
        _record("SYNTHETIC-negative-low", split="validation", answerable=False, score=0.10, model=model),
    ]


def test_thresholds_are_calibrated_per_embedding_configuration(tmp_path: Path) -> None:
    rows = _validation() + _validation("SYNTHETIC TEST FIXTURE BGE")
    result = calibrate_evidence_gates(
        rows,
        output_dir=tmp_path,
        max_false_accept_rate=0.0,
        max_unsupported_answer_rate=0.0,
    )

    assert len(result["policies"]) == 2
    policies = [EvidenceGatePolicy.from_dict(row) for row in result["policies"]]
    assert {policy.embedding_model for policy in policies} == {
        "SYNTHETIC TEST FIXTURE MiniLM",
        "SYNTHETIC TEST FIXTURE BGE",
    }
    assert len({policy.configuration_id for policy in policies}) == 2
    assert all(policy.threshold == pytest.approx(0.9) for policy in policies)
    assert all(policy.validation_metrics["false_accept_rate"] == 0 for policy in policies)
    assert all(policy.validation_metrics["false_refusal_rate"] == 0.5 for policy in policies)
    assert result["held_out_test_used_for_calibration"] is False
    assert result["score_is_probability"] is False
    assert result["metrics_path"].is_file()


def test_heldout_metrics_use_frozen_threshold_without_retuning(tmp_path: Path) -> None:
    calibration = calibrate_evidence_gates(
        _validation(),
        output_dir=tmp_path / "validation",
        max_false_accept_rate=0.0,
        max_unsupported_answer_rate=0.0,
    )
    policy = EvidenceGatePolicy.from_dict(calibration["policies"][0])
    test_rows = [
        _record("SYNTHETIC-test-positive", split="test", answerable=True, score=0.95),
        _record("SYNTHETIC-test-negative-high", split="test", answerable=False, score=0.92, status="answered"),
        _record("SYNTHETIC-test-negative-low", split="test", answerable=False, score=0.20),
    ]

    result = evaluate_gate_on_test([policy], test_rows, output_dir=tmp_path / "test")

    assert len(result["results"]) == 1
    row = result["results"][0]
    assert row["threshold_frozen_from_validation"] == policy.threshold
    assert row["false_accept_rate"] == pytest.approx(0.5)
    assert row["false_refusal_rate"] == 0
    with result["metrics_path"].open("r", encoding="utf-8", newline="") as csv_file:
        assert len(list(csv.DictReader(csv_file))) == 1


def test_calibration_rejects_unreviewed_labels_and_heldout_only_input(tmp_path: Path) -> None:
    unreviewed = _validation()
    unreviewed[0] = _record(
        "SYNTHETIC-positive-high", split="validation", answerable=True, score=0.9, reviewed=False
    )
    with pytest.raises(GateCalibrationError, match="human-review metadata"):
        calibrate_evidence_gates(
            unreviewed,
            output_dir=tmp_path / "bad-labels",
            max_false_accept_rate=0.0,
            max_unsupported_answer_rate=0.0,
        )
    with pytest.raises(GateCalibrationError, match="no validation RAG records"):
        calibrate_evidence_gates(
            [_record("SYNTHETIC-test-only", split="test", answerable=True, score=0.9)],
            output_dir=tmp_path / "test-only",
            max_false_accept_rate=0.0,
            max_unsupported_answer_rate=0.0,
        )


def test_calibration_requires_identical_validation_question_ids_per_config(tmp_path: Path) -> None:
    rows = _validation() + _validation("SYNTHETIC TEST FIXTURE BGE")[:-1]
    with pytest.raises(GateCalibrationError, match="same validation question IDs"):
        calibrate_evidence_gates(
            rows,
            output_dir=tmp_path,
            max_false_accept_rate=0.0,
            max_unsupported_answer_rate=0.0,
        )
