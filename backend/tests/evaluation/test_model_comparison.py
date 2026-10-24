"""Synthetic fixed-evidence model-comparison smoke and cache tests."""

from __future__ import annotations

import csv
import json

from campusreg.config import RAGSettings
from campusreg.evaluation.model_comparison import (
    MODEL_IDS,
    run_fixed_evidence_comparison,
)
from campusreg.evaluation.models import EvaluationQuestion
from campusreg.rag.errors import GenerationError
from campusreg.rag.models import RetrievedSource


def _question(qid: str = "synthetic-model-q") -> EvaluationQuestion:
    return EvaluationQuestion(
        id=qid,
        question="SYNTHETIC TEST FIXTURE ask a sample question",
        category="direct_factual",
        expected_answer="SYNTHETIC TEST FIXTURE expected answer",
        answerable=True,
        expected_document_ids=("synthetic-doc",),
        expected_pages=(),
        expected_conflict=False,
        split="development",
    )


def _source() -> RetrievedSource:
    return RetrievedSource(
        page=2,
        text="SYNTHETIC TEST FIXTURE same fixed passage",
        source_name="SYNTHETIC TEST FIXTURE doc.pdf",
        score=0.6,
        document_id="synthetic-doc",
        chunk_id="synthetic-chunk",
    )


class MockProvider:
    def __init__(self, fail_model: str | None = None) -> None:
        self.calls = []
        self.fail_model = fail_model

    def generate(self, prompt: str, *, model: str, system_instruction: str, temperature: float) -> str:
        self.calls.append((prompt, model, system_instruction, temperature))
        if model == self.fail_model:
            raise GenerationError("synthetic failure")
        return "SYNTHETIC TEST FIXTURE response [SYNTHETIC TEST FIXTURE doc.pdf, document synthetic-doc, p. 2]"


def _config():
    return {"corpus_revision": "synthetic-corpus", "retrieval_revision": "synthetic-v1"}


def test_models_receive_identical_prompt_evidence_and_generation_settings(tmp_path) -> None:
    provider = MockProvider()
    question = _question()
    result = run_fixed_evidence_comparison(
        [question], {question.id: [_source()]}, provider=provider,
        settings=RAGSettings.baseline(), output_dir=tmp_path, run_config=_config(),
    )

    assert result["complete"] is True
    assert [call[1] for call in provider.calls] == list(MODEL_IDS)
    assert provider.calls[0][0] == provider.calls[1][0]
    assert provider.calls[0][2:] == provider.calls[1][2:]
    assert "SYNTHETIC TEST FIXTURE same fixed passage" in provider.calls[0][0]
    assert len(list(csv.DictReader((tmp_path / "model_comparison_side_by_side.csv").open()))) == 1
    outputs = [json.loads(line) for line in (tmp_path / "model_comparison_predictions.jsonl").read_text().splitlines()]
    assert outputs[0]["fixed_evidence_fingerprint"] == outputs[1]["fixed_evidence_fingerprint"]
    assert all(row["timestamp"] and row["model_id"] in MODEL_IDS for row in outputs)


def test_failures_are_recorded_without_substitution_and_make_run_incomplete(tmp_path) -> None:
    provider = MockProvider(fail_model=MODEL_IDS[1])
    question = _question()
    result = run_fixed_evidence_comparison(
        [question], {question.id: [_source()]}, provider=provider,
        settings=RAGSettings.baseline(), output_dir=tmp_path, run_config=_config(),
    )
    assert result["complete"] is False
    assert [call[1] for call in provider.calls] == list(MODEL_IDS)
    outputs = [json.loads(line) for line in (tmp_path / "model_comparison_predictions.jsonl").read_text().splitlines()]
    failed = next(row for row in outputs if row["model_id"] == MODEL_IDS[1])
    assert failed["status"] == "api_error"
    assert failed["answer"] is None
    assert failed["model_id"] != MODEL_IDS[0]


def test_completed_calls_are_cached(tmp_path) -> None:
    q = _question()
    provider = MockProvider()
    inputs = dict(
        questions=[q], evidence={q.id: [_source()]}, provider=provider,
        settings=RAGSettings.baseline(), output_dir=tmp_path, run_config=_config(),
    )
    run_fixed_evidence_comparison(**inputs)
    first_calls = len(provider.calls)
    second = run_fixed_evidence_comparison(**inputs)
    assert len(provider.calls) == first_calls == 2
    assert second["prediction_count"] == 2
