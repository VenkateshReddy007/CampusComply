"""Synthetic, mocked tests for paired no-RAG/RAG evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from campusreg.config import RAGSettings
from campusreg.evaluation.models import EvaluationQuestion
from campusreg.evaluation.paired import run_paired_evaluation, run_smoke_test
from campusreg.rag.errors import GenerationError
from campusreg.rag.models import AnswerResult, AnswerStatus, ErrorInfo, RetrievedSource


def _question() -> EvaluationQuestion:
    return EvaluationQuestion(
        id="SYNTHETIC-PAIRED-1",
        question="SYNTHETIC TEST FIXTURE: what does the sample regulation say?",
        category="direct_factual",
        expected_answer="SYNTHETIC TEST FIXTURE expected answer",
        answerable=True,
        expected_document_ids=("synthetic-doc",),
        expected_pages=(),
        expected_conflict=False,
        split="development",
    )


class MockGenerator:
    def __init__(self, result: str = "SYNTHETIC TEST FIXTURE unverified response") -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append((prompt, kwargs))
        return self.result


def _source() -> RetrievedSource:
    return RetrievedSource(
        page=2,
        text="SYNTHETIC TEST FIXTURE passage visible to RAG only",
        source_name="SYNTHETIC TEST FIXTURE policy.pdf",
        score=0.8,
        document_id="synthetic-doc",
        chunk_id="synthetic-chunk",
    )


def _run_config() -> dict[str, Any]:
    return {
        "gemma_model": "synthetic-gemma",
        "temperature": 0.2,
        "corpus_revision": "synthetic-corpus-v1",
        "prompt_revision": "synthetic-prompt-v1",
        "api_key": "PLACEHOLDER_ONLY",
    }


def test_paired_runner_keeps_no_rag_question_only_and_records_same_config(tmp_path: Path) -> None:
    generator = MockGenerator("SYNTHETIC TEST FIXTURE general model response [invented source]")
    source = _source()
    rag_calls: list[str] = []

    def rag_answerer(question: str) -> AnswerResult:
        rag_calls.append(question)
        return AnswerResult(
            status=AnswerStatus.ANSWERED,
            answer=f"SYNTHETIC TEST FIXTURE evidence response {source.citation}",
            sources=(source,),
        )

    settings = RAGSettings(gemma_model="synthetic-gemma", temperature=0.2)
    result = run_paired_evaluation(
        [_question()],
        rag_answerer=rag_answerer,
        generator=generator,
        settings=settings,
        output_dir=tmp_path,
        run_config=_run_config(),
    )

    assert generator.calls[0][0] == _question().question
    assert "passage visible to RAG" not in generator.calls[0][0]
    assert generator.calls[0][1]["model"] == settings.gemma_model
    assert generator.calls[0][1]["temperature"] == settings.temperature
    assert rag_calls == [_question().question]
    records = {row["condition"]: row for row in result["records"]}
    assert records["no_rag"]["retrieved_sources"] == []
    assert records["no_rag"]["citation_valid"] is False
    assert records["rag"]["retrieved_sources"][0]["document_id"] == "synthetic-doc"
    assert records["rag"]["retrieved_sources"][0]["page_number"] == 2
    assert records["rag"]["citation_valid"] is True
    assert records["rag"]["human_labels"]["evidence_support"] is None
    assert "PLACEHOLDER_ONLY" not in result["predictions_path"].read_text(encoding="utf-8")
    with result["side_by_side_csv_path"].open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 1
    assert rows[0]["question_id"] == _question().id


def test_paired_runner_reuses_completed_calls_and_reports_reviewed_differences(tmp_path: Path) -> None:
    generator = MockGenerator()
    rag_calls = 0

    def rag_answerer(_question: str) -> AnswerResult:
        nonlocal rag_calls
        rag_calls += 1
        source = _source()
        return AnswerResult(AnswerStatus.ANSWERED, f"SYNTHETIC TEST FIXTURE {source.citation}", (source,))

    args = {
        "questions": [_question()],
        "rag_answerer": rag_answerer,
        "generator": generator,
        "settings": RAGSettings(gemma_model="synthetic-gemma"),
        "output_dir": tmp_path,
        "run_config": _run_config(),
    }
    first = run_paired_evaluation(**args)
    assert len(generator.calls) == 1
    assert rag_calls == 1

    raw_path = first["predictions_path"]
    records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    reviewed_at = "2026-10-24T10:00:00Z"
    for record in records:
        record["human_labels"].update(
            {
                "answer_correct": record["condition"] == "rag",
                "unsupported_claims_present": record["condition"] == "no_rag",
                "reviewer": "synthetic-reviewer",
                "reviewed_at": reviewed_at,
                "comparison_tag": "rag_helped" if record["condition"] == "rag" else None,
            }
        )
    raw_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    second = run_paired_evaluation(**args)
    assert len(generator.calls) == 1
    assert rag_calls == 1
    report = second["report_path"].read_text(encoding="utf-8")
    assert "Answer-correctness difference (RAG minus no-RAG): 1.0" in report
    assert "Unsupported-claim presence difference (RAG minus no-RAG): -1.0" in report
    assert "SYNTHETIC-PAIRED-1" in report


def test_api_and_incomplete_response_failures_are_structured(tmp_path: Path) -> None:
    class EmptyGenerator(MockGenerator):
        def generate(self, prompt: str, **kwargs: Any) -> str:
            super().generate(prompt, **kwargs)
            raise GenerationError("Gemma returned no answer text.", code="empty_generation")

    result = run_paired_evaluation(
        [_question()],
        rag_answerer=lambda _: AnswerResult(
            AnswerStatus.GENERATION_ERROR,
            None,
            (_source(),),
            ErrorInfo("generation_failed", "Gemma generation request failed."),
        ),
        generator=EmptyGenerator(),
        settings=RAGSettings(gemma_model="synthetic-gemma"),
        output_dir=tmp_path,
        run_config=_run_config(),
    )
    statuses = {row["condition"]: row["status"] for row in result["records"]}
    assert statuses == {"no_rag": "incomplete_response", "rag": "generation_error"}
    assert all(row["answer"] is None for row in result["records"])


def test_synthetic_smoke_test_passes_without_external_calls() -> None:
    assert run_smoke_test() == {"status": "passed", "calls": 1, "conditions": 2}
