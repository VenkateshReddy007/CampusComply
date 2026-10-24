"""Synthetic-only tests for CampusReg evaluation; no campus-policy labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from campusreg.evaluation import (
    EvaluationDatasetError,
    EvaluationQuestion,
    EvaluationRunner,
    ConversationTurn,
    ExpectedPage,
    compute_metrics,
    fingerprint_config,
    load_dataset,
    validate_challenge_dataset,
)
from campusreg.rag.models import AnswerResult, AnswerStatus, ErrorInfo, RetrievedSource


def question(
    question_id: str,
    *,
    category: str = "direct_factual",
    answerable: bool = True,
    expected_pages: tuple[ExpectedPage, ...] | None = None,
    expected_conflict: bool = False,
    split: str = "development",
    conversation_context: tuple = (),
) -> EvaluationQuestion:
    if expected_pages is None:
        if not answerable:
            pages = ()
        elif expected_conflict:
            pages = (ExpectedPage("synthetic-doc-a", 1), ExpectedPage("synthetic-doc-b", 3))
        else:
            pages = (ExpectedPage("synthetic-doc-a", 2),)
    else:
        pages = expected_pages
    document_ids = tuple(dict.fromkeys(page.document_id for page in pages))
    expected_claims = (
        (
            {
                "claim_id": "synthetic-conflicting-claim-a",
                "claim": "SYNTHETIC TEST FIXTURE first policy passage",
                "supporting_pages": [
                    {"document_id": pages[0].document_id, "page_number": pages[0].page_number}
                ],
            },
            {
                "claim_id": "synthetic-conflicting-claim-b",
                "claim": "SYNTHETIC TEST FIXTURE second policy passage",
                "supporting_pages": [
                    {
                        "document_id": pages[1 if len(pages) > 1 else 0].document_id,
                        "page_number": pages[1 if len(pages) > 1 else 0].page_number,
                    }
                ],
            },
        )
        if expected_conflict
        else ()
    )
    return EvaluationQuestion(
        id=question_id,
        question=f"SYNTHETIC TEST FIXTURE: question {question_id}",
        category=category,
        expected_answer="SYNTHETIC TEST FIXTURE expected response" if answerable else None,
        answerable=answerable,
        expected_document_ids=document_ids,
        expected_pages=pages,
        expected_conflict=expected_conflict,
        split=split,
        conversation_context=conversation_context,
        expected_claims=expected_claims,
    )


def prediction(
    question_id: str,
    *,
    status: str = "answered",
    sources: list[dict] | None = None,
    answer: str | None = "SYNTHETIC TEST FIXTURE answer",
    citation_exists: bool | None = True,
    citation_valid: bool | None = True,
    conflict_detected: bool | None = False,
    labels: dict | None = None,
    latency_ms: float = 12.0,
    request_failed: bool = False,
) -> dict:
    return {
        "question_id": question_id,
        "question": f"SYNTHETIC TEST FIXTURE: question {question_id}",
        "category": "direct_factual",
        "split": "development",
        "cache_key": "synthetic-cache-key",
        "config_fingerprint": "synthetic-config-fingerprint",
        "status": status,
        "answer": answer,
        "retrieved_sources": sources or [],
        "latency_ms": latency_ms,
        "request_failed": request_failed,
        "error_code": None,
        "citation_exists": citation_exists,
        "citation_valid": citation_valid,
        "conflict_detected": conflict_detected,
        "conflict_detection_failed": False,
        "human_labels": labels or {},
    }


def source(document_id: str, page: int, citation: str, text: str = "SYNTHETIC TEST FIXTURE passage") -> dict:
    return {
        "document_id": document_id,
        "document_name": f"{document_id}.pdf",
        "page_number": page,
        "chunk_id": f"synthetic-chunk-{document_id}-{page}",
        "citation": citation,
        "text": text,
        "score": 0.9,
        "retrieval_method": "dense",
        "diagnostics": {"score_kind": "cosine_similarity"},
    }


def test_dataset_loader_requires_review_and_selects_split(tmp_path: Path) -> None:
    ready = {
        "id": "synthetic-ready-1",
        "question": "SYNTHETIC TEST FIXTURE: manually reviewed question",
        "category": "potential_conflict",
        "expected_answer": "SYNTHETIC TEST FIXTURE expected answer",
        "answerable": True,
        "expected_document_ids": ["synthetic-test-pdf"],
        "expected_pages": [{"document_id": "synthetic-test-pdf", "page_number": 2}],
        "expected_conflict": True,
        "split": "validation",
        "record_status": "ready",
        "conversation_context": [],
        "expected_claims": [
            {
                "claim_id": "synthetic-claim-a",
                "claim": "SYNTHETIC TEST FIXTURE first passage on this page",
                "supporting_pages": [{"document_id": "synthetic-test-pdf", "page_number": 2}],
            },
            {
                "claim_id": "synthetic-claim-b",
                "claim": "SYNTHETIC TEST FIXTURE second passage on this page",
                "supporting_pages": [{"document_id": "synthetic-test-pdf", "page_number": 2}],
            },
        ],
        "human_review": {
            "status": "reviewed",
            "source_documents_verified": True,
            "reviewer": "synthetic-reviewer",
            "reviewed_at": "2026-01-01T00:00:00Z",
            "notes": "SYNTHETIC TEST FIXTURE only; these are not campus policy sources.",
        },
    }
    template = {**ready, "id": "synthetic-template", "record_status": "template", "answerable": None}
    path = tmp_path / "questions.jsonl"
    path.write_text(
        "\n".join(json.dumps(item) for item in (template, ready)) + "\n", encoding="utf-8"
    )

    repo_root = Path(__file__).parents[3]
    dataset_schema = json.loads((repo_root / "evaluation" / "dataset.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(dataset_schema)
    dataset_validator = Draft202012Validator(dataset_schema, format_checker=FormatChecker())
    template_record = json.loads(
        (repo_root / "evaluation" / "questions.template.jsonl").read_text(encoding="utf-8")
    )
    assert list(dataset_validator.iter_errors(template_record)) == []
    assert list(dataset_validator.iter_errors(ready)) == []

    loaded = load_dataset(path, split="validation")

    assert len(loaded) == 1
    assert loaded[0].id == "synthetic-ready-1"
    assert loaded[0].expected_pages == (ExpectedPage("synthetic-test-pdf", 2),)
    assert load_dataset(path, split="test") == []

    conflict_missing_claim = {**ready, "expected_claims": ready["expected_claims"][:1]}
    path.write_text(json.dumps(conflict_missing_claim) + "\n", encoding="utf-8")
    with pytest.raises(EvaluationDatasetError, match="two supporting expected claims"):
        load_dataset(path)

    ready["human_review"]["status"] = "pending"
    path.write_text(json.dumps(ready) + "\n", encoding="utf-8")
    with pytest.raises(EvaluationDatasetError, match="completed human review"):
        load_dataset(path)


def test_retrieval_metrics_use_page_identity_and_answer_labels_stay_human() -> None:
    first = question(
        "synthetic-multi-page",
        category="multiple_passages",
        expected_pages=(
            ExpectedPage("synthetic-doc-a", 1),
            ExpectedPage("synthetic-doc-a", 2),
        ),
    )
    second = question(
        "synthetic-ranked-second",
        expected_pages=(ExpectedPage("synthetic-doc-b", 4),),
    )
    no_answer = question(
        "synthetic-unanswerable",
        category="unanswerable",
        answerable=False,
    )
    first_source = source("synthetic-doc-a", 1, "[synthetic-a.pdf, document synthetic-doc-a, p. 1]")
    second_source = source("synthetic-doc-b", 4, "[synthetic-b.pdf, document synthetic-doc-b, p. 4]")
    rows = [
        prediction(
            first.id,
            sources=[
                source("synthetic-doc-x", 9, "[synthetic-x.pdf, document synthetic-doc-x, p. 9]"),
                first_source,
            ],
            labels={
                "answer_correct": True,
                "citation_support": False,
                "evidence_support": True,
                "reviewer": "synthetic-human-reviewer",
                "reviewed_at": "2026-10-01T00:00:00Z",
            },
        ),
        prediction(second.id, sources=[second_source, source("synthetic-doc-x", 1, "[x.pdf, document x, p. 1]")]),
        prediction(
            no_answer.id,
            status="abstained",
            answer="The answer could not be found in the documents.",
            citation_exists=None,
            citation_valid=None,
            conflict_detected=None,
        ),
    ]

    metrics = compute_metrics([first, second, no_answer], rows, retrieval_k=2)

    retrieval = metrics["overall"]["retrieval"]
    assert retrieval["recall_at_2"] == {"value": 0.75, "n": 2}
    assert retrieval["mrr_at_2"] == {"value": 0.75, "n": 2}
    quality = metrics["overall"]["answer_quality"]
    assert quality["answer_correctness_human"] == {"value": 1.0, "n": 1}
    assert quality["evidence_support_human"] == {"value": 1.0, "n": 1}
    citations = metrics["overall"]["citations"]
    assert citations["citation_support_human"] == {"value": 0.0, "n": 1}
    assert metrics["overall"]["refusal"]["abstention_behavior_match"] == {"value": 1.0, "n": 3}
    unreviewed = prediction(first.id, labels={"answer_correct": True})
    with pytest.raises(ValueError, match="requires a reviewer"):
        compute_metrics([first], [unreviewed])


def test_challenge_dataset_validator_enforces_minimum_size_and_split_coverage(tmp_path: Path) -> None:
    rows = []
    for index in range(15):
        split = ("development", "validation", "test")[index // 5]
        rows.append(
            {
                "id": f"synthetic-question-{index:02d}",
                "question": f"SYNTHETIC TEST FIXTURE question {index}",
                "category": "unanswerable",
                "expected_answer": None,
                "answerable": False,
                "expected_document_ids": [],
                "expected_pages": [],
                "expected_conflict": False,
                "split": split,
                "record_status": "ready",
                "conversation_context": [],
                "expected_claims": [],
                "human_review": {
                    "status": "reviewed",
                    "source_documents_verified": None,
                    "reviewer": "synthetic-test-reviewer",
                    "reviewed_at": "2026-10-01T00:00:00Z",
                    "notes": "SYNTHETIC TEST FIXTURE; not a campus policy record.",
                },
            }
        )
    path = tmp_path / "synthetic-final-shape.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    validated = validate_challenge_dataset(path)

    assert validated["question_count"] == 15
    assert validated["split_counts"] == {"development": 5, "validation": 5, "test": 5}
    with pytest.raises(EvaluationDatasetError, match="at least 16"):
        validate_challenge_dataset(path, minimum_questions=16)


def test_runner_saves_raw_jsonl_metrics_json_csv_and_reuses_cached_predictions(tmp_path: Path) -> None:
    direct = question("synthetic-run-1")
    follow_up = question(
        "synthetic-run-follow-up",
        category="follow_up",
        conversation_context=(
            ConversationTurn(role="user", content="SYNTHETIC TEST FIXTURE prior question"),
        ),
    )
    questions = [direct, follow_up]
    calls: list[str] = []

    def answerer(query: str) -> AnswerResult:
        calls.append(query)
        assert "SYNTHETIC TEST FIXTURE" in query
        cited = RetrievedSource(
            page=2,
            text="SYNTHETIC TEST FIXTURE actual retrieved evidence",
            source_name="synthetic-doc-a.pdf",
            score=0.95,
            document_id="synthetic-doc-a",
            chunk_id="synthetic-chunk-a-2",
        )
        return AnswerResult(
            status=AnswerStatus.ANSWERED,
            answer=f"SYNTHETIC TEST FIXTURE response {cited.citation}",
            sources=(cited,),
        )

    output_dir = tmp_path / "results"
    runner = EvaluationRunner()
    config = {
        "gemma_model": "synthetic-fake-generator",
        "embedding_model": "synthetic-fake-embedder",
        "index_revision": "synthetic-index-v1",
        "prompt_revision": "synthetic-prompt-v1",
        "api_key": "safe-placeholder-only",
    }
    first_run = runner.run(
        questions,
        answerer=answerer,
        output_dir=output_dir,
        split="development",
        retrieval_k=5,
        run_config=config,
    )
    assert len(calls) == 2
    assert first_run.metrics["overall"]["citations"]["citation_validity"] == {"value": 1.0, "n": 2}
    assert first_run.predictions_path.is_file()
    assert first_run.metrics_json_path.is_file()
    assert first_run.metrics_csv_path.is_file()
    raw = first_run.predictions_path.read_text(encoding="utf-8")
    assert "SYNTHETIC TEST FIXTURE actual retrieved evidence" in raw
    metadata_text = first_run.metrics_json_path.read_text(encoding="utf-8")
    assert "safe-placeholder-only" not in metadata_text
    assert fingerprint_config({**config, "api_key": "placeholder-one"}) == fingerprint_config(
        {**config, "api_key": "placeholder-two"}
    )
    prediction_schema = json.loads(
        (Path(__file__).parents[3] / "evaluation" / "prediction.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(prediction_schema)
    prediction_validator = Draft202012Validator(prediction_schema, format_checker=FormatChecker())
    raw_prediction = json.loads(raw.splitlines()[0])
    assert list(prediction_validator.iter_errors(raw_prediction)) == []
    invalid_review = {**raw_prediction, "human_labels": {**raw_prediction["human_labels"], "answer_correct": True}}
    assert list(prediction_validator.iter_errors(invalid_review))

    # A reviewer can add labels without causing an expensive answer call on resume.
    records = [json.loads(line) for line in first_run.predictions_path.read_text(encoding="utf-8").splitlines()]
    records[0]["human_labels"]["answer_correct"] = True
    records[0]["human_labels"]["citation_support"] = False
    records[0]["human_labels"]["reviewer"] = "synthetic-human-reviewer"
    records[0]["human_labels"]["reviewed_at"] = "2026-10-01T00:00:00Z"
    assert list(prediction_validator.iter_errors(records[0])) == []
    first_run.predictions_path.write_text(
        "\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8"
    )
    resumed = runner.run(
        questions,
        answerer=answerer,
        output_dir=output_dir,
        split="development",
        retrieval_k=5,
        run_config=config,
    )
    assert len(calls) == 2
    assert resumed.metrics["overall"]["answer_quality"]["answer_correctness_human"] == {
        "value": 1.0,
        "n": 1,
    }
    assert resumed.metrics["overall"]["citations"]["citation_support_human"] == {
        "value": 0.0,
        "n": 1,
    }

    runner.run(
        questions,
        answerer=answerer,
        output_dir=output_dir,
        split="development",
        retrieval_k=5,
        run_config={**config, "prompt_revision": "synthetic-prompt-v2"},
    )
    assert len(calls) == 4


def test_runner_records_failed_requests_and_conflict_detection(tmp_path: Path) -> None:
    conflict_question = question(
        "synthetic-conflict",
        category="potential_conflict",
        expected_conflict=True,
    )
    failed_question = question("synthetic-failure")
    generation_error_question = question("synthetic-generation-error")

    def answerer(query: str) -> AnswerResult:
        if "synthetic-failure" in query:
            raise RuntimeError("synthetic test-only failure detail")
        if "synthetic-generation-error" in query:
            return AnswerResult(
                status=AnswerStatus.GENERATION_ERROR,
                answer=None,
                sources=(),
                error=ErrorInfo(code="synthetic_provider_error", message="SYNTHETIC TEST FIXTURE"),
            )
        return AnswerResult(
            status=AnswerStatus.ABSTAINED,
            answer="The answer could not be found in the documents.",
            sources=(),
        )

    runner = EvaluationRunner()
    run = runner.run(
        [conflict_question, failed_question, generation_error_question],
        answerer=answerer,
        conflict_detector=lambda item, sources: item.category == "potential_conflict",
        output_dir=tmp_path / "failure-run",
        split="development",
        retrieval_k=3,
        run_config={
            "gemma_model": "synthetic-gemma",
            "embedding_model": "synthetic-embedder",
            "index_revision": "synthetic-index",
            "prompt_revision": "synthetic-prompt",
        },
    )

    conflict = run.metrics["overall"]["potential_conflict_detection"]
    assert conflict["true_positive"] == 1
    assert conflict["false_positive"] == 0
    assert run.metrics["overall"]["operations"]["failed_requests"] == {"value": 2, "n": 3}
    failed = next(item for item in run.predictions if item.question_id == failed_question.id)
    assert failed.status == "request_error"
    assert failed.error_code == "request_exception"
    assert failed.request_failed is True
    generation_error = next(
        item for item in run.predictions if item.question_id == generation_error_question.id
    )
    assert generation_error.status == "generation_error"
    assert generation_error.answer is None
    assert generation_error.request_failed is True
    assert "synthetic test-only failure detail" not in run.predictions_path.read_text(encoding="utf-8")
