"""Synthetic-only conflict detection tests; no institutional policy is used."""

from __future__ import annotations

from datetime import date

import pytest
import json
from pathlib import Path
from jsonschema import validate

from campusreg.rag.conflicts import (
    ConflictAssessment,
    ConflictDetector,
    ConflictPassage,
    ConflictStatus,
    conflict_metrics,
)
from campusreg.rag.models import RetrievedSource


SYNTHETIC_CASES = (
    {
        "id": "synthetic-contradiction",
        "a": "SYNTHETIC FIXTURE attendance exam eligibility requires 75 percent attendance.",
        "b": "SYNTHETIC FIXTURE attendance exam eligibility requires 80 percent attendance.",
        "status": ConflictStatus.POTENTIAL_CONTRADICTION,
        "positive": True,
    },
    {
        "id": "synthetic-supersession",
        "a": "SYNTHETIC FIXTURE library renewal notice: superseded from 2024-01-01.",
        "b": "SYNTHETIC FIXTURE library renewal notice: this rule supersedes the 2024 rule from 2025-01-01.",
        "status": ConflictStatus.POSSIBLE_SUPERSESSION,
        "positive": True,
    },
    {
        "id": "synthetic-different-scope",
        "a": "SYNTHETIC FIXTURE hostel visitor rule applies to first-year residents.",
        "b": "SYNTHETIC FIXTURE hostel visitor rule applies to postgraduate residents.",
        "status": ConflictStatus.DIFFERENT_SCOPES,
        "positive": False,
    },
    {
        "id": "synthetic-valid-exception",
        "a": "SYNTHETIC FIXTURE library access rule applies during ordinary hours.",
        "b": "SYNTHETIC FIXTURE library access rule has an exception for exam week.",
        "status": ConflictStatus.VALID_EXCEPTION,
        "positive": False,
    },
    {
        "id": "synthetic-near-duplicate",
        "a": "SYNTHETIC FIXTURE hostel visitor register records visitor name.",
        "b": "SYNTHETIC FIXTURE hostel visitor register records visitor name at entry.",
        "status": ConflictStatus.NOT_CONFLICTING,
        "positive": False,
    },
    {
        "id": "synthetic-insufficient",
        "a": "SYNTHETIC FIXTURE examination appeal process is described here.",
        "b": "SYNTHETIC FIXTURE examination appeal process details are incomplete.",
        "status": ConflictStatus.INSUFFICIENT_EVIDENCE,
        "positive": False,
    },
)


class LabelledSyntheticComparator:
    def __init__(self) -> None:
        self._case_index = 0
        self.seen_metadata = []

    def compare(self, a, b, *, metadata_a=None, metadata_b=None):
        case = SYNTHETIC_CASES[self._case_index]
        self._case_index += 1
        self.seen_metadata.append((metadata_a, metadata_b))
        return ConflictAssessment(
            subject="SYNTHETIC FIXTURE shared test subject",
            status=case["status"],
            explanation="SYNTHETIC FIXTURE comparison used only the paired passages.",
            quote_a=a.text,
            quote_b=b.text,
            scope_a="first-year residents" if case["id"] == "synthetic-different-scope" else None,
            scope_b="postgraduate residents" if case["id"] == "synthetic-different-scope" else None,
            date_a="2024-01-01" if case["id"] == "synthetic-supersession" else None,
            date_b="2025-01-01" if case["id"] == "synthetic-supersession" else None,
            limitations=("Synthetic fixture only; human review required.",),
        )


def _pair(case, index: int) -> tuple[RetrievedSource, RetrievedSource]:
    # A shared subject token sequence makes the candidate selector pair these
    # passages; each item has a distinct synthetic document identity.
    return (
        RetrievedSource(1, case["a"], f"synthetic-{index}-a.pdf", 0.8, f"synthetic-{index}-a", f"chunk-{index}-a"),
        RetrievedSource(2, case["b"], f"synthetic-{index}-b.pdf", 0.7, f"synthetic-{index}-b", f"chunk-{index}-b"),
    )


def test_synthetic_labelled_conflict_categories_and_precision_recall() -> None:
    comparator = LabelledSyntheticComparator()
    detector = ConflictDetector(comparator)
    reports = [
        detector.detect(_pair(case, index))[0]
        for index, case in enumerate(SYNTHETIC_CASES)
    ]

    assert len(reports) == 6
    assert [report.status for report in reports] == [case["status"] for case in SYNTHETIC_CASES]
    for report, case in zip(reports, SYNTHETIC_CASES):
        assert report.passage_a.text == case["a"]
        assert report.passage_b.text == case["b"]
        assert report.passage_a.page == 1
        assert report.passage_b.page == 2
        assert report.human_review_required is True
        assert report.to_dict()["passage_a"]["text"] == case["a"]
        assert report.to_dict()["passage_b"]["text"] == case["b"]

    metrics = conflict_metrics([
        (case["id"], report.status in {
            ConflictStatus.POTENTIAL_CONTRADICTION, ConflictStatus.POSSIBLE_SUPERSESSION,
        }, case["positive"])
        for case, report in zip(SYNTHETIC_CASES, reports)
    ])
    assert metrics["true_positive"] == 2
    assert metrics["false_positive"] == 0
    assert metrics["false_negative"] == 0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["false_positive_ids"] == []
    assert metrics["false_negative_ids"] == []


def test_model_quotes_and_dates_must_match_indexed_text_or_verified_metadata() -> None:
    class ForgingComparator:
        def compare(self, a, b, **kwargs):
            return ConflictAssessment(
                "attendance", ConflictStatus.POTENTIAL_CONTRADICTION, "different values",
                "fabricated quote from another document", b.text, date_a="2099-01-01",
            )

    first, second = _pair(SYNTHETIC_CASES[0], 7)
    reports = ConflictDetector(ForgingComparator()).detect([first, second])

    assert len(reports) == 1
    assert reports[0].status is ConflictStatus.INSUFFICIENT_EVIDENCE
    assert "could not be compared" in reports[0].explanation


def test_scope_and_date_metadata_are_verified_before_exposure() -> None:
    case = SYNTHETIC_CASES[1]
    a, b = _pair(case, 9)

    class MetadataComparator:
        def compare(self, left, right, *, metadata_a=None, metadata_b=None):
            assert metadata_a.effective_date == date(2024, 1, 1)
            assert metadata_b.effective_date == date(2025, 1, 1)
            return ConflictAssessment(
                "SYNTHETIC FIXTURE library renewal", ConflictStatus.POSSIBLE_SUPERSESSION,
                "SYNTHETIC FIXTURE newer passage explicitly references supersession.",
                left.text, right.text, date_a="2024-01-01", date_b="2025-01-01",
            )

    def record(document_id: str, effective: date):
        # The detector only uses the verified effective-date field here.
        from campusreg.storage.models import DocumentRecord, IngestionStatus, UploadStatus
        return DocumentRecord(
            id=document_id, filename=document_id + ".pdf", content_hash="hash", source_url=None,
            effective_date=effective, document_type="synthetic-test-fixture",
            upload_status=UploadStatus.STORED, ingestion_status=IngestionStatus.INDEXED,
            size_bytes=1, page_count=1, chunk_count=1, embedding_model="synthetic",
            embedding_dimension=2, last_error_code=None, created_at="", updated_at="",
        )

    reports = ConflictDetector(MetadataComparator()).detect(
        [a, b], documents={a.document_id: record(a.document_id, date(2024, 1, 1)),
                           b.document_id: record(b.document_id, date(2025, 1, 1))},
    )

    assert reports[0].effective_date_a == "2024-01-01"
    assert reports[0].effective_date_b == "2025-01-01"


def test_conflict_metrics_rejects_malformed_label_records() -> None:
    with pytest.raises(ValueError, match="id, prediction, and gold"):
        conflict_metrics([("bad", True)])  # type: ignore[list-item]


def test_conflict_report_matches_published_json_schema() -> None:
    class Comparator:
        def compare(self, a, b, **kwargs):
            return ConflictAssessment(
                "SYNTHETIC FIXTURE attendance exam eligibility",
                ConflictStatus.POTENTIAL_CONTRADICTION,
                "SYNTHETIC FIXTURE requirements differ; human review required.",
                a.text,
                b.text,
            )

    first, second = _pair(SYNTHETIC_CASES[0], 17)
    report = ConflictDetector(Comparator()).detect([first, second])[0]
    root = Path(__file__).resolve().parents[2]
    schema = json.loads((root / "docs" / "conflict-report.schema.json").read_text(encoding="utf-8"))

    validate(instance=report.to_dict(), schema=schema)
