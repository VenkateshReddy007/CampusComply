"""Synthetic privacy tests for the local PRAG preprocessing boundary."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from campusreg.config import RAGSettings
from campusreg.rag.models import AnswerStatus
from campusreg.rag.pipeline import MultiDocumentRAGEngine
from campusreg.rag.errors import PrivacyReindexRequiredError
from campusreg.rag.privacy import RedactionSession, redact_pages, redact_text
from campusreg.storage.sqlite_store import SQLiteDocumentStore
from test_multidocument import synthetic_pdf


class RecordingEmbedder:
    def __init__(self) -> None:
        self.inputs: list[str] = []

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.inputs.extend(texts)
        return np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))


class RecordingGenerator:
    def __init__(self) -> None:
        self.prompt = ""

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.prompt = prompt
        import re

        citations = re.findall(r"\[[^\]\n]+, document [^\]\n]+, p\. \d+\]", prompt)
        return "Attendance requirements are stated in the cited passage. " + " ".join(citations)


def test_supported_sensitive_fields_are_redacted_and_policy_numbers_remain() -> None:
    raw = (
        "Student name: Jane Doe; student ID: S123456; email: jane@example.edu; "
        "phone: +919876543210. Section 4.2 sets attendance at 75% by 2026-11-01."
    )
    clean, counts = redact_text(raw)

    assert "Jane Doe" not in clean
    assert "S123456" not in clean
    assert "jane@example.edu" not in clean
    assert "+919876543210" not in clean
    assert "[PERSON_001]" in clean
    assert "[STUDENT_ID_001]" in clean
    assert "[EMAIL_001]" in clean
    assert "[PHONE_001]" in clean
    assert "Section 4.2" in clean and "75%" in clean and "2026-11-01" in clean
    assert counts == {"PERSON": 1, "STUDENT_ID": 1, "EMAIL": 1, "PHONE": 1}


def test_placeholder_identity_is_consistent_within_one_document_scope() -> None:
    session = RedactionSession()
    first, _ = session.redact("Student name: Jane Doe")
    second, _ = session.redact("Reviewed by student name: Jane Doe")
    assert first == "Student name: [PERSON_001]"
    assert second.endswith("[PERSON_001]")


def test_ingestion_and_question_models_receive_only_redacted_content(tmp_path) -> None:
    raw_policy = (
        "SYNTHETIC TEST FIXTURE Student name: Jane Doe student ID: S123456 "
        "email: jane@example.edu attendance requirements appear in section 4.2 at 75 percent "
        "and the document provides a review process for the stated requirement."
    )
    embedder = RecordingEmbedder()
    generator = RecordingGenerator()
    store = SQLiteDocumentStore(tmp_path / "synthetic-private.sqlite3")
    engine = MultiDocumentRAGEngine(
        RAGSettings(top_k=2, relevance_threshold=None), embedder, generator, store
    )
    record = engine.ingest_pdf(
        synthetic_pdf([raw_policy]), "synthetic-policy-fixture.pdf", document_type="synthetic-test-fixture"
    ).document

    assert all("Jane Doe" not in value and "S123456" not in value and "jane@example.edu" not in value for value in embedder.inputs)
    stored = store.load_knowledge_base()
    assert all("Jane Doe" not in chunk.text and "S123456" not in chunk.text for chunk in stored.chunks)
    assert record.redaction_summary["total"] == 3
    assert record.redaction_summary["counts_by_page"] == {"1": {"PERSON": 1, "STUDENT_ID": 1, "EMAIL": 1}}

    result = engine.answer(
        "Campus attendance question from student name: Jane Doe, student ID: S123456, jane@example.edu"
    )
    assert result.status is AnswerStatus.ANSWERED
    assert "Jane Doe" not in generator.prompt
    assert "S123456" not in generator.prompt
    assert "jane@example.edu" not in generator.prompt
    assert "[PERSON_001]" in generator.prompt


def test_page_redaction_preserves_source_page_and_reports_safe_summary() -> None:
    from campusreg.rag.models import DocumentPage

    pages, summary = redact_pages((DocumentPage(7, "Employee name: Alex Smith email: alex@example.edu", "policy.pdf"),))
    assert pages[0].page_number == 7
    assert "Alex Smith" not in pages[0].text
    assert "alex@example.edu" not in pages[0].text
    assert summary.counts_by_page == {7: {"PERSON": 1, "EMAIL": 1}}


def test_pre_privacy_index_is_blocked_until_explicit_reindex(tmp_path) -> None:
    embedder = RecordingEmbedder()
    store = SQLiteDocumentStore(tmp_path / "synthetic-legacy.sqlite3")
    engine = MultiDocumentRAGEngine(
        RAGSettings(relevance_threshold=None), embedder, RecordingGenerator(), store
    )
    record = engine.ingest_pdf(
        synthetic_pdf(["SYNTHETIC TEST FIXTURE campus attendance requirement has enough words"]),
        "synthetic-legacy.pdf",
    ).document
    with store._connect() as connection, connection:
        connection.execute("UPDATE documents SET redaction_version = 0 WHERE id = ?", (record.id,))

    try:
        store.load_knowledge_base()
    except PrivacyReindexRequiredError:
        pass
    else:
        raise AssertionError("pre-privacy text index must remain unavailable to model paths")
    assert store.list_source_graph_chunks() == ()

    reindexed = engine.reindex_document(record.id)
    assert reindexed.document.redaction_version == 1
    assert store.load_knowledge_base().chunks
