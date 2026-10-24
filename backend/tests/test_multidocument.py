"""Synthetic PDF ingestion and multi-source citation tests; no policy data."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from campusreg.config import RAGSettings
from campusreg.ingestion import PDFIngestor
from campusreg.rag import (
    AnswerStatus,
    DocumentIngestionError,
    FileTooLargeError,
    InvalidFileTypeError,
    MultiDocumentRAGEngine,
    PDFParseError,
    UnsupportedScannedPDFError,
    read_pdf,
)
from campusreg.storage import IngestionStatus, SQLiteDocumentStore


def synthetic_pdf(page_texts: Sequence[str]) -> bytes:
    """Build a small, valid PDF fixture carrying only labelled synthetic text."""

    page_ids = [3 + 2 * index for index in range(len(page_texts))]
    font_id = 3 + 2 * len(page_texts)
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        (
            b"<< /Type /Pages /Kids ["
            + b" ".join(f"{page_id} 0 R".encode() for page_id in page_ids)
            + f"] /Count {len(page_texts)} >>".encode()
        ),
    ]
    for index, text in enumerate(page_texts):
        page_id = page_ids[index]
        content_id = page_id + 1
        safe_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content = f"BT /F1 11 Tf 36 720 Td ({safe_text}) Tj ET".encode("ascii")
        page = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode()
        stream = f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream"
        objects.extend((page, stream))
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    output = bytearray(b"%PDF-1.4\n% synthetic test fixture\n")
    offsets = [0]
    for object_id, value in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{object_id} 0 obj\n".encode())
        output.extend(value)
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(offsets)}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()
    )
    return bytes(output)


class TopicEmbedder:
    """A deterministic test double: shared synthetic topic maps to one axis."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return np.asarray(
            [[1.0, 0.0] if "sharedtopic" in text.lower() else [0.0, 1.0] for text in texts],
            dtype=np.float32,
        )


class CitationEchoGenerator:
    def __init__(self) -> None:
        self.prompt = ""

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.prompt = prompt
        citations = re.findall(r"\[[^\]\n]+\]", prompt)
        return "Synthetic fixture evidence contains the shared topic. " + " ".join(citations)


def make_engine(tmp_path: Path, *, settings: RAGSettings | None = None):
    selected = settings or RAGSettings(top_k=8)
    embedder = TopicEmbedder()
    generator = CitationEchoGenerator()
    store = SQLiteDocumentStore(tmp_path / "synthetic-documents.sqlite3")
    engine = MultiDocumentRAGEngine(selected, embedder, generator, store)
    return engine, embedder, generator, store


def test_two_synthetic_pdfs_retrieve_overlapping_topics_with_exact_citations(tmp_path: Path) -> None:
    engine, _, generator, store = make_engine(tmp_path)
    first_pdf = synthetic_pdf(
        [
            "SYNTHETIC FIXTURE unrelated astronomy passage with several words",
            "SYNTHETIC FIXTURE sharedtopic appeal process passage with distinct source alpha",
        ]
    )
    second_pdf = synthetic_pdf(
        [
            "SYNTHETIC FIXTURE unrelated gardening passage with several words",
            "SYNTHETIC FIXTURE sharedtopic appeal process passage with distinct source beta",
        ]
    )

    first = engine.ingest_pdf(
        first_pdf,
        "synthetic-alpha.pdf",
        source_url="https://policy.example.invalid/synthetic-alpha.pdf",
        effective_date="2025-01-02",
        document_type="synthetic-test-fixture",
    )
    second = engine.ingest_pdf(second_pdf, "synthetic-beta.pdf", document_type="synthetic-test-fixture")

    assert first.document.content_hash != second.document.content_hash
    assert first.document.effective_date == date(2025, 1, 2)
    assert first.document.upload_status.value == "stored"
    assert first.document.ingestion_status is IngestionStatus.INDEXED
    assert [record.filename for record in store.list_documents()] == [
        "synthetic-alpha.pdf",
        "synthetic-beta.pdf",
    ]

    result = engine.answer("campus sharedtopic question")

    assert result.status is AnswerStatus.ANSWERED
    relevant_sources = [source for source in result.sources if "sharedtopic" in source.text.lower()]
    assert {(source.document_id, source.source_name, source.page_number) for source in relevant_sources} == {
        (first.document.id, "synthetic-alpha.pdf", 2),
        (second.document.id, "synthetic-beta.pdf", 2),
    }
    for source in result.sources:
        assert source.citation in generator.prompt
        assert source.text in generator.prompt
    assert f"document {first.document.id}, p. 2" in result.answer
    assert f"document {second.document.id}, p. 2" in result.answer


def test_duplicate_hash_restart_and_reindex_keep_other_document_intact(tmp_path: Path) -> None:
    engine, embedder, _, store = make_engine(tmp_path)
    alpha_pdf = synthetic_pdf(
        ["SYNTHETIC FIXTURE unrelated words for page one", "SYNTHETIC FIXTURE sharedtopic alpha passage"]
    )
    beta_pdf = synthetic_pdf(
        ["SYNTHETIC FIXTURE unrelated words for page one", "SYNTHETIC FIXTURE sharedtopic beta passage"]
    )
    alpha = engine.ingest_pdf(alpha_pdf, "synthetic-alpha.pdf")
    beta = engine.ingest_pdf(beta_pdf, "synthetic-beta.pdf")
    calls_before_duplicate = len(embedder.calls)

    duplicate = engine.ingest_pdf(alpha_pdf, "renamed-copy.pdf")
    assert duplicate.duplicate is True
    assert duplicate.document.id == alpha.document.id
    assert duplicate.document.filename == "synthetic-alpha.pdf"
    assert len(embedder.calls) == calls_before_duplicate

    reopened_store = SQLiteDocumentStore(store.path)
    reopened = MultiDocumentRAGEngine(engine.settings, engine.embedder, engine.generator, reopened_store)
    before = reopened_store.load_knowledge_base()
    beta_chunks_before = [chunk for chunk in before.chunks if chunk.document_id == beta.document.id]
    assert beta_chunks_before
    assert len(reopened.list_documents()) == 2

    reindexed = reopened.reindex_document(alpha.document.id)
    after_reindex = reopened_store.load_knowledge_base()
    beta_chunks_after = [chunk for chunk in after_reindex.chunks if chunk.document_id == beta.document.id]
    assert reindexed.document.ingestion_status is IngestionStatus.INDEXED
    assert beta_chunks_after == beta_chunks_before
    assert len(after_reindex.chunks) == len(before.chunks)

    assert reopened.delete_document(alpha.document.id) is True
    remaining = reopened_store.load_knowledge_base()
    assert {chunk.document_id for chunk in remaining.chunks} == {beta.document.id}
    assert [record.id for record in reopened.list_documents()] == [beta.document.id]


def test_failed_reindex_preserves_previous_index_and_other_sources(tmp_path: Path) -> None:
    engine, _, _, store = make_engine(tmp_path)
    alpha = engine.ingest_pdf(
        synthetic_pdf(["SYNTHETIC FIXTURE sharedtopic alpha content passage"]), "synthetic-alpha.pdf"
    )
    beta = engine.ingest_pdf(
        synthetic_pdf(["SYNTHETIC FIXTURE sharedtopic beta content passage"]), "synthetic-beta.pdf"
    )
    before = store.load_knowledge_base()

    class BrokenEmbedder:
        def encode(self, texts: Sequence[str]) -> np.ndarray:
            raise RuntimeError("test-only embedding failure")

    engine.ingestor.embedder = BrokenEmbedder()
    with pytest.raises(DocumentIngestionError, match="PDF embeddings could not be created") as error:
        engine.reindex_document(alpha.document.id)
    assert error.value.code == "embedding_error"

    after = store.load_knowledge_base()
    assert after.chunks == before.chunks
    assert store.get_document(alpha.document.id).ingestion_status is IngestionStatus.INDEXED
    assert store.get_document(beta.document.id).ingestion_status is IngestionStatus.INDEXED


def test_upload_validation_rejects_wrong_type_size_malformed_and_scanned(tmp_path: Path) -> None:
    small_settings = RAGSettings(max_pdf_size_bytes=200)
    engine, embedder, _, store = make_engine(tmp_path, settings=small_settings)
    ingestor = PDFIngestor(store, embedder, small_settings)

    with pytest.raises(InvalidFileTypeError):
        ingestor.ingest_pdf(b"not a pdf", "fixture.txt")
    with pytest.raises(FileTooLargeError):
        ingestor.ingest_pdf(synthetic_pdf(["SYNTHETIC FIXTURE text long enough to index"]), "large.pdf")

    larger_settings = RAGSettings(max_pdf_size_bytes=100_000)
    larger_ingestor = PDFIngestor(store, embedder, larger_settings)
    with pytest.raises(PDFParseError):
        larger_ingestor.ingest_pdf(b"%PDF-1.4\ntruncated synthetic fixture", "malformed.pdf")
    with pytest.raises(UnsupportedScannedPDFError):
        larger_ingestor.ingest_pdf(synthetic_pdf([""]), "synthetic-image-only.pdf")

    failed = store.list_documents()
    assert {record.ingestion_status for record in failed} == {IngestionStatus.FAILED}


def test_physical_page_records_are_retained_in_persistent_storage(tmp_path: Path) -> None:
    pdf = synthetic_pdf(
        [
            "SYNTHETIC FIXTURE first physical page text with enough words",
            "SYNTHETIC FIXTURE second physical page text with enough words",
            "SYNTHETIC FIXTURE third physical page text with enough words",
        ]
    )
    pdf_path = tmp_path / "synthetic-pages.pdf"
    pdf_path.write_bytes(pdf)
    pages = read_pdf(pdf_path)
    assert [page.page_number for page in pages] == [1, 2, 3]

    engine, _, _, store = make_engine(tmp_path)
    record = engine.ingest_pdf(pdf, "synthetic-pages.pdf").document
    with store._connect() as connection:
        stored_pages = connection.execute(
            "SELECT page_number, text FROM pages WHERE document_id = ? ORDER BY page_number",
            (record.id,),
        ).fetchall()
    assert [row["page_number"] for row in stored_pages] == [1, 2, 3]
    assert [chunk.page_number for chunk in store.load_knowledge_base().chunks] == [1, 2, 3]
