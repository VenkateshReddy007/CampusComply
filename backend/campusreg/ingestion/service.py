"""Validated, persistent PDF ingestion using the shared RAG algorithms."""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from pathlib import Path, PureWindowsPath
from urllib.parse import urlsplit
from uuid import uuid4

from campusreg.config import RAGSettings
from campusreg.rag.chunking import split_into_chunks
from campusreg.rag.documents import read_pdf_bytes
from campusreg.rag.embeddings import EmbeddingModel, normalize_embedding_matrix
from campusreg.rag.errors import (
    DocumentIngestionError,
    EmptyDocumentError,
    FileTooLargeError,
    InvalidFileTypeError,
    PDFParseError,
    UnsupportedScannedPDFError,
)
from campusreg.storage.models import DocumentRecord, IngestionResult
from campusreg.storage.sqlite_store import SQLiteDocumentStore


class PDFIngestor:
    """Validate uploads, build page-anchored chunks, and persist their vectors."""

    def __init__(
        self,
        store: SQLiteDocumentStore,
        embedder: EmbeddingModel,
        settings: RAGSettings | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.settings = settings or RAGSettings.baseline()

    def ingest_pdf(
        self,
        content: bytes,
        filename: str,
        *,
        source_url: str | None = None,
        effective_date: date | str | None = None,
        document_type: str = "unknown",
    ) -> IngestionResult:
        """Store a new PDF, or return its existing record when the hash repeats."""

        self._validate_file(content, filename)
        parsed_date = self._parse_date(effective_date)
        parsed_url = self._parse_url(source_url)
        if not isinstance(document_type, str):
            raise DocumentIngestionError("document_type must be text.", "invalid_document_type")
        normalized_type = document_type.strip()
        if not normalized_type:
            raise DocumentIngestionError("document_type must not be empty.", "invalid_document_type")
        content_hash = hashlib.sha256(content).hexdigest()
        existing = self.store.get_by_hash(content_hash)
        if existing is not None:
            return IngestionResult(document=existing, duplicate=True)

        document_id = str(uuid4())
        record, created = self.store.create_uploaded_document(
            document_id=document_id,
            filename=filename,
            content_hash=content_hash,
            source_url=parsed_url,
            effective_date=parsed_date,
            document_type=normalized_type,
            content=content,
        )
        if not created:
            return IngestionResult(document=record, duplicate=True)
        self.store.mark_processing(record.id)
        try:
            indexed = self._index(record.id, filename, content)
        except Exception as exc:
            code = exc.code if isinstance(exc, DocumentIngestionError) else (
                "empty_document" if isinstance(exc, EmptyDocumentError) else "indexing_failed"
            )
            self.store.mark_failed(record.id, code)
            if isinstance(exc, (DocumentIngestionError, EmptyDocumentError)):
                raise
            raise DocumentIngestionError("PDF indexing failed.", code) from None
        return IngestionResult(document=indexed, duplicate=False)

    def reindex_document(self, document_id: str) -> IngestionResult:
        """Rebuild one existing index from its stored PDF; other sources remain intact."""

        record = self.store.get_document(document_id)
        if record is None:
            raise KeyError(document_id)
        content = self.store.get_original_pdf(document_id)
        self.store.mark_processing(document_id, reindex=True)
        try:
            indexed = self._index(document_id, record.filename, content)
        except Exception as exc:
            code = exc.code if isinstance(exc, DocumentIngestionError) else (
                "empty_document" if isinstance(exc, EmptyDocumentError) else "indexing_failed"
            )
            self.store.mark_failed(document_id, code, reindex=True)
            if isinstance(exc, (DocumentIngestionError, EmptyDocumentError)):
                raise
            raise DocumentIngestionError("PDF re-indexing failed.", code) from None
        return IngestionResult(document=indexed, duplicate=False)

    def _index(self, document_id: str, filename: str, content: bytes) -> DocumentRecord:
        try:
            pages = read_pdf_bytes(content, filename, document_id=document_id)
        except ValueError as exc:
            if str(exc) == "encrypted_pdf_unsupported":
                raise DocumentIngestionError(
                    "Encrypted PDFs are not supported.", "encrypted_pdf_unsupported"
                ) from None
            raise PDFParseError() from None
        except Exception:
            raise PDFParseError() from None
        if not pages or all(not page.text.strip() for page in pages):
            raise UnsupportedScannedPDFError()
        chunks = split_into_chunks(pages, settings=self.settings, document_id=document_id)
        if not chunks:
            raise EmptyDocumentError("Extracted PDF text was too short to index.")
        try:
            vectors = normalize_embedding_matrix(
                self.embedder.encode([chunk.text for chunk in chunks]), expected_rows=len(chunks)
            )
        except Exception:
            raise DocumentIngestionError("PDF embeddings could not be created.", "embedding_error") from None
        return self.store.commit_index(
            document_id,
            pages,
            chunks,
            vectors,
            embedding_model=self.settings.embedding_model,
        )

    def _validate_file(self, content: bytes, filename: str) -> None:
        if not isinstance(content, bytes) or not content or not isinstance(filename, str):
            raise InvalidFileTypeError()
        path_name = Path(filename).name
        if (
            not filename
            or path_name != filename
            or PureWindowsPath(filename).name != filename
            or Path(filename).suffix.lower() != ".pdf"
        ):
            raise InvalidFileTypeError()
        if len(content) > self.settings.max_pdf_size_bytes:
            raise FileTooLargeError(self.settings.max_pdf_size_bytes)
        if content[:1024].find(b"%PDF-") < 0:
            raise InvalidFileTypeError()

    @staticmethod
    def _parse_date(value: date | str | None) -> date | None:
        if value is None or value == "":
            return None
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        try:
            return date.fromisoformat(value)
        except (TypeError, ValueError):
            raise DocumentIngestionError(
                "effective_date must be an ISO calendar date.", "invalid_effective_date"
            ) from None

    @staticmethod
    def _parse_url(value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise DocumentIngestionError("source_url must be an HTTP or HTTPS URL.", "invalid_source_url")
        if not value.strip():
            return None
        candidate = value.strip()
        parsed = urlsplit(candidate)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise DocumentIngestionError(
                "source_url must be an HTTP or HTTPS URL.", "invalid_source_url"
            )
        return candidate
