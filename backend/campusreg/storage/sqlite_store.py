"""SQLite persistence for uploaded PDFs, page text, chunks and vectors."""

from __future__ import annotations

import sqlite3
import json
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any, Iterator, Sequence
from uuid import uuid4

import numpy as np

from campusreg.rag.errors import IndexCompatibilityError, KnowledgeBaseNotLoadedError
from campusreg.rag.models import DocumentPage, KnowledgeBase, TextChunk
from campusreg.storage.models import ConversationMessage, DocumentRecord, IngestionStatus, UploadStatus


class SQLiteDocumentStore:
    """Store source PDFs and indexes in a local SQLite database.

    The database contains private uploaded source documents and should be kept
    out of version control. Each indexing replacement is committed atomically.
    """

    def __init__(self, path: str | Path) -> None:
        requested_path = str(path)
        self._uri = requested_path == ":memory:"
        self._anchor: sqlite3.Connection | None = None
        if self._uri:
            self.path = f"file:campusreg-{uuid4().hex}?mode=memory&cache=shared"
            self._anchor = sqlite3.connect(self.path, uri=True, timeout=15)
            self._anchor.execute("PRAGMA foreign_keys = ON")
        else:
            self.path = requested_path
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE,
                    source_url TEXT,
                    effective_date TEXT,
                    document_type TEXT NOT NULL,
                    upload_status TEXT NOT NULL,
                    ingestion_status TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    original_pdf BLOB NOT NULL,
                    page_count INTEGER NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    embedding_model TEXT,
                    embedding_dimension INTEGER,
                    last_error_code TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pages (
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    page_number INTEGER NOT NULL CHECK (page_number > 0),
                    text TEXT NOT NULL,
                    PRIMARY KEY (document_id, page_number)
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    page_number INTEGER NOT NULL CHECK (page_number > 0),
                    ordinal INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    vector_dimension INTEGER NOT NULL CHECK (vector_dimension > 0),
                    UNIQUE (document_id, ordinal),
                    FOREIGN KEY (document_id, page_number)
                        REFERENCES pages(document_id, page_number) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS chunks_by_document
                    ON chunks(document_id, ordinal);
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    original_question TEXT,
                    retrieval_query TEXT,
                    answer_status TEXT,
                    sources_json TEXT NOT NULL DEFAULT '[]'
                );
                CREATE INDEX IF NOT EXISTS conversation_messages_order
                    ON conversation_messages(conversation_id, id);
                PRAGMA user_version = 2;
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=15, uri=self._uri)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _record(row: sqlite3.Row | None) -> DocumentRecord | None:
        if row is None:
            return None
        return DocumentRecord(
            id=row["id"],
            filename=row["filename"],
            content_hash=row["content_hash"],
            source_url=row["source_url"],
            effective_date=(date.fromisoformat(row["effective_date"]) if row["effective_date"] else None),
            document_type=row["document_type"],
            upload_status=UploadStatus(row["upload_status"]),
            ingestion_status=IngestionStatus(row["ingestion_status"]),
            size_bytes=row["size_bytes"],
            page_count=row["page_count"],
            chunk_count=row["chunk_count"],
            embedding_model=row["embedding_model"],
            embedding_dimension=row["embedding_dimension"],
            last_error_code=row["last_error_code"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get_by_hash(self, content_hash: str) -> DocumentRecord | None:
        with self._connect() as connection:
            return self._record(
                connection.execute(
                    "SELECT * FROM documents WHERE content_hash = ?", (content_hash,)
                ).fetchone()
            )

    def create_uploaded_document(
        self,
        *,
        document_id: str,
        filename: str,
        content_hash: str,
        source_url: str | None,
        effective_date: date | None,
        document_type: str,
        content: bytes,
    ) -> tuple[DocumentRecord, bool]:
        """Insert a new source or return the existing row for a concurrent duplicate."""

        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._connect() as connection, connection:
                connection.execute(
                    """INSERT INTO documents (
                        id, filename, content_hash, source_url, effective_date,
                        document_type, upload_status, ingestion_status, size_bytes,
                        original_pdf, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'stored', 'uploaded', ?, ?, ?, ?)""",
                    (
                        document_id,
                        filename,
                        content_hash,
                        source_url,
                        effective_date.isoformat() if effective_date else None,
                        document_type,
                        len(content),
                        sqlite3.Binary(content),
                        now,
                        now,
                    ),
                )
        except sqlite3.IntegrityError:
            existing = self.get_by_hash(content_hash)
            if existing is None:
                raise
            return existing, False
        record = self.get_document(document_id)
        assert record is not None
        return record, True

    def get_document(self, document_id: str) -> DocumentRecord | None:
        with self._connect() as connection:
            return self._record(
                connection.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
            )

    def list_documents(self) -> list[DocumentRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY created_at, id"
            ).fetchall()
        return [record for row in rows if (record := self._record(row)) is not None]

    def get_original_pdf(self, document_id: str) -> bytes:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT original_pdf FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        if row is None:
            raise KeyError(document_id)
        return bytes(row["original_pdf"])

    def mark_processing(self, document_id: str, *, reindex: bool = False) -> None:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        status = "reindexing" if reindex else "processing"
        with self._connect() as connection, connection:
            connection.execute(
                "UPDATE documents SET ingestion_status = ?, updated_at = ? WHERE id = ?",
                (status, now, document_id),
            )

    def mark_failed(self, document_id: str, error_code: str, *, reindex: bool = False) -> None:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        if reindex:
            current = self.get_document(document_id)
            status = "indexed" if current and current.chunk_count else "failed"
        else:
            status = "failed"
        with self._connect() as connection, connection:
            connection.execute(
                """UPDATE documents SET ingestion_status = ?, last_error_code = ?, updated_at = ?
                   WHERE id = ?""",
                (status, error_code, now, document_id),
            )

    def commit_index(
        self,
        document_id: str,
        pages: Sequence[DocumentPage],
        chunks: Sequence[TextChunk],
        vectors: np.ndarray,
        *,
        embedding_model: str,
    ) -> DocumentRecord:
        """Replace one document's page/chunk index as a single transaction."""

        from datetime import datetime, timezone

        matrix = np.asarray(vectors, dtype="<f4")
        if matrix.ndim != 2 or matrix.shape[0] != len(chunks) or not len(chunks):
            raise ValueError("vectors must align with at least one chunk")
        if matrix.shape[1] == 0 or not np.isfinite(matrix).all():
            raise ValueError("vectors must be finite, non-empty vectors")
        if any(chunk.document_id != document_id for chunk in chunks):
            raise ValueError("all chunks must belong to the indexed document")
        if any(page.document_id != document_id for page in pages):
            raise ValueError("all pages must belong to the indexed document")
        if [page.page_number for page in pages] != list(range(1, len(pages) + 1)):
            raise ValueError("page numbers must match the physical order in the original PDF")
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection, connection:
            connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            connection.execute("DELETE FROM pages WHERE document_id = ?", (document_id,))
            connection.executemany(
                "INSERT INTO pages(document_id, page_number, text) VALUES (?, ?, ?)",
                [(document_id, page.page_number, page.text) for page in pages],
            )
            connection.executemany(
                """INSERT INTO chunks(
                    chunk_id, document_id, filename, page_number, ordinal, text, vector, vector_dimension
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        chunk.chunk_id,
                        document_id,
                        chunk.source_name,
                        chunk.page_number,
                        ordinal,
                        chunk.text,
                        sqlite3.Binary(matrix[ordinal].tobytes()),
                        matrix.shape[1],
                    )
                    for ordinal, chunk in enumerate(chunks)
                ],
            )
            connection.execute(
                """UPDATE documents SET ingestion_status = 'indexed', page_count = ?,
                    chunk_count = ?, embedding_model = ?, embedding_dimension = ?,
                    last_error_code = NULL, updated_at = ? WHERE id = ?""",
                (len(pages), len(chunks), embedding_model, matrix.shape[1], now, document_id),
            )
        record = self.get_document(document_id)
        if record is None:
            raise KeyError(document_id)
        return record

    def load_knowledge_base(self) -> KnowledgeBase:
        """Load all indexed sources into one matrix without losing their identities."""

        with self._connect() as connection:
            rows = connection.execute(
                """SELECT c.*, d.embedding_model FROM chunks c
                   JOIN documents d ON d.id = c.document_id
                   WHERE d.ingestion_status IN ('indexed', 'reindexing')
                   ORDER BY d.created_at, d.id, c.ordinal"""
            ).fetchall()
        if not rows:
            raise KnowledgeBaseNotLoadedError("No indexed documents are available.")
        models = {row["embedding_model"] for row in rows}
        dimensions = {row["vector_dimension"] for row in rows}
        if len(models) != 1 or len(dimensions) != 1:
            raise IndexCompatibilityError(
                "Indexed documents use incompatible embedding models or dimensions; re-index them with one model."
            )
        chunks: list[TextChunk] = []
        vectors: list[np.ndarray] = []
        expected_dimension = next(iter(dimensions))
        for row in rows:
            vector = np.frombuffer(row["vector"], dtype="<f4")
            if vector.size != expected_dimension:
                raise IndexCompatibilityError("Stored vector data is incomplete; re-index the affected document.")
            chunks.append(
                TextChunk(
                    page=row["page_number"],
                    text=row["text"],
                    source_name=row["filename"],
                    document_id=row["document_id"],
                    chunk_id=row["chunk_id"],
                )
            )
            vectors.append(vector)
        return KnowledgeBase(
            name="indexed documents",
            chunks=tuple(chunks),
            vectors=np.vstack(vectors).astype(np.float32, copy=False),
        )

    def list_source_graph_chunks(self, *, limit: int = 501) -> tuple[TextChunk, ...]:
        """Read bounded passage metadata for a provenance graph without vectors."""

        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT c.document_id, c.filename, c.page_number, c.chunk_id, c.text
                   FROM chunks c JOIN documents d ON d.id = c.document_id
                   WHERE d.ingestion_status = 'indexed'
                   ORDER BY d.created_at, d.id, c.ordinal LIMIT ?""",
                (limit,),
            ).fetchall()
        return tuple(
            TextChunk(
                page=row["page_number"], text=row["text"],
                source_name=row["filename"], document_id=row["document_id"],
                chunk_id=row["chunk_id"],
            )
            for row in rows
        )

    def delete_document(self, document_id: str) -> bool:
        with self._connect() as connection, connection:
            cursor = connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        return cursor.rowcount > 0

    def create_conversation(self) -> str:
        """Create an isolated, persistent conversation and return its ID."""

        from datetime import datetime, timezone

        conversation_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection, connection:
            connection.execute(
                "INSERT INTO conversations(id, created_at, updated_at) VALUES (?, ?, ?)",
                (conversation_id, now, now),
            )
        return conversation_id

    def conversation_exists(self, conversation_id: str) -> bool:
        with self._connect() as connection:
            return connection.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone() is not None

    def append_conversation_message(
        self,
        conversation_id: str,
        *,
        role: str,
        content: str,
        original_question: str | None = None,
        retrieval_query: str | None = None,
        answer_status: str | None = None,
        sources: Sequence[dict[str, Any]] = (),
    ) -> ConversationMessage:
        """Persist a user or assistant message plus question-rewrite audit fields."""

        from datetime import datetime, timezone

        if role not in {"user", "assistant"}:
            raise ValueError("conversation message role must be user or assistant")
        if not isinstance(content, str):
            raise ValueError("conversation message content must be text")
        now = datetime.now(timezone.utc).isoformat()
        sources_json = json.dumps(list(sources), ensure_ascii=False, sort_keys=True)
        with self._connect() as connection, connection:
            exists = connection.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if exists is None:
                raise KeyError(conversation_id)
            cursor = connection.execute(
                """INSERT INTO conversation_messages(
                    conversation_id, role, content, created_at, original_question,
                    retrieval_query, answer_status, sources_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    conversation_id, role, content, now, original_question,
                    retrieval_query, answer_status, sources_json,
                ),
            )
            message_id = int(cursor.lastrowid)
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
        message = self.get_conversation_message(message_id)
        if message is None:
            raise RuntimeError("conversation message was not persisted")
        return message

    @staticmethod
    def _conversation_message(row: sqlite3.Row | None) -> ConversationMessage | None:
        if row is None:
            return None
        try:
            sources = json.loads(row["sources_json"])
        except (json.JSONDecodeError, TypeError):
            sources = []
        if not isinstance(sources, list):
            sources = []
        return ConversationMessage(
            id=row["id"],
            conversation_id=row["conversation_id"],
            role=row["role"],
            content=row["content"],
            created_at=row["created_at"],
            original_question=row["original_question"],
            retrieval_query=row["retrieval_query"],
            answer_status=row["answer_status"],
            sources=tuple(item for item in sources if isinstance(item, dict)),
        )

    def get_conversation_message(self, message_id: int) -> ConversationMessage | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM conversation_messages WHERE id = ?", (message_id,)
            ).fetchone()
        return self._conversation_message(row)

    def list_conversation_messages(
        self,
        conversation_id: str,
        *,
        limit: int | None = None,
    ) -> list[ConversationMessage]:
        if limit is not None and limit <= 0:
            raise ValueError("conversation message limit must be positive")
        if not self.conversation_exists(conversation_id):
            raise KeyError(conversation_id)
        with self._connect() as connection:
            if limit is None:
                rows = connection.execute(
                    "SELECT * FROM conversation_messages WHERE conversation_id = ? ORDER BY id",
                    (conversation_id,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """SELECT * FROM (
                        SELECT * FROM conversation_messages WHERE conversation_id = ?
                        ORDER BY id DESC LIMIT ?
                    ) ORDER BY id""",
                    (conversation_id, limit),
                ).fetchall()
        return [message for row in rows if (message := self._conversation_message(row)) is not None]

    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear messages but retain the conversation ID for reuse."""

        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection, connection:
            exists = connection.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if exists is None:
                return False
            connection.execute(
                "DELETE FROM conversation_messages WHERE conversation_id = ?", (conversation_id,)
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id)
            )
        return True

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and its messages from persistent storage."""

        with self._connect() as connection, connection:
            cursor = connection.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        return cursor.rowcount > 0
