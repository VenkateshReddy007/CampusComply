"""Word-based page chunking that preserves the starter's overlap behavior."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import uuid4

from campusreg.config import RAGSettings
from campusreg.rag.models import DocumentPage, TextChunk


def split_into_chunks(
    pages: Sequence[DocumentPage | str],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    *,
    source_name: str = "document",
    document_id: str = "",
    settings: RAGSettings | None = None,
) -> list[TextChunk]:
    """Split each page into overlapping word windows, skipping <=5 words.

    Page boundaries remain independent, as in the notebook implementation.
    """

    selected = settings or RAGSettings.baseline()
    size = selected.chunk_size if chunk_size is None else chunk_size
    overlap = selected.chunk_overlap if chunk_overlap is None else chunk_overlap
    if size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= size:
        raise ValueError("chunk_overlap must be non-negative and less than chunk_size")

    page_document_ids = {
        page.document_id
        for page in pages
        if isinstance(page, DocumentPage) and page.document_id
    }
    fallback_document_id = document_id or (
        next(iter(page_document_ids)) if len(page_document_ids) == 1 else str(uuid4())
    )
    chunks: list[TextChunk] = []
    step = max(1, size - overlap)
    for index, page in enumerate(pages, start=1):
        if isinstance(page, DocumentPage):
            page_number = page.page
            page_text = page.text
            name = page.source_name
            source_document_id = page.document_id or fallback_document_id
        else:
            page_number = index
            page_text = page
            name = source_name
            source_document_id = fallback_document_id

        words = page_text.split()
        for start in range(0, len(words), step):
            chunk_words = words[start : start + size]
            if len(chunk_words) > 5:
                chunks.append(
                    TextChunk(
                        page=page_number,
                        text=" ".join(chunk_words),
                        source_name=name,
                        document_id=source_document_id,
                        chunk_id=str(uuid4()),
                    )
                )
            if start + size >= len(words):
                break
    return chunks
