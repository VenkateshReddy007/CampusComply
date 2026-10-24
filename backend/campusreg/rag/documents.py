"""PDF text extraction with one-based page and source metadata."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from campusreg.rag.models import DocumentPage


def read_pdf(path: str | Path, *, document_id: str = "") -> list[DocumentPage]:
    """Extract one text record per PDF page using pypdf."""

    pdf_path = Path(path)
    reader = PdfReader(str(pdf_path))
    return [
        DocumentPage(
            page=index,
            text=page.extract_text() or "",
            source_name=pdf_path.name,
            document_id=document_id,
        )
        for index, page in enumerate(reader.pages, start=1)
    ]


def read_pdf_bytes(
    content: bytes,
    filename: str,
    *,
    document_id: str = "",
) -> list[DocumentPage]:
    """Extract one text record per physical PDF page from uploaded bytes."""

    reader = PdfReader(BytesIO(content), strict=True)
    if reader.is_encrypted:
        raise ValueError("encrypted_pdf_unsupported")
    return [
        DocumentPage(
            page=index,
            text=page.extract_text() or "",
            source_name=filename,
            document_id=document_id,
        )
        for index, page in enumerate(reader.pages, start=1)
    ]
