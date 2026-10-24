"""Build a source-provenance graph without inferring policy relationships."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from campusreg.rag.models import TextChunk


def build_source_knowledge_graph(
    chunks: Sequence[TextChunk], *, max_passages: int = 500
) -> dict[str, Any]:
    """Return document → physical PDF page → indexed passage relationships.

    The graph records provenance and exact source text only. It does not infer
    entities, legal relationships, applicability, or conflicts between rules.
    """

    if max_passages <= 0:
        raise ValueError("max_passages must be greater than zero")

    selected = list(chunks[:max_passages])
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    seen_documents: set[str] = set()
    seen_pages: set[tuple[str, int]] = set()

    for chunk in selected:
        document_id = chunk.document_id
        if document_id not in seen_documents:
            seen_documents.add(document_id)
            nodes.append({
                "id": f"document:{document_id}", "type": "document",
                "label": chunk.source_name, "document_id": document_id,
                "document_name": chunk.source_name, "page": None,
                "chunk_id": None, "citation": None, "text": None,
            })

        page_key = (document_id, chunk.page)
        page_id = f"page:{document_id}:{chunk.page}"
        if page_key not in seen_pages:
            seen_pages.add(page_key)
            nodes.append({
                "id": page_id, "type": "page", "label": f"PDF page {chunk.page}",
                "document_id": document_id, "document_name": chunk.source_name,
                "page": chunk.page, "chunk_id": None, "citation": None, "text": None,
            })
            edges.append({
                "source": f"document:{document_id}", "target": page_id,
                "relation": "contains_page",
            })

        passage_id = f"passage:{chunk.chunk_id}"
        nodes.append({
            "id": passage_id, "type": "passage",
            "label": f"Passage {chunk.chunk_id[:12]}",
            "document_id": document_id, "document_name": chunk.source_name,
            "page": chunk.page, "chunk_id": chunk.chunk_id,
            "citation": f"[{chunk.source_name}, document {document_id}, p. {chunk.page}]",
            "text": chunk.text,
        })
        edges.append({"source": page_id, "target": passage_id, "relation": "contains_passage"})

    return {
        "nodes": nodes,
        "edges": edges,
        "truncated": len(chunks) > max_passages,
        "scope_note": (
            "Connections show document provenance only: document to physical PDF page to indexed passage. "
            "No policy facts or relationships are inferred."
        ),
    }
