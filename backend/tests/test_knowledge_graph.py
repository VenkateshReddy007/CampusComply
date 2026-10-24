from __future__ import annotations

import pytest

from campusreg.rag.knowledge_graph import build_source_knowledge_graph
from campusreg.rag.models import TextChunk


def test_graph_preserves_document_page_and_passage_provenance() -> None:
    chunks = [
        TextChunk(1, "First source text.", "policy-a.pdf", "doc-a", "chunk-a"),
        TextChunk(1, "Another passage on the same page.", "policy-a.pdf", "doc-a", "chunk-b"),
        TextChunk(2, "Different source text.", "policy-b.pdf", "doc-b", "chunk-c"),
    ]

    graph = build_source_knowledge_graph(chunks)

    assert [node["type"] for node in graph["nodes"]].count("document") == 2
    assert [node["type"] for node in graph["nodes"]].count("page") == 2
    passage_nodes = [node for node in graph["nodes"] if node["type"] == "passage"]
    assert [node["text"] for node in passage_nodes] == [chunk.text for chunk in chunks]
    assert passage_nodes[0]["citation"] == "[policy-a.pdf, document doc-a, p. 1]"
    assert {edge["relation"] for edge in graph["edges"]} == {"contains_page", "contains_passage"}
    assert "No policy facts or relationships are inferred." in graph["scope_note"]
    assert graph["truncated"] is False


def test_graph_bounds_passages_and_reports_truncation() -> None:
    chunks = [TextChunk(1, str(i), "policy.pdf", "doc", f"chunk-{i}") for i in range(3)]
    graph = build_source_knowledge_graph(chunks, max_passages=2)
    assert len([node for node in graph["nodes"] if node["type"] == "passage"]) == 2
    assert graph["truncated"] is True


def test_graph_requires_positive_limit() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        build_source_knowledge_graph([], max_passages=0)
