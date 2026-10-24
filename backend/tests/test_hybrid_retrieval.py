"""BM25 and RRF tests use explicitly synthetic passage fixtures."""

from __future__ import annotations

import numpy as np

from campusreg.config import RAGSettings
from campusreg.rag.models import KnowledgeBase, TextChunk
from campusreg.rag.retrieval import retrieve, retrieve_bm25, retrieve_configured, retrieve_hybrid


class QueryEmbedder:
    def encode(self, texts):
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


def _knowledge_base() -> KnowledgeBase:
    chunks = (
        TextChunk(1, "SYNTHETIC TEST FIXTURE general registration information", "guide.pdf", "doc-a", "chunk-semantic"),
        TextChunk(7, "SYNTHETIC TEST FIXTURE rule 47-C deadline 2028-09-30", "calendar.pdf", "doc-b", "chunk-exact"),
        TextChunk(3, "SYNTHETIC TEST FIXTURE unrelated dining information", "other.pdf", "doc-c", "chunk-other"),
    )
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32)
    return KnowledgeBase("SYNTHETIC TEST FIXTURE corpus", chunks, vectors)


def test_bm25_retrieves_exact_rule_number_and_deadline() -> None:
    kb = _knowledge_base()
    results = retrieve_bm25(
        "rule 47-C deadline 2028-09-30",
        kb,
        RAGSettings(top_k=1, candidate_count=3),
    )
    assert results[0].chunk_id == "chunk-exact"
    assert results[0].document_id == "doc-b"
    assert results[0].page_number == 7
    assert results[0].retrieval_method == "bm25"
    assert results[0].diagnostics["score_kind"] == "bm25_score"


def test_dense_baseline_remains_available_and_hybrid_rrf_uses_ranks() -> None:
    kb = _knowledge_base()
    settings = RAGSettings(top_k=1, candidate_count=3, rrf_constant=60)
    query = "rule 47-C deadline 2028-09-30"
    dense = retrieve(query, kb, QueryEmbedder(), settings)
    hybrid = retrieve_hybrid(query, kb, QueryEmbedder(), settings)
    assert dense[0].chunk_id == "chunk-semantic"
    assert dense[0].retrieval_method == "dense"
    assert hybrid[0].chunk_id == "chunk-exact"
    assert hybrid[0].document_id == "doc-b"
    assert hybrid[0].page_number == 7
    assert hybrid[0].retrieval_method == "hybrid_rrf"
    assert hybrid[0].diagnostics["score_kind"] == "reciprocal_rank_fusion_score"
    assert hybrid[0].diagnostics["dense_rank"] == 3
    assert hybrid[0].diagnostics["bm25_rank"] == 1
    assert hybrid[0].score == 1 / 63 + 1 / 61


def test_configured_pipeline_can_select_hybrid_with_consistent_structure() -> None:
    settings = RAGSettings(retrieval_method="hybrid", top_k=2, candidate_count=3)
    result = retrieve_configured("rule 47-C", _knowledge_base(), QueryEmbedder(), settings)
    assert result
    assert all(row.retrieval_method == "hybrid_rrf" for row in result)
    assert all(row.document_id and row.source_name and row.page_number > 0 and row.chunk_id and row.text for row in result)


def test_bm25_can_be_used_as_an_ablation_without_dense_fusion() -> None:
    kb = _knowledge_base()
    result = retrieve_bm25("47-C", kb, top_k=2)
    assert result[0].retrieval_method == "bm25"
    assert result[0].chunk_id == "chunk-exact"


def test_rrf_keeps_identically_named_chunks_from_separate_documents_distinct() -> None:
    kb = KnowledgeBase(
        "SYNTHETIC TEST FIXTURE multi-document corpus",
        (
            TextChunk(1, "SYNTHETIC TEST FIXTURE code 91 alpha", "one.pdf", "doc-one", "same-chunk-id"),
            TextChunk(2, "SYNTHETIC TEST FIXTURE general information", "two.pdf", "doc-two", "same-chunk-id"),
        ),
        np.array([[0.8, 0.6], [1.0, 0.0]], dtype=np.float32),
    )
    results = retrieve_hybrid("code 91", kb, QueryEmbedder(), RAGSettings(top_k=2, candidate_count=2))
    assert {result.document_id for result in results} == {"doc-one", "doc-two"}
