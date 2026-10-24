"""Retrieval evaluation tests use synthetic fixtures and no policy data."""

from __future__ import annotations

import csv
import json

import numpy as np

from campusreg.config import RAGSettings
from campusreg.evaluation.hybrid_comparison import compare_retrieval_methods
from campusreg.evaluation.models import EvaluationQuestion, ExpectedPage
from campusreg.rag.models import KnowledgeBase, TextChunk


class FakeEmbedder:
    def encode(self, texts):
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


def test_dense_bm25_hybrid_ablation_records_same_heldout_question(tmp_path) -> None:
    kb = KnowledgeBase(
        "SYNTHETIC TEST FIXTURE corpus",
        (
            TextChunk(1, "SYNTHETIC TEST FIXTURE broad overview", "other.pdf", "other-doc", "other-chunk"),
            TextChunk(4, "SYNTHETIC TEST FIXTURE rule 47-C deadline 2028-09-30", "specific.pdf", "target-doc", "target-chunk"),
        ),
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    question = EvaluationQuestion(
        id="SYNTHETIC-heldout-question",
        question="rule 47-C deadline 2028-09-30",
        category="direct_factual",
        expected_answer="SYNTHETIC TEST FIXTURE expected answer",
        answerable=True,
        expected_document_ids=("target-doc",),
        expected_pages=(ExpectedPage("target-doc", 4),),
        expected_conflict=False,
        split="test",
    )

    result = compare_retrieval_methods(
        [question], kb, embedder=FakeEmbedder(), settings=RAGSettings(top_k=1),
        output_dir=tmp_path, corpus_revision="synthetic-corpus-v1", candidate_count=2,
    )

    assert set(result["summaries"]) == {"dense", "bm25", "hybrid_rrf"}
    assert all(summary["question_count"] == 1 for summary in result["summaries"].values())
    dense_row = next(row for row in result["records"] if row["method"] == "dense")
    hybrid_row = next(row for row in result["records"] if row["method"] == "hybrid_rrf")
    assert dense_row["retrieved_sources"][0]["document_id"] == "other-doc"
    assert hybrid_row["retrieved_sources"][0]["document_id"] == "target-doc"
    assert hybrid_row["retrieved_sources"][0]["page_number"] == 4
    assert hybrid_row["retrieved_sources"][0]["diagnostics"]["score_kind"] == "reciprocal_rank_fusion_score"
    assert len(list(csv.DictReader(result["csv_path"].open()))) == 3
    configuration = json.loads(result["config_path"].read_text())
    assert configuration["selection_or_tuning"].startswith("none")
    assert "raw cosine and BM25 scores are not added" in configuration["score_combination"]


def test_comparison_refuses_to_repurpose_heldout_as_validation(tmp_path) -> None:
    kb = KnowledgeBase(
        "SYNTHETIC TEST FIXTURE corpus",
        (TextChunk(1, "SYNTHETIC TEST FIXTURE text", "doc.pdf", "doc", "chunk"),),
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    question = EvaluationQuestion(
        id="synthetic-dev", question="SYNTHETIC TEST FIXTURE q", category="direct_factual",
        expected_answer="SYNTHETIC TEST FIXTURE expected", answerable=True,
        expected_document_ids=("doc",), expected_pages=(), expected_conflict=False, split="development",
    )
    try:
        compare_retrieval_methods(
            [question], kb, embedder=FakeEmbedder(), settings=RAGSettings(), output_dir=tmp_path,
            corpus_revision="synthetic", split="development",
        )
    except ValueError as exc:
        assert "split='test'" in str(exc)
    else:
        raise AssertionError("held-out comparison must reject non-test split")
