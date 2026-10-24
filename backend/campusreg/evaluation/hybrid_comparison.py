"""Retrieval-only dense/BM25/RRF comparison on a fixed held-out question set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from campusreg.config import RAGSettings
from campusreg.evaluation.models import EvaluationQuestion
from campusreg.rag.embeddings import EmbeddingModel, SentenceTransformerEmbedder
from campusreg.rag.models import DocumentPage, KnowledgeBase, RetrievedSource
from campusreg.rag.retrieval import (
    BM25_B,
    BM25_K1,
    build_knowledge_base,
    retrieve_bm25,
    retrieve_hybrid,
)

METHODS = ("dense", "bm25", "hybrid_rrf")


class HybridComparisonError(ValueError):
    pass


def _relevant(question: EvaluationQuestion, results: Sequence[RetrievedSource]) -> tuple[float | None, float | None]:
    if not question.answerable:
        return None, None
    expected_pages = {(page.document_id, page.page_number) for page in question.expected_pages}
    expected_docs = set(question.expected_document_ids)
    if expected_pages:
        found = [(source.document_id, source.page_number) in expected_pages for source in results]
        total = len(expected_pages)
    else:
        found = [source.document_id in expected_docs for source in results]
        total = len(expected_docs)
    if total == 0:
        raise HybridComparisonError(f"answerable question {question.id!r} has no expected source labels")
    recall = sum(found) / total
    first = next((index for index, is_relevant in enumerate(found, start=1) if is_relevant), None)
    reciprocal_rank = 1.0 / first if first else 0.0
    return recall, reciprocal_rank


def _source(source: RetrievedSource) -> dict[str, Any]:
    return {
        "document_id": source.document_id,
        "document_name": source.source_name,
        "page_number": source.page_number,
        "chunk_id": source.chunk_id,
        "text": source.text,
        "retrieval_method": source.retrieval_method,
        "retrieval_score": source.score,
        "diagnostics": dict(source.diagnostics),
    }


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if len(values) < 5:
        return None
    ordered = sorted(values)
    index = max(0, int(np.ceil(percentile * len(ordered))) - 1)
    return float(ordered[index])


def compare_retrieval_methods(
    questions: Sequence[EvaluationQuestion],
    knowledge_base: KnowledgeBase,
    *,
    embedder: EmbeddingModel,
    settings: RAGSettings,
    output_dir: str | Path,
    corpus_revision: str,
    split: str = "test",
    candidate_count: int = 20,
    rrf_constant: int = 60,
    indexing_time_ms: float | None = None,
) -> dict[str, Any]:
    """Run fixed dense, BM25, and RRF retrieval ablations without generation."""
    if split != "test":
        raise HybridComparisonError("held-out retrieval comparison requires split='test'")
    selected = [question for question in questions if question.split == split]
    if not selected:
        raise HybridComparisonError("selected held-out split is empty")
    if candidate_count <= 0 or rrf_constant <= 0:
        raise ValueError("candidate_count and rrf_constant must be positive")
    if not corpus_revision.strip():
        raise HybridComparisonError("record the exact source-corpus revision")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    latency_by_method: dict[str, list[float]] = {method: [] for method in METHODS}
    metric_values: dict[str, dict[str, list[float]]] = {
        method: {"recall": [], "mrr": []} for method in METHODS
    }
    shared_config = {
        "split": "test",
        "question_ids": [question.id for question in selected],
        "corpus_revision": corpus_revision,
        "embedding_model": settings.embedding_model,
        "embedding_dimension": int(knowledge_base.vectors.shape[1]),
        "chunk_size_words": settings.chunk_size,
        "chunk_overlap_words": settings.chunk_overlap,
        "top_k": settings.top_k,
        "candidate_count": candidate_count,
        "dense_method": "normalized cosine similarity",
        "keyword_method": {"name": "BM25", "k1": BM25_K1, "b": BM25_B, "tokenizer": "Unicode case-folded word tokens"},
        "fusion_method": {"name": "Reciprocal Rank Fusion", "constant": rrf_constant,
                          "formula": "sum(1 / (rrf_constant + one_based_rank))"},
        "score_combination": "rank fusion only; raw cosine and BM25 scores are not added",
        "indexing_time_ms": indexing_time_ms,
        "vector_index_bytes": int(knowledge_base.vectors.nbytes),
        "selection_or_tuning": "none; fixed methods and parameters are compared on held-out data",
    }
    # The index is supplied by the caller and identical across retrieval methods.
    for method in METHODS:
        method_settings = replace(
            settings,
            retrieval_method="hybrid" if method == "hybrid_rrf" else "dense",
            relevance_threshold=None,
            candidate_count=candidate_count,
            rrf_constant=rrf_constant,
        )
        for question in selected:
            started = perf_counter()
            try:
                if method == "bm25":
                    found = retrieve_bm25(
                        question.question, knowledge_base, method_settings,
                        top_k=settings.top_k, candidate_count=candidate_count,
                    )
                elif method == "hybrid_rrf":
                    found = retrieve_hybrid(
                        question.question, knowledge_base, embedder, method_settings,
                        top_k=settings.top_k, candidate_count=candidate_count,
                        rrf_constant=rrf_constant,
                    )
                else:
                    # Import locally to keep the dense baseline explicit.
                    from campusreg.rag.retrieval import retrieve
                    found = retrieve(question.question, knowledge_base, embedder, method_settings, top_k=settings.top_k)
                elapsed_ms = (perf_counter() - started) * 1000.0
                recall, mrr = _relevant(question, found)
                if recall is not None and mrr is not None:
                    metric_values[method]["recall"].append(recall)
                    metric_values[method]["mrr"].append(mrr)
                latency_by_method[method].append(elapsed_ms)
                records.append({
                    "method": method,
                    "question_id": question.id,
                    "question": question.question,
                    "split": question.split,
                    "status": "ok",
                    "recall_at_k": recall,
                    "mrr_at_k": mrr,
                    "retrieval_latency_ms": elapsed_ms,
                    "retrieved_sources": [_source(source) for source in found],
                    "error_code": None,
                })
            except Exception as exc:
                elapsed_ms = (perf_counter() - started) * 1000.0
                latency_by_method[method].append(elapsed_ms)
                records.append({
                    "method": method,
                    "question_id": question.id,
                    "question": question.question,
                    "split": question.split,
                    "status": "error",
                    "recall_at_k": None,
                    "mrr_at_k": None,
                    "retrieval_latency_ms": elapsed_ms,
                    "retrieved_sources": [],
                    "error_code": type(exc).__name__,
                })

    summaries: dict[str, Any] = {}
    for method in METHODS:
        recalls = metric_values[method]["recall"]
        mrrs = metric_values[method]["mrr"]
        latencies = latency_by_method[method]
        summaries[method] = {
            "question_count": len(selected),
            "successful_question_count": sum(row["method"] == method and row["status"] == "ok" for row in records),
            "failed_question_count": sum(row["method"] == method and row["status"] == "error" for row in records),
            "recall_at_k": mean(recalls) if recalls else None,
            "recall_n": len(recalls),
            "mrr_at_k": mean(mrrs) if mrrs else None,
            "mrr_n": len(mrrs),
            "retrieval_latency_mean_ms": mean(latencies) if latencies else None,
            "retrieval_latency_median_ms": median(latencies) if latencies else None,
            "retrieval_latency_p95_ms": _percentile(latencies, 0.95),
        }

    config_path = root / "hybrid_experiment_config.json"
    config_path.write_text(json.dumps({**shared_config, "methods": list(METHODS)}, indent=2), encoding="utf-8")
    results_path = root / "hybrid_retrieval_results.jsonl"
    results_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records), encoding="utf-8")
    csv_path = root / "hybrid_retrieval_results.csv"
    columns = ["method", "question_id", "question", "split", "status", "recall_at_k", "mrr_at_k",
               "retrieval_latency_ms", "retrieved_sources_json", "error_code"]
    with csv_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=columns)
        writer.writeheader()
        for row in records:
            writer.writerow({
                "method": row["method"],
                "question_id": row["question_id"],
                "question": row["question"],
                "split": row["split"],
                "status": row["status"],
                "recall_at_k": row["recall_at_k"],
                "mrr_at_k": row["mrr_at_k"],
                "retrieval_latency_ms": row["retrieval_latency_ms"],
                "retrieved_sources_json": json.dumps(row["retrieved_sources"], ensure_ascii=False),
                "error_code": row["error_code"],
            })
    report_lines = [
        "# Dense and hybrid retrieval comparison", "",
        "This retrieval-only experiment uses the same fixed held-out questions, corpus, embedding index, chunking, and top-k. No generation runs. Dense, BM25, and dense+BM25 RRF are shown as separate ablations.", "",
        f"- Corpus revision: `{corpus_revision}`", f"- Held-out question count: {len(selected)}",
        f"- top_k: {settings.top_k}; candidate_count: {candidate_count}; RRF constant: {rrf_constant}", "",
        "| Method | Recall@k | MRR@k | Mean latency (ms) | Median latency (ms) | p95 latency (ms) | Failures |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = summaries[method]
        report_lines.append(
            f"| {method} | {item['recall_at_k']} (n={item['recall_n']}) | {item['mrr_at_k']} (n={item['mrr_n']}) | "
            f"{item['retrieval_latency_mean_ms']} | {item['retrieval_latency_median_ms']} | {item['retrieval_latency_p95_ms']} | {item['failed_question_count']} |"
        )
    report_lines.extend(["", "Only measured values are shown. Percentile latency is omitted when fewer than five timings exist. No method is declared better automatically; compare the primary held-out metrics and their sample counts. RRF combines ranks, not uncalibrated cosine and BM25 scores.", ""])
    report_path = root / "hybrid_retrieval_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return {"summaries": summaries, "records": records, "config_path": config_path,
            "results_path": results_path, "csv_path": csv_path, "report_path": report_path}


class _SmokeEmbedder:
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


def _smoke(output_dir: str | Path) -> None:
    from campusreg.evaluation.models import EvaluationQuestion, ExpectedPage
    from campusreg.rag.models import TextChunk

    chunks = (
        TextChunk(1, "SYNTHETIC TEST FIXTURE general topic", "synthetic-a.pdf", "synthetic-a", "a"),
        TextChunk(4, "SYNTHETIC TEST FIXTURE rule 47-C deadline 2028-09-30", "synthetic-b.pdf", "synthetic-b", "b"),
    )
    kb = KnowledgeBase("SYNTHETIC TEST FIXTURE", chunks, np.array([[1., 0.], [0., 1.]], dtype=np.float32))
    q = EvaluationQuestion(
        id="synthetic-hybrid-q", question="47-C deadline 2028-09-30",
        category="direct_factual", expected_answer="SYNTHETIC TEST FIXTURE expected",
        answerable=True, expected_document_ids=("synthetic-b",),
        expected_pages=(ExpectedPage("synthetic-b", 4),), expected_conflict=False, split="test",
    )
    compare_retrieval_methods([q], kb, embedder=_SmokeEmbedder(), settings=RAGSettings(top_k=1),
                              output_dir=output_dir, corpus_revision="synthetic-only", candidate_count=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--corpus-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results/hybrid-comparison"))
    parser.add_argument("--candidate-count", type=int, default=20)
    parser.add_argument("--rrf-constant", type=int, default=60)
    args = parser.parse_args()
    if args.smoke:
        _smoke(args.output_dir)
        print("dense/BM25/RRF retrieval smoke passed (synthetic held-out fixture)")
        return
    if args.dataset is None or args.corpus_manifest is None:
        parser.error("--full requires --dataset and --corpus-manifest")
    from campusreg.evaluation.dataset import load_dataset
    from campusreg.evaluation.retrieval_experiments import load_pdf_manifest

    from campusreg.evaluation.dataset import validate_challenge_dataset
    validate_challenge_dataset(args.dataset)
    questions = load_dataset(args.dataset, split="test")
    pages, revisions = load_pdf_manifest(args.corpus_manifest)
    actual_documents = {page.document_id for page in pages}
    actual_pages = {(page.document_id, page.page_number) for page in pages}
    expected_documents = {doc_id for question in questions for doc_id in question.expected_document_ids}
    expected_pages = {
        (page.document_id, page.page_number)
        for question in questions for page in question.expected_pages
    }
    if expected_documents - actual_documents or expected_pages - actual_pages:
        raise HybridComparisonError("held-out question labels reference sources/pages missing from the verified corpus")
    settings = RAGSettings.from_env()
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    started = perf_counter()
    kb = build_knowledge_base(pages, "verified held-out corpus", embedder, settings)
    indexing_ms = (perf_counter() - started) * 1000.0
    result = compare_retrieval_methods(
        questions, kb, embedder=embedder, settings=settings, output_dir=args.output_dir,
        corpus_revision=hashlib.sha256(json.dumps(revisions, sort_keys=True).encode()).hexdigest(),
        candidate_count=args.candidate_count, rrf_constant=args.rrf_constant,
        indexing_time_ms=indexing_ms,
    )
    print(f"hybrid comparison complete for {len(questions)} held-out questions")


if __name__ == "__main__":
    main()
