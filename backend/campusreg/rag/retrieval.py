"""Knowledge-base construction and dense cosine-similarity retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from collections import Counter
from dataclasses import replace
import math
import re
from uuid import uuid4

import numpy as np

from campusreg.config import RAGSettings
from campusreg.rag.chunking import split_into_chunks
from campusreg.rag.embeddings import (
    EmbeddingModel,
    normalize_embedding_matrix,
    normalize_query_vector,
)
from campusreg.rag.errors import EmptyDocumentError
from campusreg.rag.models import DocumentPage, KnowledgeBase, RetrievedSource
from campusreg.rag.privacy import redact_pages, redact_text

_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
BM25_K1 = 1.5
BM25_B = 0.75


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.casefold())


def build_knowledge_base(
    pages: Sequence[DocumentPage | str],
    name: str,
    embedder: EmbeddingModel,
    settings: RAGSettings | None = None,
    *,
    document_id: str | None = None,
) -> KnowledgeBase:
    """Chunk pages and build an in-memory normalized dense-vector index."""

    selected = settings or RAGSettings.baseline()
    selected_document_id = document_id or next(
        (page.document_id for page in pages if isinstance(page, DocumentPage) and page.document_id),
        str(uuid4()),
    )
    safe_pages, _ = redact_pages(
        tuple(
            page if isinstance(page, DocumentPage) else DocumentPage(index, page, name, selected_document_id)
            for index, page in enumerate(pages, start=1)
        )
    )
    chunks = split_into_chunks(
        safe_pages,
        settings=selected,
        source_name=name,
        document_id=selected_document_id,
    )
    if not chunks:
        raise EmptyDocumentError("No extractable text was long enough to index.")
    vectors = normalize_embedding_matrix(
        embedder.encode([chunk.text for chunk in chunks]),
        expected_rows=len(chunks),
    )
    return KnowledgeBase(name=name, chunks=tuple(chunks), vectors=vectors)


def retrieve(
    question: str,
    knowledge_base: KnowledgeBase,
    embedder: EmbeddingModel | None,
    settings: RAGSettings | None = None,
    *,
    top_k: int | None = None,
    relevance_threshold: float | None = None,
    query_vector: np.ndarray | None = None,
) -> list[RetrievedSource]:
    """Rank chunks by cosine similarity, optionally using a precomputed query vector."""

    selected = settings or RAGSettings.baseline()
    limit = selected.top_k if top_k is None else top_k
    threshold = selected.relevance_threshold if relevance_threshold is None else relevance_threshold
    if limit <= 0:
        raise ValueError("top_k must be greater than zero")
    question, _ = redact_text(question)
    if not question.strip():
        raise ValueError("question must not be empty")

    if query_vector is None:
        if embedder is None:
            raise ValueError("embedder is required when query_vector is not supplied")
        raw_query = embedder.encode([question])
    else:
        raw_query = query_vector
    query = normalize_query_vector(raw_query, expected_dimension=knowledge_base.vectors.shape[1])
    similarities = knowledge_base.vectors @ query
    best_indices = np.argsort(similarities)[::-1][:limit]
    results = [
        RetrievedSource(
            page=knowledge_base.chunks[index].page,
            text=knowledge_base.chunks[index].text,
            source_name=knowledge_base.chunks[index].source_name,
            score=float(similarities[index]),
            document_id=knowledge_base.chunks[index].document_id,
            chunk_id=knowledge_base.chunks[index].chunk_id,
            retrieval_method="dense",
            diagnostics={"score_kind": "cosine_similarity"},
        )
        for index in best_indices
        if threshold is None or float(similarities[index]) >= threshold
    ]
    return results


def retrieve_bm25(
    question: str,
    knowledge_base: KnowledgeBase,
    settings: RAGSettings | None = None,
    *,
    top_k: int | None = None,
    candidate_count: int | None = None,
) -> list[RetrievedSource]:
    """Rank indexed chunks by BM25 keyword relevance (k1=1.5, b=0.75)."""
    selected = settings or RAGSettings.baseline()
    limit = selected.top_k if top_k is None else top_k
    candidates = selected.candidate_count if candidate_count is None else candidate_count
    if limit <= 0 or candidates <= 0:
        raise ValueError("top_k and candidate_count must be positive")
    question, _ = redact_text(question)
    if not question.strip():
        raise ValueError("question must not be empty")
    query_terms = _tokens(question)
    if not query_terms or not knowledge_base.chunks:
        return []

    tokenized = [_tokens(chunk.text) for chunk in knowledge_base.chunks]
    lengths = [len(tokens) for tokens in tokenized]
    average_length = sum(lengths) / len(lengths) if lengths else 0.0
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))
    scores: list[float] = []
    count = len(tokenized)
    for tokens, length in zip(tokenized, lengths):
        frequencies = Counter(tokens)
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if not frequency:
                continue
            df = document_frequency[term]
            inverse_frequency = math.log1p((count - df + 0.5) / (df + 0.5))
            length_norm = 1.0 - BM25_B + BM25_B * length / average_length if average_length else 1.0
            score += inverse_frequency * (
                frequency * (BM25_K1 + 1.0) / (frequency + BM25_K1 * length_norm)
            )
        scores.append(score)
    ranked = sorted(range(count), key=lambda index: (-scores[index], index))
    ranked = [index for index in ranked if scores[index] > 0][:candidates]
    output = []
    for rank, index in enumerate(ranked, start=1):
        chunk = knowledge_base.chunks[index]
        output.append(RetrievedSource(
            page=chunk.page,
            text=chunk.text,
            source_name=chunk.source_name,
            score=float(scores[index]),
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            retrieval_method="bm25",
            diagnostics={
                "score_kind": "bm25_score",
                "bm25_rank": rank,
                "bm25_k1": BM25_K1,
                "bm25_b": BM25_B,
            },
        ))
        if len(output) >= limit:
            break
    return output


def retrieve_hybrid(
    question: str,
    knowledge_base: KnowledgeBase,
    embedder: EmbeddingModel | None,
    settings: RAGSettings | None = None,
    *,
    top_k: int | None = None,
    candidate_count: int | None = None,
    rrf_constant: int | None = None,
    query_vector: np.ndarray | None = None,
) -> list[RetrievedSource]:
    """Fuse dense and BM25 ranks with RRF; raw scores are never added together."""
    selected = settings or RAGSettings.baseline()
    limit = selected.top_k if top_k is None else top_k
    count = selected.candidate_count if candidate_count is None else candidate_count
    constant = selected.rrf_constant if rrf_constant is None else rrf_constant
    if limit <= 0 or count <= 0 or constant <= 0:
        raise ValueError("top_k, candidate_count, and rrf_constant must be positive")

    dense = retrieve(
        question,
        knowledge_base,
        embedder,
        replace(selected, relevance_threshold=None),
        top_k=count,
        relevance_threshold=None,
        query_vector=query_vector,
    )
    sparse = retrieve_bm25(question, knowledge_base, selected, top_k=count, candidate_count=count)
    def identity(source: RetrievedSource) -> tuple[str, str]:
        source_id = source.document_id or source.source_name
        passage_id = source.chunk_id or f"{source.page}|{source.text}"
        return source_id, passage_id

    dense_ranks = {identity(source): (rank, source) for rank, source in enumerate(dense, start=1)}
    sparse_ranks = {identity(source): (rank, source) for rank, source in enumerate(sparse, start=1)}
    fused: list[tuple[float, int, RetrievedSource]] = []
    order = {
        (
            chunk.document_id or chunk.source_name,
            chunk.chunk_id or f"{chunk.page}|{chunk.text}",
        ): index
        for index, chunk in enumerate(knowledge_base.chunks)
    }
    for chunk_key in dense_ranks.keys() | sparse_ranks.keys():
        dense_item = dense_ranks.get(chunk_key)
        sparse_item = sparse_ranks.get(chunk_key)
        rrf_score = (1.0 / (constant + dense_item[0]) if dense_item else 0.0) + (
            1.0 / (constant + sparse_item[0]) if sparse_item else 0.0
        )
        source = dense_item[1] if dense_item else sparse_item[1]
        diagnostics: dict[str, float | int | str | None] = {
            "score_kind": "reciprocal_rank_fusion_score",
            "dense_rank": dense_item[0] if dense_item else None,
            "dense_cosine_similarity": dense_item[1].score if dense_item else None,
            "bm25_rank": sparse_item[0] if sparse_item else None,
            "bm25_score": sparse_item[1].score if sparse_item else None,
            "rrf_constant": constant,
            "candidate_count": count,
        }
        fused.append((rrf_score, order.get(chunk_key, len(order)), RetrievedSource(
            page=source.page,
            text=source.text,
            source_name=source.source_name,
            score=rrf_score,
            document_id=source.document_id,
            chunk_id=source.chunk_id,
            retrieval_method="hybrid_rrf",
            diagnostics=diagnostics,
        )))
    fused.sort(key=lambda item: (-item[0], item[1]))
    threshold = selected.relevance_threshold
    return [
        source for score, _, source in fused[:limit]
        if threshold is None or score >= threshold
    ]


def retrieve_configured(
    question: str,
    knowledge_base: KnowledgeBase,
    embedder: EmbeddingModel,
    settings: RAGSettings,
    *,
    query_vector: np.ndarray | None = None,
) -> list[RetrievedSource]:
    """Dispatch to dense baseline or configured dense+BM25 hybrid retrieval."""
    if settings.retrieval_method == "hybrid":
        return retrieve_hybrid(question, knowledge_base, embedder, settings, query_vector=query_vector)
    return retrieve(question, knowledge_base, embedder, settings, query_vector=query_vector)
