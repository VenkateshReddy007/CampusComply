"""Typed runtime settings for the shared CampusReg RAG implementation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class RAGSettings:
    """RAG parameters, with defaults matching the starter notebook baseline."""

    gemma_model: str = "gemma-4-26b-a4b-it"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 120
    chunk_overlap: int = 30
    top_k: int = 3
    temperature: float = 0.2
    relevance_threshold: float | None = None
    max_pdf_size_bytes: int = 20 * 1024 * 1024
    retrieval_method: str = "dense"
    candidate_count: int = 20
    rrf_constant: int = 60
    corpus_revision: str = ""

    def __post_init__(self) -> None:
        if not self.gemma_model.strip():
            raise ValueError("gemma_model must not be empty")
        if not self.embedding_model.strip():
            raise ValueError("embedding_model must not be empty")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be non-negative and less than chunk_size")
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if self.relevance_threshold is not None and not -1.0 <= self.relevance_threshold <= 1.0:
            raise ValueError("relevance_threshold must be between -1.0 and 1.0")
        if self.max_pdf_size_bytes <= 0:
            raise ValueError("max_pdf_size_bytes must be greater than zero")
        if self.retrieval_method not in {"dense", "hybrid"}:
            raise ValueError("retrieval_method must be 'dense' or 'hybrid'")
        if self.candidate_count <= 0:
            raise ValueError("candidate_count must be greater than zero")
        if self.rrf_constant <= 0:
            raise ValueError("rrf_constant must be greater than zero")

    @classmethod
    def baseline(cls) -> "RAGSettings":
        """Return the exact configurable values used by the Colab baseline."""

        return cls()

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "RAGSettings":
        """Load RAG options from environment variables without reading secrets."""

        values = os.environ if environ is None else environ

        def _int(name: str, default: int) -> int:
            raw = values.get(name)
            if raw is None or not raw.strip():
                return default
            try:
                return int(raw)
            except ValueError as exc:
                raise ValueError(f"{name} must be an integer") from exc

        def _float(name: str, default: float) -> float:
            raw = values.get(name)
            if raw is None or not raw.strip():
                return default
            try:
                return float(raw)
            except ValueError as exc:
                raise ValueError(f"{name} must be a number") from exc

        raw_threshold = values.get("RELEVANCE_THRESHOLD")
        threshold = None
        if raw_threshold is not None and raw_threshold.strip():
            try:
                threshold = float(raw_threshold)
            except ValueError as exc:
                raise ValueError("RELEVANCE_THRESHOLD must be a number") from exc

        baseline = cls.baseline()
        return cls(
            gemma_model=values.get("GEMMA_MODEL", baseline.gemma_model),
            embedding_model=values.get("EMBED_MODEL_NAME", baseline.embedding_model),
            chunk_size=_int("CHUNK_SIZE", baseline.chunk_size),
            chunk_overlap=_int("CHUNK_OVERLAP", baseline.chunk_overlap),
            top_k=_int("TOP_K", baseline.top_k),
            temperature=_float("TEMPERATURE", baseline.temperature),
            relevance_threshold=threshold,
            max_pdf_size_bytes=_int("MAX_PDF_SIZE_BYTES", baseline.max_pdf_size_bytes),
            retrieval_method=values.get("RETRIEVAL_METHOD", baseline.retrieval_method),
            candidate_count=_int("RETRIEVAL_CANDIDATE_COUNT", baseline.candidate_count),
            rrf_constant=_int("RRF_CONSTANT", baseline.rrf_constant),
            corpus_revision=values.get("CORPUS_REVISION", baseline.corpus_revision),
        )
