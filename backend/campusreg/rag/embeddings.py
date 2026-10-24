"""Embedding interfaces and lazy SentenceTransformer integration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np


class EmbeddingModel(Protocol):
    """Interface used by the core, allowing tests to inject a small fake."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return a two-dimensional matrix with one vector per input text."""


class SentenceTransformerEmbedder:
    """Adapter for SentenceTransformer using normalized dense embeddings."""

    def __init__(self, model_name: str, model: Any | None = None) -> None:
        self.model_name = model_name
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Install campusreg-rag with the [embeddings] extra to use SentenceTransformer."
                ) from exc
            model = SentenceTransformer(model_name)
        self._model = model

    @property
    def dimension(self) -> int | None:
        """Return the model's declared output width when the adapter exposes it."""

        getter = getattr(self._model, "get_sentence_embedding_dimension", None)
        if not callable(getter):
            return None
        value = getter()
        return int(value) if value is not None else None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)


def normalize_embedding_matrix(vectors: np.ndarray, expected_rows: int) -> np.ndarray:
    """Validate and unit-normalize rows so dot products equal cosine scores."""

    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != expected_rows or matrix.shape[1] == 0:
        raise ValueError("embedding matrix must have one non-empty vector per input text")
    if not np.isfinite(matrix).all():
        raise ValueError("embedding matrix contains non-finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embedding vectors must have non-zero length")
    return matrix / norms


def normalize_query_vector(vectors: np.ndarray, expected_dimension: int) -> np.ndarray:
    """Validate and unit-normalize a single query embedding."""

    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim == 2 and matrix.shape[0] == 1:
        matrix = matrix[0]
    if matrix.ndim != 1 or matrix.shape[0] != expected_dimension:
        raise ValueError("query embedding dimension does not match the knowledge base")
    if not np.isfinite(matrix).all():
        raise ValueError("query embedding contains non-finite values")
    norm = float(np.linalg.norm(matrix))
    if norm == 0:
        raise ValueError("query embedding must have non-zero length")
    return matrix / norm
