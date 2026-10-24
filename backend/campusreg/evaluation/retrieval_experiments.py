"""Resumable retrieval-only experiments over reviewed development/validation data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import numpy as np

from campusreg.config import RAGSettings
from campusreg.evaluation.dataset import load_dataset, validate_challenge_dataset
from campusreg.evaluation.metrics import compute_metrics
from campusreg.evaluation.models import EVALUATION_SPLITS, EvaluationQuestion
from campusreg.rag.documents import read_pdf_bytes
from campusreg.rag.embeddings import (
    EmbeddingModel,
    SentenceTransformerEmbedder,
    normalize_query_vector,
)
from campusreg.rag.models import DocumentPage, KnowledgeBase, RetrievedSource, TextChunk
from campusreg.rag.retrieval import build_knowledge_base, retrieve

DEFAULT_EMBEDDING_MODELS = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
)
DEFAULT_CHUNK_SIZES = (80, 120, 200)
DEFAULT_CHUNK_OVERLAP = 30
DEFAULT_TOP_KS = (1, 3, 5)
METRIC_CUTOFFS = (1, 3, 5)
CHUNKING_REVISION = "word-page-window-v1"
RETRIEVAL_REVISION = "normalized-cosine-full-sort-v1"
EXPERIMENT_REVISION = "campusreg-retrieval-experiments-v1"
SELECTION_RULE: dict[str, Any] = {
    "primary": {"metric": "recall_at_3", "direction": "max"},
    "secondary": {"metric": "mrr_at_3", "direction": "max"},
    "tie_breakers": [
        {"metric": "retrieval_latency_mean_ms", "direction": "min"},
        {"metric": "vector_index_bytes", "direction": "min"},
        {"metric": "experiment_id", "direction": "asc"},
    ],
    "eligible_rows": "complete validation rows with at least one answerable question",
}

_RESULT_FIELDS = (
    "experiment_id",
    "split",
    "status",
    "is_baseline",
    "embedding_model",
    "model_dimension",
    "chunk_size_words",
    "chunk_overlap_words",
    "top_k",
    "dataset_fingerprint",
    "corpus_fingerprint",
    "question_count",
    "answerable_question_count",
    "failed_question_count",
    "failed_question_rate",
    "recall_at_1",
    "recall_at_1_n",
    "mrr_at_1",
    "mrr_at_1_n",
    "recall_at_3",
    "recall_at_3_n",
    "mrr_at_3",
    "mrr_at_3_n",
    "recall_at_5",
    "recall_at_5_n",
    "mrr_at_5",
    "mrr_at_5_n",
    "indexing_time_ms",
    "index_cache_hit",
    "query_embedding_latency_mean_ms",
    "vector_search_latency_mean_ms",
    "retrieval_latency_mean_ms",
    "retrieval_latency_p50_ms",
    "retrieval_latency_p95_ms",
    "embedding_model_parameter_bytes",
    "vector_index_bytes",
)


class RetrievalExperimentError(ValueError):
    """Raised when the experiment inputs cannot support a reproducible run."""


@dataclass(frozen=True, slots=True)
class RetrievalExperimentConfig:
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int

    def __post_init__(self) -> None:
        if not self.embedding_model.strip():
            raise ValueError("embedding_model must not be empty")
        if self.chunk_size <= 0 or self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk size must be positive and overlap must be smaller than chunk size")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")

    @property
    def experiment_id(self) -> str:
        known_aliases = {
            DEFAULT_EMBEDDING_MODELS[0]: "minilm_l6_v2",
            DEFAULT_EMBEDDING_MODELS[1]: "bge_small_en_v1_5",
        }
        alias = known_aliases.get(self.embedding_model)
        if alias is None:
            alias = re.sub(r"[^a-zA-Z0-9]+", "_", self.embedding_model).strip("_").lower()
        return f"{alias}-c{self.chunk_size}-o{self.chunk_overlap}-k{self.top_k}"

    @property
    def is_baseline(self) -> bool:
        baseline = RAGSettings.baseline()
        return (
            self.embedding_model == baseline.embedding_model
            and self.chunk_size == baseline.chunk_size
            and self.chunk_overlap == baseline.chunk_overlap
            and self.top_k == baseline.top_k
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "embedding_model": self.embedding_model,
            "chunk_size_words": self.chunk_size,
            "chunk_overlap_words": self.chunk_overlap,
            "top_k": self.top_k,
            "is_baseline": self.is_baseline,
        }


def build_experiment_grid(
    *,
    embedding_models: Sequence[str] = DEFAULT_EMBEDDING_MODELS,
    chunk_sizes: Sequence[int] = DEFAULT_CHUNK_SIZES,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    top_ks: Sequence[int] = DEFAULT_TOP_KS,
) -> tuple[RetrievalExperimentConfig, ...]:
    """Build the full embedding × chunk-size × top-k grid (18 by default)."""

    configs = tuple(
        RetrievalExperimentConfig(model, size, chunk_overlap, top_k)
        for model in embedding_models
        for size in chunk_sizes
        for top_k in top_ks
    )
    if len({config.experiment_id for config in configs}) != len(configs):
        raise ValueError("experiment grid contains duplicate configuration IDs")
    return configs


def load_experiment_plan(path: str | Path) -> tuple[tuple[RetrievalExperimentConfig, ...], dict[str, Any]]:
    """Load the checked-in grid and metric selection rule."""

    try:
        plan = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalExperimentError("unable to read retrieval experiment plan") from exc
    if not isinstance(plan, dict) or plan.get("schema_version") != 1:
        raise RetrievalExperimentError("unsupported retrieval experiment plan")
    try:
        configs = build_experiment_grid(
            embedding_models=plan["embedding_models"],
            chunk_sizes=plan["chunk_sizes_words"],
            chunk_overlap=plan["chunk_overlap_words"],
            top_ks=plan["top_k_values"],
        )
        rule = plan["validation_selection"]
    except (KeyError, TypeError, ValueError) as exc:
        raise RetrievalExperimentError("retrieval experiment plan has invalid grid settings") from exc
    if not isinstance(rule, dict) or not isinstance(rule.get("primary"), dict):
        raise RetrievalExperimentError("retrieval experiment plan needs a validation selection rule")
    return configs, rule


def load_pdf_manifest(path: str | Path) -> tuple[tuple[DocumentPage, ...], tuple[dict[str, Any], ...]]:
    """Read explicitly selected PDFs and verify their SHA-256 entries."""

    manifest_path = Path(path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalExperimentError("unable to read PDF corpus manifest") from exc
    documents = manifest.get("documents") if isinstance(manifest, dict) else None
    if not isinstance(documents, list) or not documents:
        raise RetrievalExperimentError("PDF corpus manifest must contain a non-empty documents array")

    pages: list[DocumentPage] = []
    revisions: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for document in documents:
        if not isinstance(document, dict):
            raise RetrievalExperimentError("each corpus manifest document must be an object")
        document_id = document.get("document_id")
        filename = document.get("filename")
        relative_path = document.get("path")
        expected_hash = document.get("sha256")
        if not all(isinstance(value, str) and value.strip() for value in (document_id, filename, relative_path, expected_hash)):
            raise RetrievalExperimentError("each corpus entry requires document_id, filename, path, and sha256")
        if document_id in seen_ids:
            raise RetrievalExperimentError("corpus manifest has duplicate document IDs")
        seen_ids.add(document_id)
        pdf_path = Path(relative_path)
        if not pdf_path.is_absolute():
            pdf_path = manifest_path.parent / pdf_path
        try:
            content = pdf_path.read_bytes()
        except OSError as exc:
            raise RetrievalExperimentError(f"cannot read corpus PDF {filename!r}") from exc
        actual_hash = hashlib.sha256(content).hexdigest()
        if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash) or actual_hash.lower() != expected_hash.lower():
            raise RetrievalExperimentError(f"SHA-256 mismatch for corpus PDF {filename!r}")
        try:
            document_pages = read_pdf_bytes(content, filename, document_id=document_id)
        except Exception as exc:
            raise RetrievalExperimentError(f"cannot extract text from corpus PDF {filename!r}") from exc
        if not document_pages or not any(page.text.strip() for page in document_pages):
            raise RetrievalExperimentError(f"corpus PDF {filename!r} has no extractable text")
        pages.extend(document_pages)
        revisions.append(
            {
                "document_id": document_id,
                "filename": filename,
                "sha256": actual_hash,
                "page_count": len(document_pages),
            }
        )
    return tuple(pages), tuple(sorted(revisions, key=lambda row: row["document_id"]))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _dataset_fingerprint(questions: Sequence[EvaluationQuestion]) -> str:
    return _fingerprint([asdict(question) for question in questions])


def _corpus_fingerprint(
    pages: Sequence[DocumentPage], source_documents: Sequence[Mapping[str, Any]]
) -> str:
    page_rows = [
        {
            "document_id": page.document_id,
            "source_name": page.source_name,
            "page_number": page.page_number,
            "text": page.text,
        }
        for page in sorted(pages, key=lambda item: (item.document_id, item.page_number))
    ]
    return _fingerprint({"pages": page_rows, "source_documents": list(source_documents)})


def _check_inputs(
    questions: Sequence[EvaluationQuestion], pages: Sequence[DocumentPage]
) -> tuple[list[EvaluationQuestion], str]:
    selected = [question for question in questions if question.split in {"development", "validation"}]
    if not selected:
        raise RetrievalExperimentError("no development or validation questions were supplied")
    if not any(question.split == "development" for question in selected):
        raise RetrievalExperimentError("development split is empty")
    if not any(question.split == "validation" for question in selected):
        raise RetrievalExperimentError("validation split is required for configuration selection")
    if len({question.id for question in selected}) != len(selected):
        raise RetrievalExperimentError("question IDs must be unique across development and validation")
    if not pages:
        raise RetrievalExperimentError("the selected verified source corpus is empty")
    page_keys = [(page.document_id, page.page_number) for page in pages]
    if any(not page.document_id or not page.source_name or page.page_number < 1 for page in pages):
        raise RetrievalExperimentError("every corpus page needs a document ID, filename, and physical page number")
    if len(set(page_keys)) != len(page_keys):
        raise RetrievalExperimentError("corpus pages contain duplicate document/page identities")
    available_pages = set(page_keys)
    corpus_document_ids = {page.document_id for page in pages}
    missing_ids = sorted(
        {
            document_id
            for question in selected
            if question.answerable
            for document_id in question.expected_document_ids
            if document_id not in corpus_document_ids
        }
    )
    if missing_ids:
        raise RetrievalExperimentError(
            "expected source document IDs are missing from the selected corpus: " + ", ".join(missing_ids)
        )
    missing_pages = sorted(
        {
            (page.document_id, page.page_number)
            for question in selected
            if question.answerable
            for page in question.expected_pages
            if (page.document_id, page.page_number) not in available_pages
        }
    )
    if missing_pages:
        formatted = ", ".join(f"{document_id}:p.{page}" for document_id, page in missing_pages)
        raise RetrievalExperimentError("expected PDF pages are missing from the selected corpus: " + formatted)
    return selected, _dataset_fingerprint(selected)


def _model_parameter_bytes(embedder: EmbeddingModel) -> int | None:
    model = getattr(embedder, "_model", embedder)
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return None
    try:
        return sum(int(parameter.numel()) * int(parameter.element_size()) for parameter in parameters())
    except Exception:
        return None


def _declared_dimension(embedder: EmbeddingModel) -> int | None:
    dimension = getattr(embedder, "dimension", None)
    if isinstance(dimension, int) and dimension > 0:
        return dimension
    model = getattr(embedder, "_model", embedder)
    getter = getattr(model, "get_sentence_embedding_dimension", None)
    if callable(getter):
        value = getter()
        return int(value) if isinstance(value, int) and value > 0 else None
    return None


def _index_cache_key(config: RetrievalExperimentConfig, corpus_fingerprint: str) -> str:
    return _fingerprint(
        {
            "embedding_model": config.embedding_model,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "corpus_fingerprint": corpus_fingerprint,
            "chunking_revision": CHUNKING_REVISION,
            "retrieval_revision": RETRIEVAL_REVISION,
        }
    )


def _query_cache_key(question: EvaluationQuestion, model: str, dataset_fingerprint: str) -> str:
    return _fingerprint(
        {
            "question": asdict(question),
            "embedding_model": model,
            "dataset_fingerprint": dataset_fingerprint,
        }
    )


def _observation_cache_key(
    question: EvaluationQuestion,
    config: RetrievalExperimentConfig,
    dataset_fingerprint: str,
    corpus_fingerprint: str,
) -> str:
    return _fingerprint(
        {
            "question": asdict(question),
            "config": config.to_dict(),
            "dataset_fingerprint": dataset_fingerprint,
            "corpus_fingerprint": corpus_fingerprint,
            "experiment_revision": EXPERIMENT_REVISION,
        }
    )


def _atomic_write(path: Path, writer: Callable[[Any], None], *, binary: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        mode = "wb" if binary else "w"
        kwargs = {} if binary else {"encoding": "utf-8", "newline": "\n"}
        with os.fdopen(fd, mode, **kwargs) as target:
            writer(target)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _write_json(path: Path, value: Any) -> None:
    def write(target: Any) -> None:
        json.dump(value, target, ensure_ascii=False, indent=2, sort_keys=True)
        target.write("\n")

    _atomic_write(path, write)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    def write(target: Any) -> None:
        for row in rows:
            target.write(_canonical_json(row) + "\n")

    _atomic_write(path, write)


def _write_numpy(path: Path, vectors: np.ndarray) -> None:
    def write(target: Any) -> None:
        np.save(target, np.asarray(vectors, dtype=np.float32), allow_pickle=False)

    _atomic_write(path, write, binary=True)


def _load_index_cache(
    cache_dir: Path,
    config: RetrievalExperimentConfig,
    cache_key: str,
) -> tuple[KnowledgeBase, dict[str, Any]] | None:
    vector_path = cache_dir / f"index-{cache_key}.npy"
    metadata_path = cache_dir / f"index-{cache_key}.json"
    if not vector_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        vector_bytes = vector_path.read_bytes()
        if metadata.get("cache_key") != cache_key:
            return None
        if metadata.get("embedding_model") != config.embedding_model:
            return None
        if hashlib.sha256(vector_bytes).hexdigest() != metadata.get("vector_sha256"):
            return None
        vectors = np.load(vector_path, allow_pickle=False)
        chunks = tuple(TextChunk(**chunk) for chunk in metadata["chunks"])
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            return None
        if vectors.shape[1] != metadata.get("model_dimension"):
            return None
        return KnowledgeBase(name="retrieval experiment index", chunks=chunks, vectors=vectors), metadata
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _build_or_load_index(
    pages: Sequence[DocumentPage],
    config: RetrievalExperimentConfig,
    cache_dir: Path,
    corpus_fingerprint: str,
    embedder: EmbeddingModel | None,
) -> tuple[KnowledgeBase, dict[str, Any], bool]:
    key = _index_cache_key(config, corpus_fingerprint)
    cached = _load_index_cache(cache_dir, config, key)
    if cached is not None:
        knowledge_base, metadata = cached
        return knowledge_base, metadata, True
    if embedder is None:
        raise RetrievalExperimentError("embedding model is required to build a missing index")

    selected_settings = RAGSettings(
        embedding_model=config.embedding_model,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        top_k=config.top_k,
        relevance_threshold=None,
    )
    started = perf_counter()
    knowledge_base = build_knowledge_base(pages, "retrieval experiment corpus", embedder, selected_settings)
    indexing_time_ms = (perf_counter() - started) * 1000.0
    dimension = int(knowledge_base.vectors.shape[1])
    declared = _declared_dimension(embedder)
    if declared is not None and declared != dimension:
        raise RetrievalExperimentError(
            f"embedding model declared dimension {declared} but produced {dimension} corpus dimensions"
        )
    if any(chunk.document_id not in {page.document_id for page in pages} for chunk in knowledge_base.chunks):
        raise RetrievalExperimentError("index chunks do not preserve their source document identity")

    vector_path = cache_dir / f"index-{key}.npy"
    metadata_path = cache_dir / f"index-{key}.json"
    _write_numpy(vector_path, knowledge_base.vectors)
    metadata = {
        "cache_key": key,
        "embedding_model": config.embedding_model,
        "model_dimension": dimension,
        "chunk_size_words": config.chunk_size,
        "chunk_overlap_words": config.chunk_overlap,
        "chunk_count": len(knowledge_base.chunks),
        "indexing_time_ms": indexing_time_ms,
        "embedding_model_parameter_bytes": _model_parameter_bytes(embedder),
        "vector_index_bytes": int(knowledge_base.vectors.nbytes),
        "vector_sha256": hashlib.sha256(vector_path.read_bytes()).hexdigest(),
        "chunks": [asdict(chunk) for chunk in knowledge_base.chunks],
    }
    _write_json(metadata_path, metadata)
    return knowledge_base, metadata, False


def _load_query_cache(path: Path, model: str, dataset_fingerprint: str) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("embedding_model") != model or value.get("dataset_fingerprint") != dataset_fingerprint:
            return {}
        rows = value.get("queries")
        if not isinstance(rows, dict):
            return {}
        return rows
    except (OSError, json.JSONDecodeError, AttributeError):
        return {}


def _write_query_cache(
    path: Path,
    model: str,
    dataset_fingerprint: str,
    queries: Mapping[str, Mapping[str, Any]],
) -> None:
    _write_json(
        path,
        {
            "embedding_model": model,
            "dataset_fingerprint": dataset_fingerprint,
            "queries": dict(queries),
        },
    )


def _source_dict(source: RetrievedSource) -> dict[str, Any]:
    return {
        "document_id": source.document_id,
        "document_name": source.source_name,
        "page_number": source.page_number,
        "chunk_id": source.chunk_id,
        "text": source.text,
        "score": float(source.score),
        "retrieval_method": source.retrieval_method,
        "diagnostics": dict(source.diagnostics),
    }


def _error_observation(
    question: EvaluationQuestion,
    config: RetrievalExperimentConfig,
    *,
    cache_key: str,
    dataset_fingerprint: str,
    corpus_fingerprint: str,
    failure_stage: str,
    error_type: str,
) -> dict[str, Any]:
    return {
        "cache_key": cache_key,
        "experiment_id": config.experiment_id,
        "split": question.split,
        "question_id": question.id,
        "question": question.question,
        "category": question.category,
        "expected_document_ids": list(question.expected_document_ids),
        "expected_pages": [asdict(page) for page in question.expected_pages],
        "status": "error",
        "failure_stage": failure_stage,
        "error_type": error_type,
        "retrieved_sources": [],
        "query_embedding_latency_ms": None,
        "vector_search_latency_ms": None,
        "retrieval_latency_ms": None,
        "model_dimension": None,
        "indexing_time_ms": None,
        "index_cache_hit": None,
        "embedding_model_parameter_bytes": None,
        "vector_index_bytes": None,
        "dataset_fingerprint": dataset_fingerprint,
        "corpus_fingerprint": corpus_fingerprint,
    }


def _read_observations(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RetrievalExperimentError("unable to read experiment observation cache") from exc
    for index, raw in enumerate(lines):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1:
                break  # tolerate a truncated final line after an interrupted write
            raise RetrievalExperimentError("experiment observation cache is corrupt") from exc
        if isinstance(row, dict) and isinstance(row.get("cache_key"), str):
            rows[row["cache_key"]] = row
    return rows


def _flush_observations(path: Path, observations: Mapping[str, Mapping[str, Any]]) -> None:
    _write_jsonl(path, list(observations.values()))


def _metric_value(metric: Mapping[str, Any]) -> float | None:
    value = metric.get("value")
    return float(value) if isinstance(value, (float, int)) else None


def _aggregate_results(
    questions: Sequence[EvaluationQuestion],
    configs: Sequence[RetrievalExperimentConfig],
    observations: Mapping[str, Mapping[str, Any]],
    *,
    dataset_fingerprint: str,
    corpus_fingerprint: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_experiment_split: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in observations.values():
        by_experiment_split[(row["experiment_id"], row["split"])].append(dict(row))

    question_map = {question.id: question for question in questions}
    for config in configs:
        for split in ("development", "validation"):
            split_questions = [question for question in questions if question.split == split]
            group = [
                item
                for item in by_experiment_split.get((config.experiment_id, split), [])
                if item.get("question_id") in question_map
            ]
            by_id = {item["question_id"]: item for item in group}
            successful = [
                question for question in split_questions
                if question.id in by_id and by_id[question.id].get("status") == "ok"
            ]
            predictions = [
                {
                    "question_id": question.id,
                    "retrieved_sources": by_id[question.id].get("retrieved_sources", []),
                }
                for question in successful
            ]
            metric_values: dict[str, Any] = {}
            for cutoff in METRIC_CUTOFFS:
                metric = compute_metrics(successful, predictions, retrieval_k=cutoff)["overall"]["retrieval"]
                metric_values[f"recall_at_{cutoff}"] = metric[f"recall_at_{cutoff}"]
                metric_values[f"mrr_at_{cutoff}"] = metric[f"mrr_at_{cutoff}"]

            completed_observations = [by_id[q.id] for q in successful]
            errors = [item for item in group if item.get("status") != "ok"]
            failed_count = len(split_questions) - len(successful)
            if failed_count == 0 and len(successful) == len(split_questions):
                status = "complete"
            elif successful:
                status = "partial_failure"
            else:
                status = "failed"
            successful_count = len(successful)
            base = next((item for item in completed_observations if item.get("model_dimension") is not None), {})
            latency_rows = [
                float(item["retrieval_latency_ms"])
                for item in completed_observations
                if isinstance(item.get("retrieval_latency_ms"), (int, float))
            ]
            query_latencies = [
                float(item["query_embedding_latency_ms"])
                for item in completed_observations
                if isinstance(item.get("query_embedding_latency_ms"), (int, float))
            ]
            search_latencies = [
                float(item["vector_search_latency_ms"])
                for item in completed_observations
                if isinstance(item.get("vector_search_latency_ms"), (int, float))
            ]
            result = {
                "experiment_id": config.experiment_id,
                "split": split,
                "status": status,
                "is_baseline": config.is_baseline,
                "embedding_model": config.embedding_model,
                "model_dimension": base.get("model_dimension"),
                "chunk_size_words": config.chunk_size,
                "chunk_overlap_words": config.chunk_overlap,
                "top_k": config.top_k,
                "dataset_fingerprint": dataset_fingerprint,
                "corpus_fingerprint": corpus_fingerprint,
                "question_count": len(split_questions),
                "answerable_question_count": sum(question.answerable for question in split_questions),
                "failed_question_count": failed_count,
                "failed_question_rate": failed_count / len(split_questions) if split_questions else None,
                "indexing_time_ms": base.get("indexing_time_ms"),
                "index_cache_hit": base.get("index_cache_hit"),
                "query_embedding_latency_mean_ms": mean(query_latencies) if query_latencies else None,
                "vector_search_latency_mean_ms": mean(search_latencies) if search_latencies else None,
                "retrieval_latency_mean_ms": mean(latency_rows) if latency_rows else None,
                "retrieval_latency_p50_ms": _percentile(latency_rows, 0.50),
                "retrieval_latency_p95_ms": _percentile(latency_rows, 0.95),
                "embedding_model_parameter_bytes": base.get("embedding_model_parameter_bytes"),
                "vector_index_bytes": base.get("vector_index_bytes"),
            }
            for cutoff in METRIC_CUTOFFS:
                for metric_name in (f"recall_at_{cutoff}", f"mrr_at_{cutoff}"):
                    metric = metric_values[metric_name]
                    result[metric_name] = _metric_value(metric)
                    result[f"{metric_name}_n"] = metric.get("n", 0)
            rows.append(result)
    return rows


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _select_best(
    results: Sequence[Mapping[str, Any]],
    rule: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = [
        dict(row)
        for row in results
        if row.get("split") == "validation"
        and row.get("status") == "complete"
        and int(row.get("answerable_question_count", 0)) > 0
        and row.get(rule.get("primary", {}).get("metric")) is not None
        and row.get(rule.get("secondary", {}).get("metric")) is not None
    ]
    if not candidates:
        return {
            "status": "not_selected",
            "split": "validation",
            "selection_rule": dict(rule),
            "reason": "No complete validation configuration had answerable reviewed questions and calculable primary metrics.",
        }

    criteria = [rule["primary"], rule["secondary"], *rule.get("tie_breakers", [])]

    def sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        values: list[Any] = []
        for criterion in criteria:
            metric = criterion["metric"]
            direction = criterion["direction"]
            value = row.get(metric)
            if direction == "max":
                values.append(-float(value) if value is not None else math.inf)
            elif direction == "min":
                values.append(float(value) if value is not None else math.inf)
            elif direction == "asc":
                values.append(str(value))
            else:
                raise RetrievalExperimentError(f"unsupported selection direction {direction!r}")
        return tuple(values)

    winner = sorted(candidates, key=sort_key)[0]
    config_keys = {"experiment_id", "embedding_model", "model_dimension", "chunk_size_words", "chunk_overlap_words", "top_k", "is_baseline"}
    return {
        "status": "selected",
        "split": "validation",
        "selection_rule": dict(rule),
        "configuration": {key: winner.get(key) for key in config_keys},
        "validation_metrics": {
            "recall_at_3": winner.get("recall_at_3"),
            "recall_at_3_n": winner.get("recall_at_3_n"),
            "mrr_at_3": winner.get("mrr_at_3"),
            "mrr_at_3_n": winner.get("mrr_at_3_n"),
            "retrieval_latency_mean_ms": winner.get("retrieval_latency_mean_ms"),
            "vector_index_bytes": winner.get("vector_index_bytes"),
        },
        "rationale": "Selected by the recorded validation-only rule; this is not a held-out test result or a universal model ranking.",
    }


def _failure_cases(
    questions: Sequence[EvaluationQuestion],
    observations: Sequence[Mapping[str, Any]],
    config_by_id: Mapping[str, RetrievalExperimentConfig],
) -> list[dict[str, Any]]:
    question_by_id = {question.id: question for question in questions}
    failures: list[dict[str, Any]] = []
    for row in observations:
        question = question_by_id.get(str(row.get("question_id")))
        config = config_by_id.get(str(row.get("experiment_id")))
        if question is None or config is None:
            continue
        reason: str | None = None
        if row.get("status") != "ok":
            reason = "retrieval_request_failed"
        elif question.answerable:
            metric = compute_metrics(
                [question],
                [{"question_id": question.id, "retrieved_sources": row.get("retrieved_sources", [])}],
                retrieval_k=config.top_k,
            )["overall"]["retrieval"][f"recall_at_{config.top_k}"]
            if metric.get("value") is not None and metric["value"] < 1.0:
                reason = "expected_evidence_not_fully_retrieved"
        if reason is None:
            continue
        failures.append(
            {
                "experiment_id": config.experiment_id,
                "split": question.split,
                "question_id": question.id,
                "question": question.question,
                "category": question.category,
                "reason": reason,
                "failure_stage": row.get("failure_stage"),
                "error_type": row.get("error_type"),
                "expected_document_ids": list(question.expected_document_ids),
                "expected_pages": [asdict(page) for page in question.expected_pages],
                "retrieved_sources": row.get("retrieved_sources", []),
            }
        )
    return failures


def _write_results_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    def write(target: Any) -> None:
        writer = csv.DictWriter(target, fieldnames=_RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    _atomic_write(path, write)


def _write_interpretation(path: Path, results: Sequence[Mapping[str, Any]], best: Mapping[str, Any]) -> None:
    if best.get("status") != "selected":
        content = (
            "# Retrieval experiment interpretation\n\n"
            "No configuration was selected. There were no complete validation measurements with answerable reviewed questions. "
            "No performance conclusion is available.\n"
        )
    else:
        selected = best["configuration"]
        metrics = best["validation_metrics"]
        baseline = next(
            (row for row in results if row.get("split") == "validation" and row.get("is_baseline") is True),
            None,
        )
        comparison = "The baseline was not fully measured, so a baseline delta is unavailable."
        if baseline is not None and baseline.get("status") == "complete":
            recall_delta = float(metrics["recall_at_3"]) - float(baseline.get("recall_at_3") or 0.0)
            mrr_delta = float(metrics["mrr_at_3"]) - float(baseline.get("mrr_at_3") or 0.0)
            comparison = (
                f"Compared with the baseline, validation Recall@3 changed by {recall_delta:+.4f} "
                f"and MRR@3 by {mrr_delta:+.4f}."
            )
        content = (
            "# Retrieval experiment interpretation\n\n"
            f"The validation-only selection rule chose `{selected['experiment_id']}` "
            f"({selected['embedding_model']}, chunk size {selected['chunk_size_words']}, "
            f"overlap {selected['chunk_overlap_words']}, top-k {selected['top_k']}). "
            f"Its measured Recall@3 was {metrics['recall_at_3']:.4f} "
            f"and MRR@3 was {metrics['mrr_at_3']:.4f} "
            f"over {metrics['recall_at_3_n']} answerable validation questions. {comparison}\n\n"
            "This is the winner under the documented validation rule only; it does not establish performance on held-out test data."
        )

    _atomic_write(path, lambda target: target.write(content))


def run_retrieval_experiments(
    questions: Sequence[EvaluationQuestion],
    pages: Sequence[DocumentPage],
    *,
    output_dir: str | Path,
    configs: Sequence[RetrievalExperimentConfig] | None = None,
    embedder_factory: Callable[[str], EmbeddingModel] = SentenceTransformerEmbedder,
    selection_rule: Mapping[str, Any] = SELECTION_RULE,
    source_documents: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Run retrieval-only variants, cache intermediate indexes, and select on validation."""

    selected_questions, dataset_fingerprint = _check_inputs(questions, pages)
    selected_configs = tuple(build_experiment_grid() if configs is None else configs)
    if not selected_configs:
        raise RetrievalExperimentError("at least one retrieval configuration is required")
    if len({config.experiment_id for config in selected_configs}) != len(selected_configs):
        raise RetrievalExperimentError("configuration IDs must be unique")
    corpus_fingerprint = _corpus_fingerprint(pages, source_documents)
    output_path = Path(output_dir)
    cache_dir = output_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    observations_path = output_path / "observations.jsonl"
    observations = _read_observations(observations_path)

    current_keys = {
        _observation_cache_key(question, config, dataset_fingerprint, corpus_fingerprint)
        for config in selected_configs
        for question in selected_questions
    }
    observations = {
        key: row
        for key, row in observations.items()
        if key in current_keys and row.get("status") == "ok"
    }
    _flush_observations(observations_path, observations)

    configs_by_model: dict[str, list[RetrievalExperimentConfig]] = defaultdict(list)
    for config in selected_configs:
        configs_by_model[config.embedding_model].append(config)

    for model_name, model_configs in configs_by_model.items():
        query_cache_path = cache_dir / f"queries-{_fingerprint({'model': model_name, 'dataset': dataset_fingerprint})}.json"
        query_cache = _load_query_cache(query_cache_path, model_name, dataset_fingerprint)
        missing_questions: dict[str, EvaluationQuestion] = {}
        needed_configs: list[RetrievalExperimentConfig] = []
        for config in model_configs:
            if any(
                _observation_cache_key(question, config, dataset_fingerprint, corpus_fingerprint) not in observations
                for question in selected_questions
            ):
                needed_configs.append(config)
                for question in selected_questions:
                    key = _observation_cache_key(question, config, dataset_fingerprint, corpus_fingerprint)
                    if key not in observations:
                        missing_questions[question.id] = question
        if not needed_configs:
            continue

        chunk_groups: dict[tuple[int, int], list[RetrievalExperimentConfig]] = defaultdict(list)
        for config in needed_configs:
            chunk_groups[(config.chunk_size, config.chunk_overlap)].append(config)

        index_cache_lookups = {
            (size, overlap): _load_index_cache(
                cache_dir,
                configs[0],
                _index_cache_key(configs[0], corpus_fingerprint),
            )
            for (size, overlap), configs in chunk_groups.items()
        }
        requires_index_build = any(result is None for result in index_cache_lookups.values())
        requires_query_vectors = any(
            _query_cache_key(question, model_name, dataset_fingerprint) not in query_cache
            for question in missing_questions.values()
        )
        embedder: EmbeddingModel | None = None
        model_load_error: Exception | None = None
        if requires_index_build or requires_query_vectors:
            try:
                embedder = embedder_factory(model_name)
            except Exception as exc:
                model_load_error = exc

        if model_load_error is not None:
            for config in needed_configs:
                for question in selected_questions:
                    cache_key = _observation_cache_key(question, config, dataset_fingerprint, corpus_fingerprint)
                    if cache_key in observations:
                        continue
                    observations[cache_key] = _error_observation(
                        question,
                        config,
                        cache_key=cache_key,
                        dataset_fingerprint=dataset_fingerprint,
                        corpus_fingerprint=corpus_fingerprint,
                        failure_stage="model_load",
                        error_type=type(model_load_error).__name__,
                    )
            _flush_observations(observations_path, observations)
            continue

        if embedder is not None and requires_query_vectors:
            to_encode = [
                question
                for question in missing_questions.values()
                if _query_cache_key(question, model_name, dataset_fingerprint) not in query_cache
            ]
            if to_encode:
                try:
                    embedder.encode([to_encode[0].query_text])  # model warm-up; excluded from timings
                except Exception:
                    pass
            for question in to_encode:
                cache_key = _query_cache_key(question, model_name, dataset_fingerprint)
                started = perf_counter()
                try:
                    encoded = np.asarray(embedder.encode([question.query_text]), dtype=np.float32)
                    elapsed_ms = (perf_counter() - started) * 1000.0
                    vector = normalize_query_vector(encoded, expected_dimension=encoded.shape[-1])
                    query_cache[cache_key] = {
                        "vector": vector.tolist(),
                        "dimension": int(vector.shape[0]),
                        "embedding_latency_ms": elapsed_ms,
                    }
                except Exception as exc:
                    query_cache[cache_key] = {
                        "error_type": type(exc).__name__,
                    }
                _write_query_cache(query_cache_path, model_name, dataset_fingerprint, query_cache)

        for group_key, group_configs in chunk_groups.items():
            example_config = group_configs[0]
            cached_index = index_cache_lookups[group_key]
            try:
                if cached_index is None:
                    knowledge_base, index_metadata, index_hit = _build_or_load_index(
                        pages,
                        example_config,
                        cache_dir,
                        corpus_fingerprint,
                        embedder,
                    )
                else:
                    knowledge_base, index_metadata = cached_index
                    index_hit = True
            except Exception as exc:
                for config in group_configs:
                    for question in selected_questions:
                        cache_key = _observation_cache_key(question, config, dataset_fingerprint, corpus_fingerprint)
                        if cache_key in observations:
                            continue
                        observations[cache_key] = _error_observation(
                            question,
                            config,
                            cache_key=cache_key,
                            dataset_fingerprint=dataset_fingerprint,
                            corpus_fingerprint=corpus_fingerprint,
                            failure_stage="index_build",
                            error_type=type(exc).__name__,
                        )
                _flush_observations(observations_path, observations)
                continue

            settings = RAGSettings(
                embedding_model=model_name,
                chunk_size=example_config.chunk_size,
                chunk_overlap=example_config.chunk_overlap,
                top_k=example_config.top_k,
                relevance_threshold=None,
            )
            for config in group_configs:
                for question in selected_questions:
                    cache_key = _observation_cache_key(question, config, dataset_fingerprint, corpus_fingerprint)
                    if cache_key in observations:
                        continue
                    query = query_cache.get(_query_cache_key(question, model_name, dataset_fingerprint), {})
                    if "error_type" in query or "vector" not in query:
                        observations[cache_key] = _error_observation(
                            question,
                            config,
                            cache_key=cache_key,
                            dataset_fingerprint=dataset_fingerprint,
                            corpus_fingerprint=corpus_fingerprint,
                            failure_stage="query_embedding",
                            error_type=str(query.get("error_type", "MissingQueryVector")),
                        )
                    else:
                        try:
                            vector = np.asarray(query["vector"], dtype=np.float32)
                            if int(query.get("dimension", -1)) != int(knowledge_base.vectors.shape[1]):
                                raise ValueError("embedding dimension does not match model-specific index")
                            started = perf_counter()
                            sources = retrieve(
                                question.query_text,
                                knowledge_base,
                                None,
                                settings,
                                top_k=config.top_k,
                                query_vector=vector,
                            )
                            search_ms = (perf_counter() - started) * 1000.0
                            embed_ms = float(query["embedding_latency_ms"])
                            observation = {
                                "cache_key": cache_key,
                                "experiment_id": config.experiment_id,
                                "split": question.split,
                                "question_id": question.id,
                                "question": question.question,
                                "category": question.category,
                                "expected_document_ids": list(question.expected_document_ids),
                                "expected_pages": [asdict(page) for page in question.expected_pages],
                                "status": "ok",
                                "retrieved_sources": [_source_dict(source) for source in sources],
                                "query_embedding_latency_ms": embed_ms,
                                "vector_search_latency_ms": search_ms,
                                "retrieval_latency_ms": embed_ms + search_ms,
                                "model_dimension": int(knowledge_base.vectors.shape[1]),
                                "indexing_time_ms": float(index_metadata["indexing_time_ms"]),
                                "index_cache_hit": index_hit,
                                "embedding_model_parameter_bytes": index_metadata.get("embedding_model_parameter_bytes"),
                                "vector_index_bytes": int(knowledge_base.vectors.nbytes),
                                "dataset_fingerprint": dataset_fingerprint,
                                "corpus_fingerprint": corpus_fingerprint,
                            }
                            observations[cache_key] = observation
                        except Exception as exc:
                            observations[cache_key] = _error_observation(
                                question,
                                config,
                                cache_key=cache_key,
                                dataset_fingerprint=dataset_fingerprint,
                                corpus_fingerprint=corpus_fingerprint,
                                failure_stage="retrieval",
                                error_type=type(exc).__name__,
                            )
                    _flush_observations(observations_path, observations)

    current_observations = [
        row for key, row in observations.items()
        if key in current_keys
    ]
    results = _aggregate_results(
        selected_questions,
        selected_configs,
        {row["cache_key"]: row for row in current_observations},
        dataset_fingerprint=dataset_fingerprint,
        corpus_fingerprint=corpus_fingerprint,
    )
    best = _select_best(results, selection_rule)
    config_record = {
        "experiment_revision": EXPERIMENT_REVISION,
        "embedding_models_are_isolated": True,
        "generation_evaluation_included": False,
        "splits_run": ["development", "validation"],
        "held_out_test_used_for_selection": False,
        "metric_cutoffs": list(METRIC_CUTOFFS),
        "selection_rule": dict(selection_rule),
        "dataset_fingerprint": dataset_fingerprint,
        "corpus_fingerprint": corpus_fingerprint,
        "source_documents": list(source_documents),
        "question_counts": {
            split: sum(question.split == split for question in selected_questions)
            for split in ("development", "validation")
        },
        "configurations": [config.to_dict() for config in selected_configs],
    }
    _write_json(output_path / "experiment_config.json", config_record)
    _write_results_csv(output_path / "results.csv", results)
    config_by_id = {config.experiment_id: config for config in selected_configs}
    failures = _failure_cases(selected_questions, current_observations, config_by_id)
    _write_jsonl(output_path / "failure_cases.jsonl", failures)
    _write_json(output_path / "best_configuration.json", best)
    _write_interpretation(output_path / "interpretation.md", results, best)
    return {
        "experiment_config_path": output_path / "experiment_config.json",
        "results_csv_path": output_path / "results.csv",
        "observations_path": observations_path,
        "failure_cases_path": output_path / "failure_cases.jsonl",
        "best_configuration_path": output_path / "best_configuration.json",
        "interpretation_path": output_path / "interpretation.md",
        "results": results,
        "best_configuration": best,
        "observation_count": len(current_observations),
    }


def _load_manifest_pages(path: str | Path) -> tuple[tuple[DocumentPage, ...], tuple[dict[str, Any], ...]]:
    return load_pdf_manifest(path)


def main(argv: Sequence[str] | None = None) -> int:
    repository_root = Path(__file__).parents[3]
    parser = argparse.ArgumentParser(description="Run CampusReg retrieval-only experiments.")
    parser.add_argument("--dataset", default=repository_root / "evaluation" / "questions.jsonl")
    parser.add_argument("--corpus-manifest", required=True)
    parser.add_argument("--plan", default=repository_root / "evaluation" / "retrieval-experiment.config.json")
    parser.add_argument(
        "--output-dir",
        default=repository_root / "evaluation" / "results" / "retrieval-experiments",
    )
    args = parser.parse_args(argv)

    validate_challenge_dataset(args.dataset)
    questions = load_dataset(args.dataset)
    pages, source_documents = load_pdf_manifest(args.corpus_manifest)
    configs, rule = load_experiment_plan(args.plan)
    run = run_retrieval_experiments(
        questions,
        pages,
        output_dir=args.output_dir,
        configs=configs,
        selection_rule=rule,
        source_documents=source_documents,
    )
    print(f"Saved retrieval experiment outputs to {Path(args.output_dir).resolve()}")
    print(f"Completed {run['observation_count']} cached or measured question/configuration observations.")
    print(f"Best configuration status: {run['best_configuration']['status']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI smoke tests only
    raise SystemExit(main())
