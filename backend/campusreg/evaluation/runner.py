"""Cacheable, split-aware evaluation runner with resumable raw prediction output."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

from campusreg.evaluation.metrics import compute_metrics
from campusreg.evaluation.models import EVALUATION_SPLITS, EvaluationPrediction, EvaluationQuestion
from campusreg.rag.models import AnswerResult, AnswerStatus, RetrievedSource

ConflictDetector = Callable[[EvaluationQuestion, Sequence[RetrievedSource]], bool | None]
_CITATION_PATTERN = re.compile(r"\[[^\]\n]+\]")
_SECRET_CONFIG_KEY = re.compile(r"api[_-]?key|secret|token|password|credential|authorization", re.I)
_SAFE_RUN_CONFIG_FIELDS = {
    "gemma_model",
    "embedding_model",
    "chunk_size",
    "chunk_overlap",
    "top_k",
    "temperature",
    "relevance_threshold",
    "retrieval_k",
    "index_revision",
    "corpus_revision",
    "prompt_revision",
    "code_revision",
    "conflict_detector_revision",
    "dataset_version",
    "conversation_revision",
    "retrieval_method",
    "candidate_count",
    "rrf_constant",
}


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    split: str
    config_fingerprint: str
    predictions: tuple[EvaluationPrediction, ...]
    metrics: dict[str, Any]
    predictions_path: Path
    metrics_json_path: Path
    metrics_csv_path: Path


def fingerprint_config(config: Mapping[str, Any]) -> str:
    """Return a stable, secret-free fingerprint used as a cache key."""

    def without_secrets(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): without_secrets(item)
                for key, item in value.items()
                if not _SECRET_CONFIG_KEY.search(str(key))
            }
        if isinstance(value, (list, tuple)):
            return [without_secrets(item) for item in value]
        return value

    try:
        encoded = json.dumps(without_secrets(config), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError):
        raise ValueError("run_config must contain only JSON-serializable values") from None
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _safe_config(value: Any) -> Any:
    """Strip secret-shaped configuration keys before writing run metadata."""

    if isinstance(value, Mapping):
        return {
            str(key): _safe_config(item)
            for key, item in value.items()
            if str(key) in _SAFE_RUN_CONFIG_FIELDS and not _SECRET_CONFIG_KEY.search(str(key))
        }
    if isinstance(value, (list, tuple)):
        return [_safe_config(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _cache_key(question: EvaluationQuestion, config_fingerprint: str) -> str:
    body = json.dumps(
        {
            "question": asdict(question),
            "config_fingerprint": config_fingerprint,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _source_row(source: RetrievedSource) -> dict[str, Any]:
    return {
        "document_id": source.document_id,
        "document_name": source.source_name,
        "page_number": source.page_number,
        "chunk_id": source.chunk_id,
        "citation": source.citation,
        "text": source.text,
        "score": float(source.score),
        "retrieval_method": source.retrieval_method,
        "diagnostics": dict(source.diagnostics),
    }


def _human_label_template() -> dict[str, Any]:
    return {
        "answer_correct": None,
        "citation_support": None,
        "evidence_support": None,
        "appropriate_refusal": None,
        "reviewer": None,
        "reviewed_at": None,
        "notes": None,
    }


def _prediction_from_answer(
    question: EvaluationQuestion,
    result: AnswerResult,
    *,
    cache_key: str,
    config_fingerprint: str,
    latency_ms: float,
    conflict_detector: ConflictDetector | None,
) -> EvaluationPrediction:
    sources = [_source_row(source) for source in result.sources]
    answer_text = result.answer
    labels = _CITATION_PATTERN.findall(answer_text or "") if result.status is AnswerStatus.ANSWERED else []
    allowed = {source["citation"] for source in sources}
    citation_exists = bool(labels) if result.status is AnswerStatus.ANSWERED else None
    citation_valid = (bool(labels) and all(label in allowed for label in labels)) if citation_exists is not None else None
    conflict_detected: bool | None = None
    conflict_failed = False
    if conflict_detector is not None:
        try:
            detected = conflict_detector(question, result.sources)
            if detected is not None and not isinstance(detected, bool):
                raise TypeError("conflict detector must return bool or None")
            conflict_detected = detected
        except Exception:
            conflict_failed = True
    error_code = result.error.code if result.error else None
    return EvaluationPrediction(
        question_id=question.id,
        question=question.question,
        category=question.category,
        split=question.split,
        cache_key=cache_key,
        config_fingerprint=config_fingerprint,
        status=result.status.value,
        answer=answer_text,
        retrieved_sources=sources,
        latency_ms=latency_ms,
        request_failed=result.status is AnswerStatus.GENERATION_ERROR,
        error_code=error_code,
        citation_exists=citation_exists,
        citation_valid=citation_valid,
        conflict_detected=conflict_detected,
        conflict_detection_failed=conflict_failed,
        human_labels=_human_label_template(),
        evidence_gate=dict(result.gate_decision) if result.gate_decision is not None else None,
        retrieval_query=result.retrieval_query,
    )


def _prediction_from_exception(
    question: EvaluationQuestion,
    *,
    cache_key: str,
    config_fingerprint: str,
    latency_ms: float,
) -> EvaluationPrediction:
    return EvaluationPrediction(
        question_id=question.id,
        question=question.question,
        category=question.category,
        split=question.split,
        cache_key=cache_key,
        config_fingerprint=config_fingerprint,
        status="request_error",
        answer=None,
        retrieved_sources=[],
        latency_ms=latency_ms,
        request_failed=True,
        error_code="request_exception",
        citation_exists=None,
        citation_valid=None,
        conflict_detected=None,
        conflict_detection_failed=False,
        human_labels=_human_label_template(),
    )


def _write_jsonl_atomic(path: Path, predictions: Sequence[EvaluationPrediction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as target:
            for prediction in predictions:
                target.write(json.dumps(prediction.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
            target.flush()
            os.fsync(target.fileno())
        for attempt in range(4):
            try:
                os.replace(temporary_name, path)
                break
            except PermissionError:
                if attempt == 3:
                    raise
                sleep(0.025 * (2 ** attempt))
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _read_cache(path: Path) -> dict[str, EvaluationPrediction]:
    cached: dict[str, EvaluationPrediction] = {}
    if not path.exists():
        return cached
    with path.open("r", encoding="utf-8") as source:
        for line_number, raw in enumerate(source, start=1):
            if not raw.strip():
                continue
            try:
                item = EvaluationPrediction.from_dict(json.loads(raw))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid cached prediction at {path}:{line_number}") from exc
            cached[item.question_id] = item
    return cached


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as target:
            json.dump(value, target, ensure_ascii=False, indent=2, sort_keys=True)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _metric_rows(value: Any, prefix: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if "value" in value and "n" in value:
            rows.append({"metric": prefix, "value": value["value"], "n": value["n"]})
        else:
            for key, item in value.items():
                rows.extend(_metric_rows(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        rows.append({"metric": prefix, "value": value, "n": ""})
    return rows


def _write_metrics_csv(path: Path, metrics: Mapping[str, Any]) -> None:
    rows = _metric_rows(dict(metrics))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=["metric", "value", "n"])
            writer.writeheader()
            writer.writerows(rows)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


class EvaluationRunner:
    """Run one fixed split, reuse matching predictions, and save structured outputs."""

    def run(
        self,
        questions: Sequence[EvaluationQuestion],
        *,
        answerer: Callable[[str], AnswerResult],
        output_dir: str | Path,
        split: str,
        retrieval_k: int,
        run_config: Mapping[str, Any],
        conflict_detector: ConflictDetector | None = None,
        conversation_service: Any | None = None,
        force: bool = False,
    ) -> EvaluationRun:
        if split not in EVALUATION_SPLITS:
            raise ValueError(f"split must be one of: {', '.join(EVALUATION_SPLITS)}")
        if retrieval_k <= 0:
            raise ValueError("retrieval_k must be positive")
        if not run_config:
            raise ValueError("run_config must identify model, prompt, and index revisions")
        required_groups = {
            "Gemma model": ("gemma_model",),
            "embedding model": ("embedding_model",),
            "index or corpus revision": ("index_revision", "corpus_revision"),
            "prompt revision": ("prompt_revision",),
        }
        missing = [
            label
            for label, alternatives in required_groups.items()
            if not any(run_config.get(name) for name in alternatives)
        ]
        if missing:
            raise ValueError(f"run_config must include: {', '.join(missing)}")
        selected = [question for question in questions if question.split == split]
        if not selected:
            raise ValueError(f"no ready questions found for split {split!r}")
        if len({question.id for question in selected}) != len(selected):
            raise ValueError("question IDs must be unique within a split")

        fingerprint = fingerprint_config(run_config)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        predictions_path = output_path / f"predictions_{split}.jsonl"
        metrics_json_path = output_path / f"metrics_{split}.json"
        metrics_csv_path = output_path / f"metrics_{split}.csv"
        cached = {} if force else _read_cache(predictions_path)
        prediction_map: dict[str, EvaluationPrediction] = {}
        for question in selected:
            key = _cache_key(question, fingerprint)
            prior = cached.get(question.id)
            if prior is not None and prior.cache_key == key and not force:
                prediction_map[question.id] = prior

        for question in selected:
            if question.id in prediction_map:
                continue
            key = _cache_key(question, fingerprint)
            start = perf_counter()
            try:
                if question.category == "follow_up" and conversation_service is not None:
                    conversation_id = conversation_service.create_conversation()
                    try:
                        for turn in question.conversation_context:
                            conversation_service.store.append_conversation_message(
                                conversation_id,
                                role=turn.role,
                                content=turn.content,
                                original_question=(turn.content if turn.role == "user" else None),
                            )
                        result = conversation_service.ask(
                            conversation_id, question.question
                        ).result
                    finally:
                        conversation_service.delete_conversation(conversation_id)
                else:
                    result = answerer(question.query_text)
                if not isinstance(result, AnswerResult):
                    raise TypeError("answerer must return AnswerResult")
                elapsed_ms = (perf_counter() - start) * 1000.0
                prediction = _prediction_from_answer(
                    question,
                    result,
                    cache_key=key,
                    config_fingerprint=fingerprint,
                    latency_ms=elapsed_ms,
                    conflict_detector=conflict_detector,
                )
            except Exception:
                elapsed_ms = (perf_counter() - start) * 1000.0
                prediction = _prediction_from_exception(
                    question,
                    cache_key=key,
                    config_fingerprint=fingerprint,
                    latency_ms=elapsed_ms,
                )
            prediction_map[question.id] = prediction
            _write_jsonl_atomic(
                predictions_path,
                [prediction_map[item.id] for item in selected if item.id in prediction_map],
            )

        predictions = tuple(prediction_map[question.id] for question in selected)
        _write_jsonl_atomic(predictions_path, predictions)
        metrics = compute_metrics(selected, predictions, retrieval_k=retrieval_k)
        now = datetime.now(timezone.utc).isoformat()
        dataset_data = json.dumps(
            [asdict(question) for question in selected],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        metadata = {
            "split": split,
            "created_at": now,
            "question_count": len(selected),
            "dataset_fingerprint": hashlib.sha256(dataset_data.encode("utf-8")).hexdigest(),
            "config_fingerprint": fingerprint,
            "run_config": _safe_config(run_config),
            "retrieval_k": retrieval_k,
        }
        _atomic_json(metrics_json_path, {"metadata": metadata, "metrics": metrics})
        _write_metrics_csv(metrics_csv_path, metrics)
        return EvaluationRun(
            split=split,
            config_fingerprint=fingerprint,
            predictions=predictions,
            metrics=metrics,
            predictions_path=predictions_path,
            metrics_json_path=metrics_json_path,
            metrics_csv_path=metrics_csv_path,
        )
