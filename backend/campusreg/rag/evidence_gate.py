"""Config-bound evidence thresholds; retrieval scores are not probabilities."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from campusreg.config import RAGSettings
from campusreg.rag.models import RetrievedSource


def retrieval_configuration(settings: RAGSettings) -> dict[str, Any]:
    config: dict[str, Any] = {
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": settings.top_k,
        "retrieval_method": settings.retrieval_method,
        "corpus_revision": settings.corpus_revision,
    }
    if settings.retrieval_method == "hybrid":
        config.update({"candidate_count": settings.candidate_count, "rrf_constant": settings.rrf_constant})
    return config


def configuration_id(config: Mapping[str, Any]) -> str:
    canonical = json.dumps(dict(config), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "retrieval-v1:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class EvidenceGatePolicy:
    configuration_id: str
    embedding_model: str
    retrieval_method: str
    score_kind: str
    threshold: float
    validation_dataset_fingerprint: str
    max_false_accept_rate: float
    max_unsupported_answer_rate: float
    validation_metrics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold):
            raise ValueError("evidence threshold must be finite")
        if not 0.0 <= self.max_false_accept_rate <= 1.0:
            raise ValueError("max_false_accept_rate must be between zero and one")
        if not 0.0 <= self.max_unsupported_answer_rate <= 1.0:
            raise ValueError("max_unsupported_answer_rate must be between zero and one")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceGatePolicy":
        required = {
            "configuration_id", "embedding_model", "retrieval_method", "score_kind", "threshold",
            "validation_dataset_fingerprint", "max_false_accept_rate", "max_unsupported_answer_rate",
            "validation_metrics",
        }
        if not required.issubset(value):
            raise ValueError("evidence gate policy is missing required calibration fields")
        return cls(
            configuration_id=str(value["configuration_id"]),
            embedding_model=str(value["embedding_model"]),
            retrieval_method=str(value["retrieval_method"]),
            score_kind=str(value["score_kind"]),
            threshold=float(value["threshold"]),
            validation_dataset_fingerprint=str(value["validation_dataset_fingerprint"]),
            max_false_accept_rate=float(value["max_false_accept_rate"]),
            max_unsupported_answer_rate=float(value["max_unsupported_answer_rate"]),
            validation_metrics=dict(value["validation_metrics"]),
        )


@dataclass(frozen=True, slots=True)
class EvidenceGateDecision:
    accepted: bool
    configuration_id: str
    retrieval_score: float | None
    score_kind: str
    threshold: float
    reason: str
    missing_information: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvidenceGate:
    """Apply a validation-calibrated threshold for exactly one retrieval configuration."""

    def __init__(self, policy: EvidenceGatePolicy) -> None:
        self.policy = policy

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        embedding_model: str | None = None,
        configuration_id: str | None = None,
    ) -> "EvidenceGate":
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
            if isinstance(value, dict) and isinstance(value.get("policies"), list):
                policies = value["policies"]
                if configuration_id is not None:
                    policies = [policy for policy in policies if policy.get("configuration_id") == configuration_id]
                if len(policies) != 1:
                    raise ValueError("select exactly one calibrated configuration policy")
                policy_value = policies[0]
            else:
                policy_value = value["policy"] if isinstance(value, dict) and "policy" in value else value
            policy = EvidenceGatePolicy.from_dict(policy_value)
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError("unable to load a calibrated evidence gate policy") from exc
        if embedding_model is not None and embedding_model != policy.embedding_model:
            raise ValueError("evidence gate policy belongs to a different embedding model")
        return cls(policy)

    def evaluate(
        self,
        sources: Sequence[RetrievedSource],
        settings: RAGSettings,
    ) -> EvidenceGateDecision:
        current_id = configuration_id(retrieval_configuration(settings))
        if current_id != self.policy.configuration_id:
            raise ValueError("evidence gate calibration does not match the active retrieval configuration")
        scores = [float(source.score) for source in sources]
        if any(not math.isfinite(score) for score in scores):
            raise ValueError("retrieval evidence contains a non-finite score")
        score = max(scores) if scores else None
        accepted = score is not None and score >= self.policy.threshold
        return EvidenceGateDecision(
            accepted=accepted,
            configuration_id=current_id,
            retrieval_score=score,
            score_kind=self.policy.score_kind,
            threshold=self.policy.threshold,
            reason="threshold_met" if accepted else "insufficient_relevance_evidence",
            missing_information=(
                None if accepted else "The retrieved passages do not establish the requested policy detail."
            ),
        )
