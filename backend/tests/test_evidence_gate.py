"""Synthetic-only evidence gate runtime tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from campusreg.config import RAGSettings
from campusreg.rag.evidence_gate import (
    EvidenceGate,
    EvidenceGatePolicy,
    configuration_id,
    retrieval_configuration,
)
from campusreg.rag.models import AnswerStatus, DocumentPage, KnowledgeBase, RetrievedSource, TextChunk
from campusreg.rag.pipeline import answer


class FakeEmbedder:
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


def _settings(**changes) -> RAGSettings:
    defaults = {
        "embedding_model": "SYNTHETIC TEST FIXTURE embedding",
        "chunk_size": 8,
        "chunk_overlap": 2,
        "top_k": 1,
        "corpus_revision": "SYNTHETIC TEST FIXTURE corpus v1",
    }
    defaults.update(changes)
    return RAGSettings(**defaults)


def _policy(settings: RAGSettings, threshold: float = 0.9) -> EvidenceGatePolicy:
    return EvidenceGatePolicy(
        configuration_id=configuration_id(retrieval_configuration(settings)),
        embedding_model=settings.embedding_model,
        retrieval_method=settings.retrieval_method,
        score_kind="cosine_similarity",
        threshold=threshold,
        validation_dataset_fingerprint="SYNTHETIC TEST FIXTURE validation fingerprint",
        max_false_accept_rate=0.0,
        max_unsupported_answer_rate=0.0,
        validation_metrics={"coverage": 0.5},
    )


def test_gate_uses_config_specific_score_threshold_and_not_probability() -> None:
    settings = _settings()
    gate = EvidenceGate(_policy(settings))
    low = RetrievedSource(1, "SYNTHETIC TEST FIXTURE low evidence", "synthetic.pdf", 0.8)
    high = RetrievedSource(2, "SYNTHETIC TEST FIXTURE high evidence", "synthetic.pdf", 0.95)

    refused = gate.evaluate([low], settings)
    accepted = gate.evaluate([low, high], settings)

    assert refused.accepted is False
    assert refused.retrieval_score == pytest.approx(0.8)
    assert refused.reason == "insufficient_relevance_evidence"
    assert "confidence" not in refused.to_dict()
    assert accepted.accepted is True
    assert accepted.score_kind == "cosine_similarity"
    with pytest.raises(ValueError, match="does not match"):
        gate.evaluate([high], _settings(corpus_revision="SYNTHETIC TEST FIXTURE corpus v2"))


def test_pipeline_abstains_before_generation_when_gate_rejects() -> None:
    settings = _settings()
    pages = [
        DocumentPage(
            page=4,
            text="SYNTHETIC TEST FIXTURE sample policy text has eight words",
            source_name="SYNTHETIC TEST FIXTURE source.pdf",
            document_id="synthetic-source",
        )
    ]
    chunks = (TextChunk(4, pages[0].text, pages[0].source_name, pages[0].document_id, "synthetic-chunk"),)
    kb = KnowledgeBase("synthetic", chunks, np.asarray([[0.8, 0.6]], dtype=np.float32))

    class NeverCalledGenerator:
        called = False

        def generate(self, *args, **kwargs):
            self.called = True
            raise AssertionError("generation must be skipped when the gate refuses")

    generator = NeverCalledGenerator()
    result = answer(
        "SYNTHETIC TEST FIXTURE campus policy question",
        kb,
        FakeEmbedder(),
        generator,
        settings,
        evidence_gate=EvidenceGate(_policy(settings, threshold=0.95)),
    )

    assert result.status is AnswerStatus.ABSTAINED
    assert result.answer == "The retrieved passages do not establish the requested policy detail."
    assert result.sources[0].page_number == 4
    assert result.gate_decision["accepted"] is False
    assert result.gate_decision["threshold"] == 0.95
    assert result.gate_decision["retrieval_score"] == pytest.approx(0.8)
    assert generator.called is False


def test_gate_policy_round_trip_from_calibration_json(tmp_path: Path) -> None:
    settings = _settings()
    policy = _policy(settings)
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"policies": [policy.to_dict()]}), encoding="utf-8")

    loaded = EvidenceGate.from_file(
        path,
        embedding_model=settings.embedding_model,
        configuration_id=policy.configuration_id,
    )
    assert loaded.policy == policy


@pytest.mark.parametrize(
    ("scenario", "score", "source_document", "expected"),
    [
        ("near-threshold", 0.90, "synthetic-primary", True),
        ("keyword-heavy", 0.97, "synthetic-primary", True),
        ("unfamiliar-policy-term", 0.62, "synthetic-primary", False),
        ("unanswerable-no-evidence", None, None, False),
        ("answerable-from-second-document", 0.93, "synthetic-second-document", True),
    ],
)
def test_gate_synthetic_query_scenarios(scenario, score, source_document, expected) -> None:
    settings = _settings()
    sources = (
        []
        if score is None
        else [
            RetrievedSource(
                page=3,
                text=f"SYNTHETIC TEST FIXTURE evidence for {scenario}",
                source_name=f"SYNTHETIC TEST FIXTURE {source_document}.pdf",
                score=score,
                document_id=source_document,
                chunk_id=f"synthetic-{scenario}",
            )
        ]
    )

    decision = EvidenceGate(_policy(settings)).evaluate(sources, settings)

    assert decision.accepted is expected
    assert decision.score_kind == "cosine_similarity"
    if scenario == "answerable-from-second-document":
        assert sources[0].document_id == "synthetic-second-document"
