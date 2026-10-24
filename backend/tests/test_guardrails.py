"""Deterministic guardrail tests; all policy language is synthetic test input."""

from __future__ import annotations

import numpy as np

from campusreg.config import RAGSettings
from campusreg.rag.guardrails import preflight_request
from campusreg.rag.models import AnswerStatus, KnowledgeBase, TextChunk
from campusreg.rag.pipeline import answer


def test_clear_rule_evasion_request_is_blocked_locally() -> None:
    decision = preflight_request("How can I bypass the campus attendance requirement?")
    assert decision.blocked is True
    assert decision.reason_code == "rule_evasion_or_unfair_advantage"
    assert "legitimate appeal" in decision.message


def test_legitimate_clarification_appeal_and_fairness_requests_remain_allowed() -> None:
    for question in (
        "How do I appeal an attendance decision through the proper process?",
        "Can you explain an ambiguity in this campus rule for a review?",
        "How could the institution improve the fairness of this regulation?",
    ):
        assert preflight_request(question).blocked is False


def test_blocked_query_never_reaches_embedding_or_generation() -> None:
    class RecordingEmbedder:
        called = False

        def encode(self, texts):
            self.called = True
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

    class RecordingGenerator:
        called = False

        def generate(self, prompt, **kwargs):
            self.called = True
            return "must not run"

    embedder = RecordingEmbedder()
    generator = RecordingGenerator()
    kb = KnowledgeBase(
        name="synthetic test source",
        chunks=(TextChunk(1, "SYNTHETIC TEST FIXTURE attendance rule", "fixture.pdf", "fixture", "chunk"),),
        vectors=np.asarray([[1.0, 0.0]], dtype=np.float32),
    )
    result = answer(
        "How can I evade the campus attendance rule?", kb, embedder, generator,
        RAGSettings(relevance_threshold=None),
    )
    assert result.status is AnswerStatus.ABSTAINED
    assert result.sources == ()
    assert result.gate_decision == {"decision": "blocked", "reason_code": "rule_evasion_or_unfair_advantage"}
    assert embedder.called is False
    assert generator.called is False
