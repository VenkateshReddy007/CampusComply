"""Local request guardrails for the document-grounded campus assistant.

These conservative phrase rules are an early deterministic screen, not a
general-purpose safety classifier. They deliberately leave legitimate review,
appeal, clarification and fairness-improvement questions available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_DIRECT_EXPLOIT = re.compile(
    r"\b(?:exploit|abuse)\b.{0,60}\b(?:loopholes?|exceptions?|gaps?|weakness(?:es)?|"
    r"technicalities|rules?|polic(?:y|ies))\b|"
    r"\b(?:bypass|circumvent|evade)\b.{0,45}\b(?:rule|requirement|eligibility|"
    r"attendance|compliance|policy|procedure|verification)\b|"
    r"\breverse.engineer\b.{0,80}\b(?:unfair|advantage|bypass|exploit)\b|"
    r"\bmanipulat\w*\b.{0,50}\b(?:exception|eligibility|procedure|requirement)\b",
    re.IGNORECASE | re.DOTALL,
)
_SAFE_REVIEW_INTENT = re.compile(
    r"\b(?:appeal|clarif(?:y|ication)|ambigu(?:ity|ous)|fairness|improv\w*|"
    r"amend\w*|reform\w*|challenge\s+(?:a|the|this)\s+decision)\b",
    re.IGNORECASE,
)
_EXPLICIT_ACTIONABLE_INTENT = re.compile(
    r"\b(?:exploit|abuse|bypass|circumvent|evade|manipulat\w*|reverse.engineer)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class GuardrailDecision:
    blocked: bool
    reason_code: str
    message: str | None = None


def preflight_request(question: str) -> GuardrailDecision:
    """Reject clearly actionable rule-evasion requests before any AI call."""

    if not isinstance(question, str) or not question.strip():
        return GuardrailDecision(False, "empty_or_invalid_request")
    if not _DIRECT_EXPLOIT.search(question):
        return GuardrailDecision(False, "allowed")
    review_intent = bool(_SAFE_REVIEW_INTENT.search(question))
    actionable = bool(_EXPLICIT_ACTIONABLE_INTENT.search(question))
    if review_intent and not actionable:
        return GuardrailDecision(False, "legitimate_review_or_clarification")
    return GuardrailDecision(
        True,
        "rule_evasion_or_unfair_advantage",
        "I cannot help exploit or evade campus rules. I can help explain the documented requirement, outline a legitimate appeal or clarification request, or suggest fair ways to improve the rule.",
    )
