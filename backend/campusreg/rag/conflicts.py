"""Conservative, evidence-validated cross-document conflict detection."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping, Protocol, Sequence

from campusreg.config import RAGSettings
from campusreg.rag.generation import TextGenerator, generate
from campusreg.rag.models import RetrievedSource
from campusreg.storage.models import DocumentRecord


class ConflictStatus(StrEnum):
    POTENTIAL_CONTRADICTION = "potential_contradiction"
    POSSIBLE_SUPERSESSION = "possible_supersession"
    DIFFERENT_SCOPES = "different_scopes"
    VALID_EXCEPTION = "valid_exception"
    NOT_CONFLICTING = "not_conflicting"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


POSITIVE_CONFLICT_STATUSES = frozenset({
    ConflictStatus.POTENTIAL_CONTRADICTION,
    ConflictStatus.POSSIBLE_SUPERSESSION,
})


@dataclass(frozen=True, slots=True)
class ConflictPassage:
    document_id: str
    document_name: str
    page: int
    chunk_id: str
    text: str

    @classmethod
    def from_source(cls, source: RetrievedSource) -> "ConflictPassage":
        return cls(source.document_id, source.source_name, source.page, source.chunk_id, source.text)

    @property
    def citation(self) -> str:
        return f"[{self.document_name}, document {self.document_id}, p. {self.page}]"


@dataclass(frozen=True, slots=True)
class ConflictReport:
    subject: str
    status: ConflictStatus
    passage_a: ConflictPassage
    passage_b: ConflictPassage
    explanation: str
    scope_a: str | None = None
    scope_b: str | None = None
    effective_date_a: str | None = None
    effective_date_b: str | None = None
    limitations: tuple[str, ...] = ()
    human_review_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        def passage(value: ConflictPassage) -> dict[str, Any]:
            return {
                "document_id": value.document_id, "document_name": value.document_name,
                "page": value.page, "chunk_id": value.chunk_id,
                "citation": value.citation, "text": value.text,
            }
        return {
            "subject": self.subject, "status": self.status.value,
            "passage_a": passage(self.passage_a), "passage_b": passage(self.passage_b),
            "explanation": self.explanation,
            "scope": {"passage_a": self.scope_a, "passage_b": self.scope_b},
            "effective_dates": {"passage_a": self.effective_date_a, "passage_b": self.effective_date_b},
            "limitations": list(self.limitations), "human_review_required": self.human_review_required,
        }


@dataclass(frozen=True, slots=True)
class ConflictAssessment:
    subject: str
    status: ConflictStatus
    explanation: str
    quote_a: str
    quote_b: str
    scope_a: str | None = None
    scope_b: str | None = None
    date_a: str | None = None
    date_b: str | None = None
    limitations: tuple[str, ...] = ()


class ConflictComparator(Protocol):
    def compare(
        self, passage_a: ConflictPassage, passage_b: ConflictPassage, *,
        metadata_a: DocumentRecord | None = None, metadata_b: DocumentRecord | None = None,
    ) -> ConflictAssessment: ...


class GemmaConflictComparator:
    """Structured Gemma comparator; source passages are verified after parsing."""

    _SYSTEM = (
        "Compare two supplied regulation passages. Use only supplied evidence. A difference in values "
        "alone is not necessarily a conflict. Classify as potential_contradiction, possible_supersession, "
        "different_scopes, valid_exception, not_conflicting, or insufficient_evidence. Never decide which "
        "source is authoritative. Return only JSON with subject, status, explanation, quote_a, quote_b, "
        "scope_a, scope_b, date_a, date_b, limitations. Quotes and scope values must be verbatim; dates "
        "must be copied from passage text or verified effective-date metadata. Use null if unknown. "
        "limitations must be an array of strings. Do not return document IDs or page numbers."
    )

    def __init__(self, generator: TextGenerator, settings: RAGSettings) -> None:
        self.generator = generator
        self.settings = settings

    def compare(
        self, passage_a: ConflictPassage, passage_b: ConflictPassage, *,
        metadata_a: DocumentRecord | None = None, metadata_b: DocumentRecord | None = None,
    ) -> ConflictAssessment:
        prompt = json.dumps({
            "passage_a": {"text": passage_a.text, "effective_date": _effective_date(metadata_a)},
            "passage_b": {"text": passage_b.text, "effective_date": _effective_date(metadata_b)},
        }, ensure_ascii=False)
        raw = generate(prompt, self.generator, self.settings, system_text=self._SYSTEM)
        try:
            return _parse_assessment(json.loads(raw))
        except (json.JSONDecodeError, TypeError, ValueError, KeyError):
            return ConflictAssessment(
                "Unclear regulatory subject", ConflictStatus.INSUFFICIENT_EVIDENCE,
                "The comparison response was not valid structured evidence.", passage_a.text, passage_b.text,
                limitations=("Automated comparison output could not be validated.",),
            )


def _effective_date(record: DocumentRecord | None) -> str | None:
    return record.effective_date.isoformat() if record and record.effective_date else None


def _parse_assessment(payload: Mapping[str, Any]) -> ConflictAssessment:
    if not isinstance(payload, Mapping):
        raise TypeError("comparison result must be an object")
    status = ConflictStatus(payload["status"])
    names = ("subject", "explanation", "quote_a", "quote_b")
    required = {name: payload[name] for name in names}
    if any(not isinstance(value, str) or not value.strip() for value in required.values()):
        raise ValueError("required comparison fields must be non-empty strings")
    limitations = payload.get("limitations", [])
    if not isinstance(limitations, list) or any(not isinstance(item, str) for item in limitations):
        raise ValueError("limitations must be an array of strings")
    optional: dict[str, str | None] = {}
    for name in ("scope_a", "scope_b", "date_a", "date_b"):
        value = payload.get(name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{name} must be a non-empty string or null")
        optional[name] = value
    return ConflictAssessment(**required, status=status, **optional, limitations=tuple(limitations))


_TOKEN = re.compile(r"[\w]+", re.UNICODE)
_STOP_WORDS = frozenset("a an and are as at be by can for from has have in is it of on or the this to under was were with rule policy regulation requirement synthetic fixture".split())


def candidate_pairs(
    passages: Sequence[ConflictPassage], *, min_shared_terms: int = 2, max_pairs: int = 1000,
) -> list[tuple[ConflictPassage, ConflictPassage]]:
    """Propose deterministic cross-document pairs sharing non-stopword terms."""
    if min_shared_terms < 1 or max_pairs < 1:
        raise ValueError("candidate limits must be positive")
    terms = [
        {word for word in _TOKEN.findall(p.text.casefold()) if word not in _STOP_WORDS}
        for p in passages
    ]
    pairs: list[tuple[ConflictPassage, ConflictPassage]] = []
    for left in range(len(passages)):
        for right in range(left + 1, len(passages)):
            a, b = passages[left], passages[right]
            if not a.document_id or not b.document_id or a.document_id == b.document_id:
                continue
            if len(terms[left] & terms[right]) >= min_shared_terms:
                pairs.append((a, b))
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


class ConflictDetector:
    """Compare candidate passages and validate exact evidence before reporting."""

    def __init__(self, comparator: ConflictComparator, *, min_shared_terms: int = 2, max_pairs: int = 1000) -> None:
        self.comparator = comparator
        self.min_shared_terms = min_shared_terms
        self.max_pairs = max_pairs

    def detect(
        self, sources: Sequence[RetrievedSource | ConflictPassage], *,
        documents: Mapping[str, DocumentRecord] | None = None,
    ) -> list[ConflictReport]:
        passages = [x if isinstance(x, ConflictPassage) else ConflictPassage.from_source(x) for x in sources]
        records = documents or {}
        reports: list[ConflictReport] = []
        for a, b in candidate_pairs(passages, min_shared_terms=self.min_shared_terms, max_pairs=self.max_pairs):
            try:
                assessment = self.comparator.compare(
                    a, b, metadata_a=records.get(a.document_id), metadata_b=records.get(b.document_id),
                )
                reports.append(self._validate(assessment, a, b, records))
            except Exception:
                reports.append(ConflictReport(
                    "Unclear regulatory subject", ConflictStatus.INSUFFICIENT_EVIDENCE, a, b,
                    "The passages could not be compared reliably.",
                    limitations=("Comparison failed; human review is required.",),
                ))
        return reports

    @staticmethod
    def _validate(
        assessment: ConflictAssessment, a: ConflictPassage, b: ConflictPassage,
        records: Mapping[str, DocumentRecord],
    ) -> ConflictReport:
        if not isinstance(assessment, ConflictAssessment):
            raise ValueError("unsupported comparator result")
        if not assessment.quote_a or assessment.quote_a not in a.text:
            raise ValueError("quote_a is not present in indexed passage")
        if not assessment.quote_b or assessment.quote_b not in b.text:
            raise ValueError("quote_b is not present in indexed passage")
        if not assessment.subject.strip() or not assessment.explanation.strip():
            raise ValueError("subject and explanation are required")
        for scope, passage in ((assessment.scope_a, a), (assessment.scope_b, b)):
            if scope is not None and scope not in passage.text:
                raise ValueError("scope must be a verbatim passage excerpt")
        date_a = _verified_date(assessment.date_a, a, records.get(a.document_id))
        date_b = _verified_date(assessment.date_b, b, records.get(b.document_id))
        return ConflictReport(
            subject=assessment.subject.strip(), status=assessment.status, passage_a=a, passage_b=b,
            explanation=assessment.explanation.strip(), scope_a=assessment.scope_a, scope_b=assessment.scope_b,
            effective_date_a=date_a, effective_date_b=date_b,
            limitations=assessment.limitations or ("Potential conflict labels are not an authority determination.",),
            human_review_required=True,
        )


def _verified_date(value: str | None, passage: ConflictPassage, record: DocumentRecord | None) -> str | None:
    if value is None:
        return None
    if value in passage.text or (record and record.effective_date and value == record.effective_date.isoformat()):
        return value
    raise ValueError("date was not present in passage or verified document metadata")


def conflict_metrics(labelled: Sequence[tuple[str, bool, bool]]) -> dict[str, Any]:
    """Compute binary conflict precision/recall and retain false-case IDs."""
    tp = fp = fn = tn = 0
    false_positives: list[str] = []
    false_negatives: list[str] = []
    for item in labelled:
        if len(item) != 3:
            raise ValueError("each labelled item must contain id, prediction, and gold")
        example_id, predicted, gold = item
        if predicted and gold:
            tp += 1
        elif predicted:
            fp += 1
            false_positives.append(example_id)
        elif gold:
            fn += 1
            false_negatives.append(example_id)
        else:
            tn += 1
    return {
        "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "false_positive_ids": false_positives, "false_negative_ids": false_negatives,
    }
