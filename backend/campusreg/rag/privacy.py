"""Local, deterministic redaction for common explicitly labelled PII fields.

This is a defense-in-depth preprocessor, not a complete PII detector. It never
uses a hosted model and intentionally only claims coverage for the patterns
listed below.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, replace
from typing import Iterable

from campusreg.rag.models import DocumentPage


_DETECTORS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("STUDENT_ID", re.compile(r"\b(?:student|employee|roll|enrollment|registration)\s*(?:id|number|no\.?|#)\s*[:#-]?\s*[A-Z0-9/-]{4,24}\b", re.I)),
    ("DATE_OF_BIRTH", re.compile(r"\b(?:date\s+of\s+birth|dob|birth\s+date)\s*[:#-]?\s*(?:\d{1,4}[-/.]){2}\d{1,4}\b", re.I)),
    ("EMAIL", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)),
    ("PHONE", re.compile(r"(?<!\w)(?:\+?91[-.\s]?)?[6-9]\d{9}(?!\w)|(?<!\w)\+?1[-.\s]?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}(?!\w)")),
    ("ADDRESS", re.compile(r"\b(?:home\s+)?address\s*[:#-]\s*[^\r\n]{3,120}", re.I)),
    ("ACCOUNT_ID", re.compile(r"\b(?:account|bank|payment)\s*(?:id|number|no\.?|#)\s*[:#-]?\s*[A-Z0-9/-]{4,32}\b", re.I)),
    (
        "PERSON",
        re.compile(
            r"(?i:\b(?:student|employee|full)\s+name\s*[:#-]\s*)"
            r"(?P<value>[A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+){0,3})"
        ),
    ),
    ("SIGNATURE", re.compile(r"\b(?:signature|signed\s+by)\s*[:#-]\s*[^\r\n]{1,100}", re.I)),
)


@dataclass(frozen=True, slots=True)
class RedactionSummary:
    counts_by_category: dict[str, int]
    counts_by_page: dict[int, dict[str, int]]

    @property
    def total(self) -> int:
        return sum(self.counts_by_category.values())

    def to_dict(self) -> dict[str, object]:
        return {
            "counts_by_category": dict(sorted(self.counts_by_category.items())),
            "counts_by_page": {
                str(page): dict(sorted(counts.items()))
                for page, counts in sorted(self.counts_by_page.items())
            },
            "total": self.total,
        }


class RedactionSession:
    """Keep repeated values consistently tokenized within one document/request."""

    def __init__(self) -> None:
        self._tokens: dict[tuple[str, str], str] = {}
        self._next: Counter[str] = Counter()

    def redact(self, text: str) -> tuple[str, Counter[str]]:
        candidates: list[tuple[int, int, str, str]] = []
        for category, detector in _DETECTORS:
            for match in detector.finditer(text):
                start, end = match.span("value") if "value" in match.groupdict() else match.span()
                candidates.append((start, end, category, text[start:end]))
        candidates.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))
        accepted: list[tuple[int, int, str, str]] = []
        cursor = -1
        for candidate in candidates:
            start, end, _, _ = candidate
            if start >= cursor:
                accepted.append(candidate)
                cursor = end
        if not accepted:
            return text, Counter()
        output: list[str] = []
        counts: Counter[str] = Counter()
        cursor = 0
        for start, end, category, original in accepted:
            output.append(text[cursor:start])
            key = (category, original.casefold().strip())
            token = self._tokens.get(key)
            if token is None:
                self._next[category] += 1
                token = f"[{category}_{self._next[category]:03d}]"
                self._tokens[key] = token
            output.append(token)
            counts[category] += 1
            cursor = end
        output.append(text[cursor:])
        return "".join(output), counts


def redact_text(text: str, *, session: RedactionSession | None = None) -> tuple[str, dict[str, int]]:
    """Redact supported PII patterns and return safe category counts."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    redacted, counts = (session or RedactionSession()).redact(text)
    return redacted, dict(counts)


def redact_pages(pages: Iterable[DocumentPage]) -> tuple[tuple[DocumentPage, ...], RedactionSummary]:
    """Redact pages with document-wide placeholder continuity and safe page counts."""

    session = RedactionSession()
    clean_pages: list[DocumentPage] = []
    all_counts: Counter[str] = Counter()
    by_page: dict[int, dict[str, int]] = {}
    for page in pages:
        text, counts = session.redact(page.text)
        safe_name, name_counts = session.redact(page.source_name)
        if name_counts:
            counts.update(name_counts)
        clean_pages.append(replace(page, text=text, source_name=safe_name))
        if counts:
            by_page[page.page] = dict(counts)
            all_counts.update(counts)
    return tuple(clean_pages), RedactionSummary(dict(all_counts), by_page)
