"""Conservative campus and education topic gate for user questions."""

from __future__ import annotations

import re
from collections.abc import Iterable


_EDUCATION_TERMS = re.compile(
    r"\b(?:campus(?:es)?|universit(?:y|ies)|colleges?|schools?|education(?:al)?|"
    r"students?|faculty|professors?|teachers?|academics?|courses?|classes|classroom|"
    r"curricul(?:um|a)|syllab(?:us|i)|semesters?|admissions?|enrol(?:l)?ment|"
    r"exams?|examinations?|attendance|grades?|grading|scholarships?|tuition|fees?|"
    r"hostels?|dorm(?:s|itories)?|libraries|library|transcripts?|degrees?|registrar|"
    r"dean|financial aid|student services)\b",
    re.IGNORECASE,
)
_CONTEXT_ONLY_FOLLOW_UP = re.compile(
    r"\s*(?:(?:and|but|also|then)\s+)?(?:(?:what|how)\s+about\s+)?"
    r"(?:it|this|that|these|those|them|there|same|so|then)\s*[?!.]*\s*",
    re.IGNORECASE,
)
_FOLLOW_UP_SHAPE = re.compile(r"^\s*(?:what|how)\s+about\b|^\s*(?:does|would)\s+that\b|^\s*what\s+if\b", re.I)


def is_campus_education_question(
    question: str,
    *,
    prior_user_questions: Iterable[str] = (),
) -> bool:
    """Recognize topic-bound questions and context-only follow-ups.

    This lexical boundary only rejects obviously unrelated questions; it does
    not decide whether a policy is true or supported by indexed documents.
    """

    if _EDUCATION_TERMS.search(question):
        return True
    if _FOLLOW_UP_SHAPE.search(question):
        return any(_EDUCATION_TERMS.search(previous) for previous in prior_user_questions)
    if not _CONTEXT_ONLY_FOLLOW_UP.fullmatch(question):
        return False
    return any(_EDUCATION_TERMS.search(previous) for previous in prior_user_questions)
