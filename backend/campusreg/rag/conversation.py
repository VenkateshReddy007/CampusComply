"""Persistent, isolated conversation context for reference-aware RAG follow-ups."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from campusreg.config import RAGSettings
from campusreg.rag.errors import GenerationError
from campusreg.rag.generation import TextGenerator, generate
from campusreg.rag.models import AnswerResult, AnswerStatus, ErrorInfo, RetrievedSource
from campusreg.rag.scope import is_campus_education_question
from campusreg.storage.models import ConversationMessage
from campusreg.storage.sqlite_store import SQLiteDocumentStore

QUERY_REWRITE_SYSTEM = (
    "Rewrite the current user question as one standalone retrieval query. Use conversation history only "
    "to resolve references such as 'that deadline' or 'what about it'. Prior assistant statements are not "
    "official policy evidence and must not be repeated as facts. Do not answer the question, add policy details, "
    "or output a citation. If the history supports multiple plausible referents, return exactly AMBIGUOUS. "
    "Otherwise return only the rewritten question."
)
FOLLOW_UP_PATTERNS = (
    re.compile(r"^\s*(?:what|how)\s+about\b", re.I),
    re.compile(r"^\s*(?:(?:and|but|also|then)\s+)?(?:it|this|that|those|them|there|same)\b", re.I),
    re.compile(r"^\s*(?:does\s+that|would\s+that|what\s+if\s+it)\b", re.I),
    re.compile(r"^\s*then\s+what\b", re.I),
)


class ConversationAnswerer(Protocol):
    def answer(self, question: str, *, retrieval_query: str | None = None) -> AnswerResult:
        """Answer original question using evidence retrieved for retrieval_query."""


class QueryRewriter(Protocol):
    def rewrite(
        self,
        question: str,
        history: Sequence[ConversationMessage],
    ) -> str | None:
        """Return a standalone retrieval query, or None when context is ambiguous."""


class GemmaQueryRewriter:
    """Use the configured Gemma provider only to resolve query references."""

    def __init__(self, generator: TextGenerator, settings: RAGSettings) -> None:
        self.generator = generator
        self.settings = settings

    def rewrite(self, question: str, history: Sequence[ConversationMessage]) -> str | None:
        transcript = "\n".join(f"{message.role.title()}: {message.content}" for message in history)
        prompt = f"Conversation history (reference context only):\n{transcript}\n\nCurrent question:\n{question}"
        try:
            rewritten = generate(
                prompt,
                self.generator,
                self.settings,
                system_text=QUERY_REWRITE_SYSTEM,
            )
        except GenerationError:
            raise
        value = rewritten.strip().strip('"').strip()
        if value.casefold() == "ambiguous":
            return None
        if not value or len(value) > 1000 or "\n" in value:
            raise ValueError("query rewriter returned an invalid standalone query")
        return value


@dataclass(frozen=True, slots=True)
class ConversationTurnResult:
    conversation_id: str
    original_question: str
    retrieval_query: str | None
    rewritten: bool
    result: AnswerResult
    user_message_id: int
    assistant_message_id: int


def requires_query_rewrite(question: str) -> bool:
    text = question.strip()
    return bool(text and any(pattern.search(text) for pattern in FOLLOW_UP_PATTERNS))


def _bounded_history(
    history: Sequence[ConversationMessage], *, max_messages: int, max_chars: int
) -> tuple[ConversationMessage, ...]:
    selected = list(history[-max_messages:])
    remaining = max_chars
    bounded: list[ConversationMessage] = []
    for message in reversed(selected):
        if remaining <= 0:
            break
        content = message.content[-remaining:]
        remaining -= len(content)
        bounded.append(
            ConversationMessage(
                id=message.id,
                conversation_id=message.conversation_id,
                role=message.role,
                content=content,
                created_at=message.created_at,
                original_question=message.original_question,
                retrieval_query=message.retrieval_query,
                answer_status=message.answer_status,
                sources=(),
            )
        )
    return tuple(reversed(bounded))


class ConversationRAGService:
    """Manage chat history while retrieving fresh official evidence each turn."""

    def __init__(
        self,
        store: SQLiteDocumentStore,
        answerer: ConversationAnswerer,
        rewriter: QueryRewriter | None,
        *,
        max_history_messages: int = 8,
        max_history_chars: int = 4000,
    ) -> None:
        if max_history_messages <= 0 or max_history_chars <= 0:
            raise ValueError("conversation history bounds must be positive")
        self.store = store
        self.answerer = answerer
        self.rewriter = rewriter
        self.max_history_messages = max_history_messages
        self.max_history_chars = max_history_chars

    def create_conversation(self) -> str:
        return self.store.create_conversation()

    def clear_conversation(self, conversation_id: str) -> bool:
        return self.store.clear_conversation(conversation_id)

    def is_in_scope(self, conversation_id: str, question: str) -> bool:
        """Check the question plus context-only references in prior user turns."""

        history = self.store.list_conversation_messages(conversation_id)
        return is_campus_education_question(
            question,
            prior_user_questions=(message.content for message in history if message.role == "user"),
        )

    def delete_conversation(self, conversation_id: str) -> bool:
        return self.store.delete_conversation(conversation_id)

    def ask(self, conversation_id: str, question: str) -> ConversationTurnResult:
        original = question.strip()
        if not original:
            raise ValueError("question must not be empty")
        history = self.store.list_conversation_messages(conversation_id)
        user_message = self.store.append_conversation_message(
            conversation_id,
            role="user",
            content=original,
            original_question=original,
        )
        follow_up = requires_query_rewrite(original)
        if follow_up and not history:
            result = self._clarification_result()
            assistant = self._store_assistant(conversation_id, original, None, result)
            return ConversationTurnResult(
                conversation_id, original, None, False,
                result, user_message.id, assistant.id,
            )
        if not is_campus_education_question(
            original,
            prior_user_questions=(message.content for message in history if message.role == "user"),
        ):
            result = self.answerer.answer(original, retrieval_query=original)
            assistant = self._store_assistant(conversation_id, original, original, result)
            return ConversationTurnResult(
                conversation_id, original, original, False,
                result, user_message.id, assistant.id,
            )
        retrieval_query: str | None = original
        rewritten = False
        if follow_up:
            if not history or self.rewriter is None:
                result = self._clarification_result()
                retrieval_query = None
                retrieval_query_value = None
                assistant = self._store_assistant(
                    conversation_id, original, retrieval_query_value, result
                )
                return ConversationTurnResult(
                    conversation_id, original, retrieval_query_value, False,
                    result, user_message.id, assistant.id,
                )
            try:
                bounded = _bounded_history(
                    history,
                    max_messages=self.max_history_messages,
                    max_chars=self.max_history_chars,
                )
                rewritten_query = self.rewriter.rewrite(original, bounded)
                retrieval_query = rewritten_query.strip() if rewritten_query else ""
                if not retrieval_query:
                    result = self._clarification_result(ambiguous=True)
                    assistant = self._store_assistant(conversation_id, original, None, result)
                    return ConversationTurnResult(
                        conversation_id, original, None, False,
                        result, user_message.id, assistant.id,
                    )
                if len(retrieval_query) > 1000:
                    raise ValueError("query rewriter returned an invalid standalone query")
                rewritten = True
            except Exception:
                result = self._clarification_result(rewrite_failed=True)
                retrieval_query = None
                assistant = self._store_assistant(conversation_id, original, retrieval_query, result)
                return ConversationTurnResult(
                    conversation_id, original, retrieval_query, False,
                    result, user_message.id, assistant.id,
                )

        try:
            result = self.answerer.answer(original, retrieval_query=retrieval_query)
            if not isinstance(result, AnswerResult):
                raise TypeError("conversation answerer must return AnswerResult")
            if result.retrieval_query is None:
                result = replace(result, retrieval_query=retrieval_query)
        except Exception:
            result = AnswerResult(
                status=AnswerStatus.GENERATION_ERROR,
                answer=None,
                sources=(),
                error=ErrorInfo("conversation_answer_failed", "The answer request could not be completed."),
                retrieval_query=retrieval_query,
            )
        assistant = self._store_assistant(conversation_id, original, retrieval_query, result)
        return ConversationTurnResult(
            conversation_id=conversation_id,
            original_question=original,
            retrieval_query=retrieval_query,
            rewritten=rewritten,
            result=result,
            user_message_id=user_message.id,
            assistant_message_id=assistant.id,
        )

    @staticmethod
    def _clarification_result(
        *, rewrite_failed: bool = False, ambiguous: bool = False
    ) -> AnswerResult:
        return AnswerResult(
            status=AnswerStatus.ABSTAINED,
            answer=(
                "Please clarify which earlier topic you mean, or restate the question on its own."
                if ambiguous
                else "Please restate the follow-up as a standalone question so I can retrieve the right official document."
            ),
            sources=(),
            error=(
                ErrorInfo("query_rewrite_failed", "The follow-up could not be resolved from conversation context.")
                if rewrite_failed
                else ErrorInfo(
                    "ambiguous_follow_up" if ambiguous else "follow_up_context_missing",
                    "The follow-up could not be resolved to one prior topic." if ambiguous
                    else "No prior context is available to resolve this follow-up.",
                )
            ),
        )

    def _store_assistant(
        self,
        conversation_id: str,
        original_question: str,
        retrieval_query: str | None,
        result: AnswerResult,
    ) -> ConversationMessage:
        source_rows = [
            {
                "document_id": source.document_id,
                "document_name": source.source_name,
                "page_number": source.page_number,
                "chunk_id": source.chunk_id,
                "citation": source.citation,
                "text": source.text,
                "score": source.score,
                "retrieval_method": source.retrieval_method,
                "diagnostics": dict(source.diagnostics),
            }
            for source in result.sources
        ]
        return self.store.append_conversation_message(
            conversation_id,
            role="assistant",
            content=result.answer or "",
            original_question=original_question,
            retrieval_query=retrieval_query,
            answer_status=result.status.value,
            sources=source_rows,
        )
