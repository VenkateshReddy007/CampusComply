"""Conversation tests use synthetic history and mocked policy answers only."""

from __future__ import annotations

from campusreg.config import RAGSettings
from campusreg.rag.conversation import ConversationRAGService, GemmaQueryRewriter
from campusreg.evaluation.models import ConversationTurn, EvaluationQuestion, ExpectedPage
from campusreg.evaluation.runner import EvaluationRunner
from campusreg.rag.models import AnswerResult, AnswerStatus, RetrievedSource
from campusreg.storage.sqlite_store import SQLiteDocumentStore


class FakeAnswerer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []
        self.result = AnswerResult(
            status=AnswerStatus.ANSWERED,
            answer="SYNTHETIC TEST FIXTURE answer [fixture.pdf, document fixture-id, p. 1]",
            sources=(RetrievedSource(
                page=1,
                text="SYNTHETIC TEST FIXTURE supporting passage",
                source_name="fixture.pdf",
                score=0.7,
                document_id="fixture-id",
                chunk_id="fixture-chunk",
            ),),
        )

    def answer(self, question: str, *, retrieval_query: str | None = None) -> AnswerResult:
        self.calls.append((question, retrieval_query))
        return self.result


class FakeRewriter:
    def __init__(self, values: list[str | None]) -> None:
        self.values = values
        self.histories = []

    def rewrite(self, question: str, history) -> str | None:
        self.histories.append((question, tuple(history)))
        return self.values.pop(0)


def _service(rewriter: FakeRewriter | None = None):
    store = SQLiteDocumentStore(":memory:")
    answerer = FakeAnswerer()
    return store, answerer, ConversationRAGService(store, answerer, rewriter)


def test_follow_up_rewrites_and_retrieves_again_storing_audit_fields() -> None:
    rewriter = FakeRewriter(["SYNTHETIC TEST FIXTURE registration deadline"])
    store, answerer, service = _service(rewriter)
    conversation_id = service.create_conversation()
    service.ask(conversation_id, "SYNTHETIC TEST FIXTURE When is registration?")

    turn = service.ask(conversation_id, "What about the deadline?")

    assert turn.rewritten is True
    assert turn.original_question == "What about the deadline?"
    assert turn.retrieval_query == "SYNTHETIC TEST FIXTURE registration deadline"
    assert answerer.calls[-1] == ("What about the deadline?", turn.retrieval_query)
    messages = store.list_conversation_messages(conversation_id)
    assert [message.role for message in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[-1].original_question == "What about the deadline?"
    assert messages[-1].retrieval_query == turn.retrieval_query
    assert messages[-1].sources[0]["text"] == "SYNTHETIC TEST FIXTURE supporting passage"


def test_ambiguous_follow_up_abstains_without_retrieval() -> None:
    rewriter = FakeRewriter([None])
    _, answerer, service = _service(rewriter)
    conversation_id = service.create_conversation()
    service.ask(conversation_id, "SYNTHETIC TEST FIXTURE ask about registration")

    turn = service.ask(conversation_id, "What about it?")

    assert turn.result.status is AnswerStatus.ABSTAINED
    assert turn.result.error is not None and turn.result.error.code == "ambiguous_follow_up"
    assert len(answerer.calls) == 1


def test_gemma_query_rewriter_uses_ambiguity_sentinel_without_answering() -> None:
    class Generator:
        def generate(self, prompt, **kwargs):
            assert "Prior assistant statements are not" in kwargs["system_instruction"]
            assert "SYNTHETIC TEST FIXTURE" in prompt
            return "AMBIGUOUS"

    store = SQLiteDocumentStore(":memory:")
    conversation_id = store.create_conversation()
    history = [store.append_conversation_message(
        conversation_id, role="user", content="SYNTHETIC TEST FIXTURE one possible topic"
    )]
    rewriter = GemmaQueryRewriter(Generator(), RAGSettings.baseline())
    assert rewriter.rewrite("What about it?", history) is None


def test_no_context_follow_up_requests_clarification() -> None:
    _, answerer, service = _service(FakeRewriter([]))
    conversation_id = service.create_conversation()
    turn = service.ask(conversation_id, "What about the deadline?")
    assert turn.result.status is AnswerStatus.ABSTAINED
    assert turn.result.error is not None and turn.result.error.code == "follow_up_context_missing"
    assert answerer.calls == []


def test_topic_change_does_not_carry_prior_topic_into_query() -> None:
    rewriter = FakeRewriter([])
    _, answerer, service = _service(rewriter)
    conversation_id = service.create_conversation()
    service.ask(conversation_id, "SYNTHETIC TEST FIXTURE registration deadline?")

    turn = service.ask(conversation_id, "SYNTHETIC TEST FIXTURE Explain the library opening hours.")

    assert turn.rewritten is False
    assert answerer.calls[-1] == (turn.original_question, turn.original_question)
    assert rewriter.histories == []


def test_conversation_isolation_clear_and_delete() -> None:
    rewriter = FakeRewriter(["SYNTHETIC TEST FIXTURE unrelated standalone query"])
    store, _, service = _service(rewriter)
    first = service.create_conversation()
    second = service.create_conversation()
    service.ask(first, "SYNTHETIC TEST FIXTURE registration policy")

    isolated_follow_up = service.ask(second, "What about it?")
    assert isolated_follow_up.result.status is AnswerStatus.ABSTAINED
    assert len(rewriter.histories) == 0

    assert service.clear_conversation(first)
    assert store.conversation_exists(first)
    assert store.list_conversation_messages(first) == []
    assert service.delete_conversation(first)
    assert not store.conversation_exists(first)
    assert store.conversation_exists(second)


def test_unanswerable_follow_up_remains_abstained_after_fresh_retrieval() -> None:
    rewriter = FakeRewriter(["SYNTHETIC TEST FIXTURE absent policy detail"])
    _, answerer, service = _service(rewriter)
    answerer.result = AnswerResult(
        status=AnswerStatus.ABSTAINED,
        answer="The answer could not be found in the document.",
        sources=(),
    )
    conversation_id = service.create_conversation()
    service.ask(conversation_id, "SYNTHETIC TEST FIXTURE Ask about an available policy")

    turn = service.ask(conversation_id, "What about the missing detail?")

    assert answerer.calls[-1] == (turn.original_question, "SYNTHETIC TEST FIXTURE absent policy detail")
    assert turn.result.status is AnswerStatus.ABSTAINED
    assert turn.result.sources == ()


def test_answer_generation_exception_is_stored_as_structured_failure() -> None:
    class BrokenAnswerer:
        def answer(self, question, *, retrieval_query=None):
            raise RuntimeError("synthetic provider failure")

    store = SQLiteDocumentStore(":memory:")
    service = ConversationRAGService(store, BrokenAnswerer(), None)
    conversation_id = service.create_conversation()

    turn = service.ask(conversation_id, "SYNTHETIC TEST FIXTURE standalone question")

    assert turn.result.status is AnswerStatus.GENERATION_ERROR
    assert turn.result.answer is None
    assert turn.result.error is not None
    messages = store.list_conversation_messages(conversation_id)
    assert messages[-1].answer_status == AnswerStatus.GENERATION_ERROR.value
    assert messages[-1].content == ""


def test_history_sent_to_rewriter_is_bounded_and_drops_evidence_sources() -> None:
    rewriter = FakeRewriter(["SYNTHETIC TEST FIXTURE standalone query"])
    store, _, _ = _service(rewriter)
    service = ConversationRAGService(
        store, FakeAnswerer(), rewriter, max_history_messages=2, max_history_chars=30
    )
    conversation_id = service.create_conversation()
    store.append_conversation_message(
        conversation_id, role="user", content="SYNTHETIC TEST FIXTURE old long context " * 5
    )
    store.append_conversation_message(
        conversation_id, role="assistant", content="SYNTHETIC TEST FIXTURE prior answer",
        sources=[{"text": "SYNTHETIC TEST FIXTURE source"}],
    )

    service.ask(conversation_id, "What about the deadline?")

    bounded = rewriter.histories[0][1]
    assert len(bounded) <= 2
    assert sum(len(item.content) for item in bounded) <= 30
    assert all(item.sources == () for item in bounded)


def test_evaluation_runner_uses_conversation_service_for_follow_up(tmp_path) -> None:
    rewriter = FakeRewriter(["SYNTHETIC TEST FIXTURE registration deadline"])
    store, answerer, service = _service(rewriter)
    question = EvaluationQuestion(
        id="synthetic-follow-up",
        question="What about the deadline?",
        category="follow_up",
        expected_answer="SYNTHETIC TEST FIXTURE expected answer",
        answerable=True,
        expected_document_ids=("fixture-id",),
        expected_pages=(ExpectedPage("fixture-id", 1),),
        expected_conflict=False,
        split="development",
        conversation_context=(
            ConversationTurn("user", "SYNTHETIC TEST FIXTURE ask about registration"),
            ConversationTurn("assistant", "SYNTHETIC TEST FIXTURE prior assistant answer"),
        ),
    )

    run = EvaluationRunner().run(
        [question],
        answerer=answerer.answer,
        conversation_service=service,
        output_dir=tmp_path,
        split="development",
        retrieval_k=3,
        run_config={
            "gemma_model": "test-model",
            "embedding_model": "test-embedding",
            "index_revision": "synthetic-index",
            "prompt_revision": "synthetic-prompt",
            "conversation_revision": "synthetic-conversation-v1",
        },
    )

    prediction = run.predictions[0]
    assert answerer.calls[-1] == ("What about the deadline?", "SYNTHETIC TEST FIXTURE registration deadline")
    assert prediction.question == question.question
    assert prediction.retrieval_query == "SYNTHETIC TEST FIXTURE registration deadline"
    assert prediction.citation_valid is True
    assert store.list_documents() == []
