from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import pytest

from campusreg.config import RAGSettings
from campusreg.rag import (
    AnswerStatus,
    DocumentPage,
    EmptyDocumentError,
    GoogleGenAIProvider,
    KnowledgeBase,
    MissingAPIConfigurationError,
    RetrievedSource,
    SentenceTransformerEmbedder,
    TextChunk,
    answer,
    build_knowledge_base,
    read_pdf,
    retrieve,
    split_into_chunks,
)


class FakeEmbedder:
    def __init__(self, mapping: dict[str, Sequence[float]]) -> None:
        self.mapping = mapping

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray([self.mapping[text] for text in texts], dtype=np.float32)


def test_word_chunking_preserves_overlap_and_page_metadata() -> None:
    pages = [
        DocumentPage(
            page=4,
            source_name="official-handbook.pdf",
            text="one two three four five six seven eight nine ten",
        ),
        DocumentPage(page=5, source_name="official-handbook.pdf", text="a b c d e"),
    ]

    chunks = split_into_chunks(pages, chunk_size=6, chunk_overlap=2)

    assert [chunk.text for chunk in chunks] == [
        "one two three four five six",
        "five six seven eight nine ten",
    ]
    assert [(chunk.source_name, chunk.page) for chunk in chunks] == [
        ("official-handbook.pdf", 4),
        ("official-handbook.pdf", 4),
    ]
    assert all(chunk.document_id for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_read_pdf_extracts_page_text_and_one_based_metadata(monkeypatch, tmp_path: Path) -> None:
    class FakePage:
        def __init__(self, value: str | None) -> None:
            self.value = value

        def extract_text(self) -> str | None:
            return self.value

    class FakeReader:
        def __init__(self, path: str) -> None:
            assert path.endswith("rules.pdf")
            self.pages = [FakePage("first page"), FakePage(None), FakePage("third page")]

    import campusreg.rag.documents as documents

    monkeypatch.setattr(documents, "PdfReader", FakeReader)
    pdf_path = tmp_path / "rules.pdf"
    pdf_path.touch()

    pages = read_pdf(pdf_path)

    assert pages == [
        DocumentPage(page=1, text="first page", source_name="rules.pdf"),
        DocumentPage(page=2, text="", source_name="rules.pdf"),
        DocumentPage(page=3, text="third page", source_name="rules.pdf"),
    ]


def test_sentence_transformer_adapter_returns_normalized_embedding_dimensions() -> None:
    class FakeSentenceTransformer:
        def get_sentence_embedding_dimension(self):
            return 2

        def encode(self, texts, *, normalize_embeddings, convert_to_numpy):
            assert texts == ["one", "two"]
            assert normalize_embeddings is True
            assert convert_to_numpy is True
            return np.array([[3.0, 4.0], [0.0, 2.0]])

    adapter = SentenceTransformerEmbedder("test-model", model=FakeSentenceTransformer())
    vectors = adapter.encode(["one", "two"])

    assert vectors.shape == (2, 2)
    assert adapter.dimension == 2
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(vectors, [[3.0, 4.0], [0.0, 2.0]])


def test_build_knowledge_base_rejects_invalid_embedding_matrix_dimensions() -> None:
    pages = ["one two three four five six"]

    class BadShapeEmbedder:
        def encode(self, texts: Sequence[str]) -> np.ndarray:
            return np.array([1.0, 0.0], dtype=np.float32)

    settings = RAGSettings(chunk_size=6, chunk_overlap=0)

    with pytest.raises(ValueError, match="one non-empty vector per input text"):
        build_knowledge_base(pages, "synthetic-test.pdf", BadShapeEmbedder(), settings)


@pytest.mark.parametrize("pages", [[], [""], ["one two three four five"]])
def test_empty_documents_do_not_build_a_knowledge_base(pages) -> None:
    with pytest.raises(EmptyDocumentError):
        build_knowledge_base(pages, "synthetic-test.pdf", FakeEmbedder({}))


def test_retrieval_orders_by_cosine_similarity_and_returns_source_evidence() -> None:
    chunks = (
        TextChunk(page=1, text="orthogonal evidence", source_name="synthetic.pdf"),
        TextChunk(page=2, text="direct evidence", source_name="synthetic.pdf"),
        TextChunk(page=3, text="near evidence", source_name="synthetic.pdf"),
    )
    knowledge_base = KnowledgeBase(
        name="synthetic.pdf",
        chunks=chunks,
        vectors=np.array([[0.0, 1.0], [1.0, 0.0], [0.8, 0.6]], dtype=np.float32),
    )
    embedder = FakeEmbedder({"question": [1.0, 0.0]})

    sources = retrieve("question", knowledge_base, embedder, RAGSettings(top_k=3))

    assert [source.page for source in sources] == [2, 3, 1]
    assert [source.score for source in sources] == pytest.approx([1.0, 0.8, 0.0])
    assert sources[0] == RetrievedSource(
        page=2,
        text="direct evidence",
        source_name="synthetic.pdf",
        score=1.0,
        retrieval_method="dense",
        diagnostics={"score_kind": "cosine_similarity"},
    )


def test_retrieval_rejects_query_embedding_with_wrong_dimension() -> None:
    knowledge_base = KnowledgeBase(
        name="synthetic.pdf",
        chunks=(TextChunk(page=1, text="evidence text here", source_name="synthetic.pdf"),),
        vectors=np.array([[1.0, 0.0]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="dimension does not match"):
        retrieve("question", knowledge_base, FakeEmbedder({"question": [1, 0, 0]}))

    with pytest.raises(ValueError, match="dimension does not match"):
        retrieve(
            "question",
            knowledge_base,
            None,
            query_vector=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )


def test_retrieval_accepts_a_valid_precomputed_query_vector() -> None:
    knowledge_base = KnowledgeBase(
        name="synthetic.pdf",
        chunks=(TextChunk(page=1, text="SYNTHETIC TEST FIXTURE evidence", source_name="synthetic.pdf"),),
        vectors=np.array([[1.0, 0.0]], dtype=np.float32),
    )

    found = retrieve(
        "SYNTHETIC TEST FIXTURE question",
        knowledge_base,
        None,
        query_vector=np.array([2.0, 0.0], dtype=np.float32),
    )

    assert len(found) == 1
    assert found[0].score == pytest.approx(1.0)


def test_missing_api_configuration_is_a_typed_error() -> None:
    with pytest.raises(MissingAPIConfigurationError, match="Set GOOGLE_API_KEY"):
        GoogleGenAIProvider.from_env(environ={})


def test_generation_error_is_not_returned_as_an_answer() -> None:
    class FailingGenerator:
        def generate(self, *args, **kwargs):
            raise RuntimeError("simulated provider failure")

    knowledge_base = KnowledgeBase(
        name="synthetic.pdf",
        chunks=(TextChunk(page=7, text="retrieved evidence passage", source_name="synthetic.pdf"),),
        vectors=np.array([[1.0, 0.0]], dtype=np.float32),
    )
    embedder = FakeEmbedder({"question": [1.0, 0.0]})

    result = answer(
        "question",
        knowledge_base,
        embedder,
        FailingGenerator(),
        RAGSettings(top_k=1),
    )

    assert result.status is AnswerStatus.GENERATION_ERROR
    assert result.answer is None
    assert result.error is not None
    assert result.error.code == "generation_failed"
    assert result.error.message == "Gemma generation request failed."
    assert "simulated provider failure" not in result.error.message
    assert result.sources[0].page == 7
    assert result.sources[0].text == "retrieved evidence passage"


def test_answer_keeps_only_a_response_with_a_retrieved_source_citation() -> None:
    class CitingGenerator:
        def generate(self, prompt, **kwargs):
            assert "[synthetic-rules.pdf, p. 7]" in prompt
            return "The passage states the rule. [synthetic-rules.pdf, p. 7]"

    knowledge_base = KnowledgeBase(
        name="synthetic-rules.pdf",
        chunks=(TextChunk(page=7, text="retrieved evidence passage", source_name="synthetic-rules.pdf"),),
        vectors=np.array([[1.0, 0.0]], dtype=np.float32),
    )
    result = answer(
        "question",
        knowledge_base,
        FakeEmbedder({"question": [1.0, 0.0]}),
        CitingGenerator(),
        RAGSettings(top_k=1),
    )

    assert result.status is AnswerStatus.ANSWERED
    assert result.answer == "The passage states the rule. [synthetic-rules.pdf, p. 7]"


def test_uncited_generation_is_rejected_as_an_answer() -> None:
    class UncitedGenerator:
        def generate(self, *args, **kwargs):
            return "The answer is probably yes."

    knowledge_base = KnowledgeBase(
        name="synthetic-rules.pdf",
        chunks=(TextChunk(page=7, text="retrieved evidence passage", source_name="synthetic-rules.pdf"),),
        vectors=np.array([[1.0, 0.0]], dtype=np.float32),
    )
    result = answer(
        "question",
        knowledge_base,
        FakeEmbedder({"question": [1.0, 0.0]}),
        UncitedGenerator(),
        RAGSettings(top_k=1),
    )

    assert result.status is AnswerStatus.INVALID_RESPONSE
    assert result.answer is None
    assert result.error is not None
    assert result.error.code == "citation_missing"


def test_answer_rejects_citations_that_do_not_match_retrieved_sources() -> None:
    class MixedCitingGenerator:
        def generate(self, *args, **kwargs):
            return "Supported claim [synthetic-rules.pdf, p. 7] [unretrieved.pdf, p. 2]"

    knowledge_base = KnowledgeBase(
        name="synthetic-rules.pdf",
        chunks=(TextChunk(page=7, text="retrieved evidence passage", source_name="synthetic-rules.pdf"),),
        vectors=np.array([[1.0, 0.0]], dtype=np.float32),
    )
    result = answer(
        "question",
        knowledge_base,
        FakeEmbedder({"question": [1.0, 0.0]}),
        MixedCitingGenerator(),
        RAGSettings(top_k=1),
    )

    assert result.status is AnswerStatus.INVALID_RESPONSE
    assert result.answer is None
    assert result.error is not None
    assert result.error.code == "citation_missing"


def test_relevance_threshold_abstains_without_calling_generator() -> None:
    class UnexpectedGenerator:
        def generate(self, *args, **kwargs):
            raise AssertionError("generation must not run without relevant evidence")

    knowledge_base = KnowledgeBase(
        name="synthetic.pdf",
        chunks=(TextChunk(page=1, text="weak evidence text", source_name="synthetic.pdf"),),
        vectors=np.array([[0.0, 1.0]], dtype=np.float32),
    )
    result = answer(
        "question",
        knowledge_base,
        FakeEmbedder({"question": [1.0, 0.0]}),
        UnexpectedGenerator(),
        RAGSettings(top_k=3, relevance_threshold=0.5),
    )

    assert result.status is AnswerStatus.ABSTAINED
    assert result.answer == "The answer could not be found in the document."
    assert result.sources == ()


def test_baseline_settings_are_selectable_and_environment_configurable() -> None:
    baseline = RAGSettings.baseline()
    assert (baseline.chunk_size, baseline.chunk_overlap, baseline.top_k) == (120, 30, 3)
    assert baseline.gemma_model == "gemma-4-26b-a4b-it"
    assert baseline.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert baseline.temperature == 0.2
    assert baseline.relevance_threshold is None

    custom = RAGSettings.from_env(
        {
            "GEMMA_MODEL": "custom-gemma",
            "EMBED_MODEL_NAME": "custom-embedder",
            "CHUNK_SIZE": "80",
            "CHUNK_OVERLAP": "12",
            "TOP_K": "5",
            "TEMPERATURE": "0.1",
            "RELEVANCE_THRESHOLD": "0.35",
        }
    )
    assert custom == RAGSettings(
        gemma_model="custom-gemma",
        embedding_model="custom-embedder",
        chunk_size=80,
        chunk_overlap=12,
        top_k=5,
        temperature=0.1,
        relevance_threshold=0.35,
    )
