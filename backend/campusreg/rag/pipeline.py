"""Framework-independent orchestration of the shared RAG stages."""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from campusreg.config import RAGSettings
from campusreg.rag.documents import read_pdf
from campusreg.rag.embeddings import EmbeddingModel
from campusreg.rag.errors import GenerationError, KnowledgeBaseNotLoadedError
from campusreg.rag.errors import IndexCompatibilityError
from campusreg.rag.evidence_gate import EvidenceGate
from campusreg.rag.generation import TextGenerator, generate
from campusreg.rag.models import (
    AnswerResult,
    AnswerStatus,
    DocumentPage,
    ErrorInfo,
    KnowledgeBase,
    KnowledgeBaseSummary,
    RetrievedSource,
)
from campusreg.rag.retrieval import build_knowledge_base, retrieve, retrieve_configured
from campusreg.storage.models import DocumentRecord, IngestionResult, IngestionStatus
from campusreg.storage.sqlite_store import SQLiteDocumentStore

ANSWER_RULES = """Answer questions using ONLY the context passages provided.
Every factual claim must cite its source using the exact document name, document
ID, and page shown in the context, formatted as [document name, document ID, p. N]. Never invent a source
or page. If the answer is not supported by the context, reply exactly:
"The answer could not be found in the document." This assistant is limited to
campus policy and compliance questions answered from the supplied passages. For
questions outside that scope, reply exactly: "This assistant only answers
campus compliance questions supported by the indexed documents." Do not add
general knowledge or legal advice. Keep the answer concise."""

OUT_OF_SCOPE_ANSWER = "This assistant only answers campus compliance questions supported by the indexed documents."


def answer(
    question: str,
    knowledge_base: KnowledgeBase,
    embedder: EmbeddingModel,
    generator: TextGenerator,
    settings: RAGSettings | None = None,
    *,
    evidence_gate: EvidenceGate | None = None,
    retrieval_query: str | None = None,
) -> AnswerResult:
    """Retrieve evidence and return an answer, abstention, or typed failure."""

    selected = settings or RAGSettings.baseline()
    search_query = retrieval_query or question
    found = tuple(retrieve_configured(search_query, knowledge_base, embedder, selected))
    gate_decision = evidence_gate.evaluate(found, selected) if evidence_gate is not None else None
    if gate_decision is not None and not gate_decision.accepted:
        return AnswerResult(
            status=AnswerStatus.ABSTAINED,
            answer=gate_decision.missing_information,
            sources=found,
            gate_decision=gate_decision.to_dict(),
            retrieval_query=search_query,
        )
    if not found:
        return AnswerResult(
            status=AnswerStatus.ABSTAINED,
            answer="The answer could not be found in the document.",
            sources=(),
            gate_decision=gate_decision.to_dict() if gate_decision is not None else None,
            retrieval_query=search_query,
        )

    context = "\n\n".join(
        f"{source.citation} {source.text}" for source in found
    )
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    try:
        generated = generate(prompt, generator, selected, system_text=ANSWER_RULES)
    except GenerationError as exc:
        return AnswerResult(
            status=AnswerStatus.GENERATION_ERROR,
            answer=None,
            sources=found,
            error=ErrorInfo(code=exc.code, message=str(exc)),
            gate_decision=gate_decision.to_dict() if gate_decision is not None else None,
            retrieval_query=search_query,
        )

    if generated.strip() in {"The answer could not be found in the document.", OUT_OF_SCOPE_ANSWER}:
        return AnswerResult(
            status=AnswerStatus.ABSTAINED,
            answer=generated.strip(),
            sources=found,
            gate_decision=gate_decision.to_dict() if gate_decision is not None else None,
            retrieval_query=search_query,
        )

    cited_labels = re.findall(r"\[[^\]\n]+\]", generated)
    allowed_labels = {source.citation for source in found}
    if not cited_labels or any(label not in allowed_labels for label in cited_labels):
        return AnswerResult(
            status=AnswerStatus.INVALID_RESPONSE,
            answer=None,
            sources=found,
            error=ErrorInfo(
                code="citation_missing",
                message="Gemma response did not cite a retrieved document and page.",
            ),
            gate_decision=gate_decision.to_dict() if gate_decision is not None else None,
            retrieval_query=search_query,
        )

    return AnswerResult(
        status=AnswerStatus.ANSWERED,
        answer=generated,
        sources=found,
        gate_decision=gate_decision.to_dict() if gate_decision is not None else None,
        retrieval_query=search_query,
    )


class RAGEngine:
    """Stateful single-document engine around the shared RAG functions."""

    def __init__(
        self,
        settings: RAGSettings,
        embedder: EmbeddingModel,
        generator: TextGenerator,
        *,
        evidence_gate: EvidenceGate | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.generator = generator
        self.evidence_gate = evidence_gate
        self.knowledge_base: KnowledgeBase | None = None
        self._page_count = 0

    def build_knowledge_base(
        self,
        pages: Sequence[DocumentPage | str],
        name: str,
    ) -> KnowledgeBaseSummary:
        knowledge_base = build_knowledge_base(pages, name, self.embedder, self.settings)
        self.knowledge_base = knowledge_base
        self._page_count = len(pages)
        return KnowledgeBaseSummary(
            name=knowledge_base.name,
            page_count=self._page_count,
            chunk_count=len(knowledge_base.chunks),
        )

    def load_pdf(self, path: str | Path) -> KnowledgeBaseSummary:
        pages = read_pdf(path)
        return self.build_knowledge_base(pages, Path(path).name)

    def retrieve(self, question: str) -> list[RetrievedSource]:
        knowledge_base = self._require_knowledge_base()
        return retrieve_configured(question, knowledge_base, self.embedder, self.settings)

    def answer(self, question: str, *, retrieval_query: str | None = None) -> AnswerResult:
        knowledge_base = self._require_knowledge_base()
        return answer(
            question,
            knowledge_base,
            self.embedder,
            self.generator,
            self.settings,
            evidence_gate=self.evidence_gate,
            retrieval_query=retrieval_query,
        )

    def _require_knowledge_base(self) -> KnowledgeBase:
        if self.knowledge_base is None:
            raise KnowledgeBaseNotLoadedError("Build a knowledge base before asking questions.")
        return self.knowledge_base


class MultiDocumentRAGEngine:
    """Persistent RAG engine that retrieves across all indexed PDF sources."""

    def __init__(
        self,
        settings: RAGSettings,
        embedder: EmbeddingModel,
        generator: TextGenerator,
        store: SQLiteDocumentStore,
        *,
        evidence_gate: EvidenceGate | None = None,
    ) -> None:
        from campusreg.ingestion.service import PDFIngestor

        self.settings = settings
        self.embedder = embedder
        self.generator = generator
        self.store = store
        self.evidence_gate = evidence_gate
        self.ingestor = PDFIngestor(store, embedder, settings)

    def ingest_pdf(
        self,
        content: bytes,
        filename: str,
        *,
        source_url: str | None = None,
        effective_date: date | str | None = None,
        document_type: str = "unknown",
    ) -> IngestionResult:
        return self.ingestor.ingest_pdf(
            content,
            filename,
            source_url=source_url,
            effective_date=effective_date,
            document_type=document_type,
        )

    def reindex_document(self, document_id: str) -> IngestionResult:
        return self.ingestor.reindex_document(document_id)

    def delete_document(self, document_id: str) -> bool:
        return self.store.delete_document(document_id)

    def list_documents(self) -> list[DocumentRecord]:
        return self.store.list_documents()

    def retrieve(self, question: str) -> list[RetrievedSource]:
        return retrieve_configured(question, self._load_knowledge_base(), self.embedder, self.settings)

    def answer(self, question: str, *, retrieval_query: str | None = None) -> AnswerResult:
        return answer(
            question,
            self._load_knowledge_base(),
            self.embedder,
            self.generator,
            self.settings,
            evidence_gate=self.evidence_gate,
            retrieval_query=retrieval_query,
        )

    def _load_knowledge_base(self) -> KnowledgeBase:
        indexed = [
            record
            for record in self.store.list_documents()
            if record.ingestion_status in (IngestionStatus.INDEXED, IngestionStatus.REINDEXING)
            and record.chunk_count > 0
        ]
        if any(record.embedding_model != self.settings.embedding_model for record in indexed):
            raise IndexCompatibilityError(
                "Stored documents use a different embedding model; re-index them before retrieval."
            )
        return self.store.load_knowledge_base()
