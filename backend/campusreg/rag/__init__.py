"""Shared, framework-independent RAG core."""

from campusreg.rag.chunking import split_into_chunks
from campusreg.rag.documents import read_pdf
from campusreg.rag.embeddings import EmbeddingModel, SentenceTransformerEmbedder
from campusreg.rag.errors import (
    DocumentIngestionError,
    EmptyDocumentError,
    FileTooLargeError,
    GenerationError,
    IndexCompatibilityError,
    InvalidFileTypeError,
    KnowledgeBaseNotLoadedError,
    MissingAPIConfigurationError,
    PDFParseError,
    UnsupportedScannedPDFError,
)
from campusreg.rag.generation import GoogleGenAIProvider, TextGenerator, generate
from campusreg.rag.evidence_gate import EvidenceGate, EvidenceGateDecision, EvidenceGatePolicy
from campusreg.rag.conversation import (
    ConversationRAGService,
    ConversationTurnResult,
    GemmaQueryRewriter,
    requires_query_rewrite,
)
from campusreg.rag.conflicts import (
    ConflictAssessment,
    ConflictDetector,
    ConflictPassage,
    ConflictReport,
    ConflictStatus,
    GemmaConflictComparator,
    conflict_metrics,
)
from campusreg.rag.models import (
    AnswerResult,
    AnswerStatus,
    DocumentPage,
    ErrorInfo,
    KnowledgeBase,
    KnowledgeBaseSummary,
    RetrievedSource,
    TextChunk,
)
from campusreg.rag.pipeline import MultiDocumentRAGEngine, RAGEngine, answer
from campusreg.rag.retrieval import build_knowledge_base, retrieve, retrieve_bm25, retrieve_configured, retrieve_hybrid

__all__ = [
    "AnswerResult",
    "AnswerStatus",
    "ConversationRAGService",
    "ConversationTurnResult",
    "ConflictAssessment",
    "ConflictDetector",
    "ConflictPassage",
    "ConflictReport",
    "ConflictStatus",
    "DocumentPage",
    "EvidenceGate",
    "EvidenceGateDecision",
    "EvidenceGatePolicy",
    "DocumentIngestionError",
    "EmbeddingModel",
    "EmptyDocumentError",
    "ErrorInfo",
    "FileTooLargeError",
    "GenerationError",
    "GemmaQueryRewriter",
    "GemmaConflictComparator",
    "GoogleGenAIProvider",
    "KnowledgeBase",
    "KnowledgeBaseSummary",
    "KnowledgeBaseNotLoadedError",
    "IndexCompatibilityError",
    "InvalidFileTypeError",
    "MissingAPIConfigurationError",
    "MultiDocumentRAGEngine",
    "PDFParseError",
    "RAGEngine",
    "RetrievedSource",
    "SentenceTransformerEmbedder",
    "TextChunk",
    "TextGenerator",
    "UnsupportedScannedPDFError",
    "answer",
    "build_knowledge_base",
    "conflict_metrics",
    "generate",
    "read_pdf",
    "requires_query_rewrite",
    "retrieve",
    "retrieve_bm25",
    "retrieve_configured",
    "retrieve_hybrid",
    "split_into_chunks",
]
