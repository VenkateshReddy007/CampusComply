"""Typed errors for RAG operations and provider configuration."""


class RAGError(Exception):
    """Base class for expected RAG errors."""


class EmptyDocumentError(RAGError):
    """Raised when a document has no chunk long enough to index."""


class KnowledgeBaseNotLoadedError(RAGError):
    """Raised when retrieval is requested before building a knowledge base."""


class MissingAPIConfigurationError(RAGError):
    """Raised when no supported Google API-key environment variable is set."""


class GenerationError(RAGError):
    """Sanitized provider failure; never carries a policy answer."""

    def __init__(self, message: str = "Gemma generation failed.", code: str = "generation_failed") -> None:
        super().__init__(message)
        self.code = code


class DocumentIngestionError(RAGError):
    """Sanitized, structured rejection or ingestion failure for one PDF."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


class InvalidFileTypeError(DocumentIngestionError):
    def __init__(self) -> None:
        super().__init__("Only PDF files are accepted.", "invalid_file_type")


class FileTooLargeError(DocumentIngestionError):
    def __init__(self, limit_bytes: int) -> None:
        super().__init__(f"PDF exceeds the configured {limit_bytes}-byte upload limit.", "file_too_large")


class PDFParseError(DocumentIngestionError):
    def __init__(self) -> None:
        super().__init__("The PDF could not be parsed.", "pdf_parse_error")


class UnsupportedScannedPDFError(DocumentIngestionError):
    def __init__(self) -> None:
        super().__init__("This PDF has no extractable text; scanned PDFs require a separate OCR feature.", "ocr_unsupported")


class IndexCompatibilityError(RAGError):
    """Stored vectors were built with incompatible embedding settings."""
