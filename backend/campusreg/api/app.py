"""HTTP boundary for CampusReg; business logic remains in campusreg.rag."""

from __future__ import annotations

import hmac
import os
from dataclasses import dataclass
from datetime import date
from importlib.util import find_spec
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, Form, Header, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from campusreg.config import RAGSettings
from campusreg.evaluation.dataset import load_dataset
from campusreg.evaluation.runner import EvaluationRunner
from campusreg.rag.conflicts import ConflictDetector, GemmaConflictComparator
from campusreg.rag.conversation import ConversationRAGService, GemmaQueryRewriter
from campusreg.rag.embeddings import SentenceTransformerEmbedder
from campusreg.rag.errors import (
    DocumentIngestionError,
    GenerationError,
    KnowledgeBaseNotLoadedError,
    MissingAPIConfigurationError,
    PrivacyReindexRequiredError,
)
from campusreg.rag.generation import GoogleGenAIProvider, TextGenerator
from campusreg.rag.knowledge_graph import build_source_knowledge_graph
from campusreg.rag.models import AnswerResult, AnswerStatus, RetrievedSource
from campusreg.rag.pipeline import MultiDocumentRAGEngine
from campusreg.rag.scope import is_campus_education_question
from campusreg.rag.guardrails import preflight_request
from campusreg.storage.models import DocumentRecord
from campusreg.storage.sqlite_store import SQLiteDocumentStore
from campusreg.api.schemas import (
    ConversationCreateResponse,
    ConversationHistoryResponse,
    ConversationMessageSchema,
    DocumentSchema,
    ErrorSchema,
    EvaluationJobRequest,
    EvaluationJobSchema,
    QuestionRequest,
    QuestionResponse,
    SourceSchema,
    UploadBatchResponse,
    UploadItemSchema,
    KnowledgeGraphSchema,
)


@dataclass(slots=True)
class APIServices:
    """Injected application services; tests can replace all model providers."""

    engine: MultiDocumentRAGEngine
    conversations: ConversationRAGService
    conflicts: ConflictDetector
    requires_api_key: bool = False


class _LazyGoogleGenerator:
    """Delay SDK/key setup until generation is requested (uploads need no key)."""

    def __init__(self) -> None:
        self._provider: GoogleGenAIProvider | None = None

    def generate(self, prompt: str, **kwargs: Any) -> str:
        if self._provider is None:
            self._provider = GoogleGenAIProvider.from_env()
        return self._provider.generate(prompt, **kwargs)


class _LazySentenceTransformerEmbedder:
    """Avoid downloading/initializing an embedding model for health or empty UI loads."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._delegate: SentenceTransformerEmbedder | None = None

    def encode(self, texts: list[str]) -> Any:
        if self._delegate is None:
            self._delegate = SentenceTransformerEmbedder(self.model_name)
        return self._delegate.encode(texts)


def _api_key_is_configured() -> bool:
    return any(
        os.environ.get(name, "").strip()
        for name in ("GOOGLE_API_KEY", "GEMMA_API_KEY", "gemma_API_Key")
    )


def _source_schema(source: RetrievedSource) -> SourceSchema:
    return SourceSchema(
        document_id=source.document_id,
        document_name=source.source_name,
        page=source.page,
        chunk_id=source.chunk_id,
        citation=source.citation,
        text=source.text,
        retrieval_method=source.retrieval_method,
        score=source.score,
        diagnostics=dict(source.diagnostics),
    )


def _document_schema(record: DocumentRecord) -> DocumentSchema:
    return DocumentSchema(
        id=record.id, filename=record.filename, content_hash=record.content_hash,
        source_url=record.source_url, effective_date=record.effective_date,
        document_type=record.document_type, upload_status=record.upload_status.value,
        ingestion_status=record.ingestion_status.value, size_bytes=record.size_bytes,
        page_count=record.page_count, chunk_count=record.chunk_count,
        embedding_model=record.embedding_model, embedding_dimension=record.embedding_dimension,
        last_error_code=record.last_error_code, created_at=record.created_at, updated_at=record.updated_at,
        redaction_summary=record.redaction_summary,
        redaction_version=record.redaction_version,
    )


def _question_response(result: AnswerResult, *, conversation_id: str | None = None) -> QuestionResponse:
    response = QuestionResponse(
        status=result.status.value,
        answer=result.answer,
        citations=[source.citation for source in result.sources] if result.status is AnswerStatus.ANSWERED else [],
        supporting_passages=[_source_schema(source) for source in result.sources],
        abstained=result.status is AnswerStatus.ABSTAINED,
        conflicts=[],
        conflict_check_status="available_on_conflict_endpoint",
        conversation_id=conversation_id,
        retrieval_query=result.retrieval_query,
        error=ErrorSchema(code=result.error.code, message=result.error.message) if result.error else None,
    )
    if result.status in {AnswerStatus.GENERATION_ERROR, AnswerStatus.INVALID_RESPONSE}:
        raise _QuestionError(response)
    return response


class _QuestionError(Exception):
    def __init__(self, response: QuestionResponse) -> None:
        self.response = response


def create_app(
    services: APIServices | None = None,
    *,
    database_path: str | Path | None = None,
    evaluation_dataset_path: str | Path | None = None,
    evaluation_output_dir: str | Path | None = None,
    dev_evaluation_enabled: bool | None = None,
) -> FastAPI:
    """Create the API, optionally with injected deterministic test services."""
    application = FastAPI(
        title="CampusReg API",
        version="0.1.0",
        description="Evidence-first campus regulations API. Policy answers require user-provided official documents.",
    )
    application.state.services = services
    application.state.database_path = str(database_path or os.environ.get("CAMPUSREG_DB_PATH", "data/campusreg.sqlite3"))
    application.state.evaluation_dataset_path = Path(
        evaluation_dataset_path or os.environ.get("CAMPUSREG_EVALUATION_DATASET", "evaluation/questions.jsonl")
    )
    application.state.evaluation_output_dir = Path(
        evaluation_output_dir or os.environ.get("CAMPUSREG_EVALUATION_OUTPUT_DIR", "evaluation/results/api")
    )
    application.state.dev_evaluation_enabled = (
        dev_evaluation_enabled if dev_evaluation_enabled is not None
        else os.environ.get("CAMPUSREG_ENABLE_DEV_EVALUATION", "false").casefold() == "true"
    )
    application.state.evaluation_jobs: dict[str, dict[str, Any]] = {}
    allowed_origins = [
        origin.strip() for origin in os.environ.get(
            "CAMPUSREG_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
        ).split(",") if origin.strip()
    ]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    def get_services() -> APIServices:
        current = application.state.services
        if current is not None:
            return current
        settings = RAGSettings.from_env()
        store = SQLiteDocumentStore(application.state.database_path)
        embedder = _LazySentenceTransformerEmbedder(settings.embedding_model)
        generator: TextGenerator = _LazyGoogleGenerator()
        engine = MultiDocumentRAGEngine(settings, embedder, generator, store)
        conversations = ConversationRAGService(
            store, engine, GemmaQueryRewriter(generator, settings)
        )
        detector = ConflictDetector(GemmaConflictComparator(generator, settings))
        current = APIServices(engine, conversations, detector, requires_api_key=True)
        application.state.services = current
        return current

    def require_model_key(selected: APIServices) -> None:
        if selected.requires_api_key and not _api_key_is_configured():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"code": "missing_api_configuration", "message": "Set GOOGLE_API_KEY on the backend before asking model-assisted questions."},
            )

    def service_from_scope() -> APIServices:
        try:
            return get_services()
        except RuntimeError:
            raise HTTPException(503, detail={"code": "service_unavailable", "message": "The embedding service is unavailable. Install the API and embedding dependencies."}) from None

    @application.get("/api/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/api/ready", tags=["system"])
    def readiness() -> dict[str, Any]:
        configured = _api_key_is_configured()
        return {
            "status": "ready" if configured else "degraded",
            "model_api_key_configured": configured,
            "embedding_dependency_available": find_spec("sentence_transformers") is not None,
            "database_configured": bool(application.state.database_path),
        }

    @application.post("/api/documents", response_model=UploadBatchResponse, status_code=201, tags=["documents"])
    async def upload_documents(
        files: list[UploadFile] = File(..., description="One or more PDF files"),
        source_url: str | None = Form(default=None),
        effective_date: date | None = Form(default=None),
        document_type: str = Form(default="unknown", max_length=80),
    ) -> UploadBatchResponse:
        if not files:
            raise HTTPException(422, detail={"code": "missing_files", "message": "Upload at least one PDF."})
        if len(files) > 10:
            raise HTTPException(413, detail={"code": "too_many_files", "message": "Upload no more than 10 PDFs at a time."})
        selected = service_from_scope()
        items: list[UploadItemSchema] = []
        for upload in files:
            filename = upload.filename or ""
            try:
                if upload.content_type not in {"application/pdf", "application/octet-stream"}:
                    raise HTTPException(415, detail={"code": "invalid_mime_type", "message": "Only PDF uploads are supported."})
                limit = selected.engine.settings.max_pdf_size_bytes
                content = await upload.read(limit + 1)
                if len(content) > limit:
                    raise HTTPException(413, detail={"code": "file_too_large", "message": f"PDF exceeds the configured {limit}-byte upload limit."})
                result = selected.engine.ingest_pdf(
                    content, filename, source_url=source_url, effective_date=effective_date, document_type=document_type,
                )
                items.append(UploadItemSchema(
                    filename=filename, status="indexed", duplicate=result.duplicate,
                    document=_document_schema(result.document),
                ))
            except HTTPException as exc:
                detail = exc.detail if isinstance(exc.detail, dict) else {"code": "upload_rejected", "message": str(exc.detail)}
                items.append(UploadItemSchema(filename=filename, status="failed", error=ErrorSchema(**detail)))
            except DocumentIngestionError as exc:
                items.append(UploadItemSchema(
                    filename=filename, status="failed", error=ErrorSchema(code=exc.code, message=str(exc)),
                ))
        return UploadBatchResponse(items=items)

    @application.get("/api/documents", response_model=list[DocumentSchema], tags=["documents"])
    def list_documents() -> list[DocumentSchema]:
        return [_document_schema(record) for record in service_from_scope().engine.list_documents()]

    @application.delete("/api/documents/{document_id}", tags=["documents"])
    def delete_document(document_id: str) -> dict[str, Any]:
        if not service_from_scope().engine.delete_document(document_id):
            raise HTTPException(404, detail={"code": "document_not_found", "message": "No indexed document has that ID."})
        return {"deleted": True, "document_id": document_id}

    @application.post("/api/documents/{document_id}/reindex", response_model=DocumentSchema, tags=["documents"])
    def reindex_document(document_id: str) -> DocumentSchema:
        try:
            result = service_from_scope().engine.reindex_document(document_id)
        except KeyError:
            raise HTTPException(404, detail={"code": "document_not_found", "message": "No document has that ID."}) from None
        except DocumentIngestionError as exc:
            raise HTTPException(422, detail={"code": exc.code, "message": str(exc)}) from None
        return _document_schema(result.document)

    @application.get("/api/documents/{document_id}/evidence/{chunk_id}", tags=["evidence"])
    def read_evidence(document_id: str, chunk_id: str) -> dict[str, Any]:
        engine = service_from_scope().engine
        if not any(record.id == document_id for record in engine.list_documents()):
            raise HTTPException(404, detail={"code": "document_not_found", "message": "No document has that ID."})
        try:
            kb = engine.store.load_knowledge_base()
        except PrivacyReindexRequiredError as exc:
            raise HTTPException(409, detail={"code": "privacy_reindex_required", "message": str(exc)}) from None
        except KnowledgeBaseNotLoadedError:
            raise HTTPException(404, detail={"code": "evidence_not_found", "message": "No indexed evidence is available."}) from None
        for chunk in kb.chunks:
            if chunk.document_id == document_id and chunk.chunk_id == chunk_id:
                return {"document_id": document_id, "document_name": chunk.source_name, "page": chunk.page, "chunk_id": chunk.chunk_id, "text": chunk.text}
        raise HTTPException(404, detail={"code": "evidence_not_found", "message": "That evidence passage is not indexed."})

    @application.get("/api/knowledge-graph", response_model=KnowledgeGraphSchema, tags=["evidence"])
    def knowledge_graph() -> KnowledgeGraphSchema:
        """Return bounded, source-linked document/page/passage relationships."""

        engine = service_from_scope().engine
        if any(
            record.redaction_version < 1 and record.ingestion_status.value in {"indexed", "reindexing"}
            for record in engine.list_documents()
        ):
            raise HTTPException(409, detail={"code": "privacy_reindex_required", "message": "Re-index older sources before exposing their passage graph."})
        chunks = engine.store.list_source_graph_chunks(limit=501)
        graph = build_source_knowledge_graph(chunks, max_passages=500)
        return KnowledgeGraphSchema(**graph)

    @application.post("/api/questions", response_model=QuestionResponse, tags=["questions"])
    def ask_question(request: QuestionRequest) -> Any:
        selected = service_from_scope()
        decision = preflight_request(request.question)
        if is_campus_education_question(request.question) and not decision.blocked:
            require_model_key(selected)
        try:
            result = selected.engine.answer(request.question)
            response = _question_response(result)
        except _QuestionError as exc:
            return JSONResponse(status_code=503 if exc.response.status == "generation_error" else 502, content=exc.response.model_dump())
        except PrivacyReindexRequiredError as exc:
            raise HTTPException(409, detail={"code": "privacy_reindex_required", "message": str(exc)}) from None
        except KnowledgeBaseNotLoadedError:
            raise HTTPException(409, detail={"code": "no_indexed_documents", "message": "Upload an official, text-based PDF before asking questions."}) from None
        return response

    @application.get("/api/conflicts", tags=["conflicts"])
    def get_conflicts() -> dict[str, Any]:
        selected = service_from_scope()
        require_model_key(selected)
        try:
            kb = selected.engine.store.load_knowledge_base()
        except PrivacyReindexRequiredError as exc:
            raise HTTPException(409, detail={"code": "privacy_reindex_required", "message": str(exc)}) from None
        except KnowledgeBaseNotLoadedError:
            raise HTTPException(409, detail={"code": "no_indexed_documents", "message": "Index documents before checking cross-document conflicts."}) from None
        sources = [RetrievedSource(
            page=chunk.page, text=chunk.text, source_name=chunk.source_name, score=0.0,
            document_id=chunk.document_id, chunk_id=chunk.chunk_id,
        ) for chunk in kb.chunks]
        records = {item.id: item for item in selected.engine.list_documents()}
        reports = selected.conflicts.detect(sources, documents=records)
        return {
            "reports": [report.to_dict() for report in reports],
            "human_review_required": True,
            "authority_decided": False,
        }

    @application.post("/api/conversations", response_model=ConversationCreateResponse, tags=["conversations"])
    def create_conversation() -> ConversationCreateResponse:
        return ConversationCreateResponse(conversation_id=service_from_scope().conversations.create_conversation())

    @application.get("/api/conversations/{conversation_id}", response_model=ConversationHistoryResponse, tags=["conversations"])
    def conversation_history(conversation_id: str) -> ConversationHistoryResponse:
        try:
            messages = service_from_scope().conversations.store.list_conversation_messages(conversation_id)
        except KeyError:
            raise HTTPException(404, detail={"code": "conversation_not_found", "message": "No conversation has that ID."}) from None
        return ConversationHistoryResponse(
            conversation_id=conversation_id,
            messages=[ConversationMessageSchema(
                id=message.id, conversation_id=message.conversation_id, role=message.role,
                content=message.content, created_at=message.created_at,
                original_question=message.original_question, retrieval_query=message.retrieval_query,
                answer_status=message.answer_status, sources=list(message.sources),
            ) for message in messages],
        )

    @application.post("/api/conversations/{conversation_id}/questions", response_model=QuestionResponse, tags=["conversations"])
    def ask_conversation(conversation_id: str, request: QuestionRequest) -> Any:
        selected = service_from_scope()
        try:
            if (
                selected.conversations.is_in_scope(conversation_id, request.question)
                and not preflight_request(request.question).blocked
            ):
                require_model_key(selected)
            turn = selected.conversations.ask(conversation_id, request.question)
        except KeyError:
            raise HTTPException(404, detail={"code": "conversation_not_found", "message": "Create the conversation before using it."}) from None
        try:
            return _question_response(turn.result, conversation_id=conversation_id)
        except _QuestionError as exc:
            return JSONResponse(status_code=503 if exc.response.status == "generation_error" else 502, content=exc.response.model_dump())

    @application.delete("/api/conversations/{conversation_id}/messages", tags=["conversations"])
    def clear_conversation(conversation_id: str) -> dict[str, Any]:
        if not service_from_scope().conversations.clear_conversation(conversation_id):
            raise HTTPException(404, detail={"code": "conversation_not_found", "message": "No conversation has that ID."})
        return {"cleared": True, "conversation_id": conversation_id}

    @application.delete("/api/conversations/{conversation_id}", tags=["conversations"])
    def delete_conversation(conversation_id: str) -> dict[str, Any]:
        if not service_from_scope().conversations.delete_conversation(conversation_id):
            raise HTTPException(404, detail={"code": "conversation_not_found", "message": "No conversation has that ID."})
        return {"deleted": True, "conversation_id": conversation_id}

    def authorize_evaluation(authorization: str | None = Header(default=None)) -> None:
        if not application.state.dev_evaluation_enabled:
            raise HTTPException(404, detail={"code": "evaluation_unavailable", "message": "Development evaluation endpoints are disabled."})
        expected = os.environ.get("CAMPUSREG_DEV_TOKEN", "")
        supplied = authorization.removeprefix("Bearer ") if authorization and authorization.startswith("Bearer ") else ""
        if not expected or not hmac.compare_digest(supplied, expected):
            raise HTTPException(401, detail={"code": "evaluation_unauthorized", "message": "A valid development bearer token is required."})

    def run_evaluation_job(job_id: str, split: str, force: bool) -> None:
        job = application.state.evaluation_jobs[job_id]
        job["status"] = "running"
        try:
            questions = load_dataset(application.state.evaluation_dataset_path, split=split)
            if not questions:
                raise ValueError(f"The configured reviewed dataset has no {split} questions.")
            selected = get_services()
            if selected.requires_api_key and not _api_key_is_configured():
                raise MissingAPIConfigurationError("Set the backend model key before running model evaluation.")
            output = application.state.evaluation_output_dir / job_id
            run = EvaluationRunner().run(
                questions, answerer=selected.engine.answer, output_dir=output, split=split,
                retrieval_k=selected.engine.settings.top_k,
                run_config={"gemma_model": selected.engine.settings.gemma_model,
                            "embedding_model": selected.engine.settings.embedding_model,
                            "chunk_size": selected.engine.settings.chunk_size,
                            "chunk_overlap": selected.engine.settings.chunk_overlap,
                            "top_k": selected.engine.settings.top_k,
                            "retrieval_method": selected.engine.settings.retrieval_method},
                force=force,
            )
            job.update({"status": "completed", "result": {
                "metrics": run.metrics, "question_count": len(run.predictions),
                "predictions_file": run.predictions_path.name,
                "metrics_json_file": run.metrics_json_path.name,
                "metrics_csv_file": run.metrics_csv_path.name,
            }})
        except Exception as exc:
            # Keep outputs safe and useful; exceptions from provider libraries can include request details.
            job.update({"status": "failed", "error": {"code": getattr(exc, "code", "evaluation_failed"), "message": str(exc)[:300]}})

    @application.post("/api/development/evaluations", response_model=EvaluationJobSchema, status_code=202, tags=["development"])
    def start_evaluation(
        request: EvaluationJobRequest,
        background_tasks: BackgroundTasks,
        authorization: str | None = Header(default=None),
    ) -> EvaluationJobSchema:
        authorize_evaluation(authorization)
        if not application.state.evaluation_dataset_path.exists():
            raise HTTPException(503, detail={"code": "evaluation_dataset_missing", "message": "Configure a reviewed evaluation dataset on the backend first."})
        if any(job["status"] in {"queued", "running"} for job in application.state.evaluation_jobs.values()):
            raise HTTPException(429, detail={"code": "evaluation_job_running", "message": "Only one development evaluation job may run at a time."})
        job_id = str(uuid4())
        job = {"id": job_id, "status": "queued", "split": request.split, "result": None, "error": None}
        application.state.evaluation_jobs[job_id] = job
        background_tasks.add_task(run_evaluation_job, job_id, request.split, request.force)
        return EvaluationJobSchema(**job)

    @application.get("/api/development/evaluations", response_model=list[EvaluationJobSchema], tags=["development"])
    def list_evaluation_jobs(authorization: str | None = Header(default=None)) -> list[EvaluationJobSchema]:
        authorize_evaluation(authorization)
        return [EvaluationJobSchema(**job) for job in application.state.evaluation_jobs.values()]

    @application.get("/api/development/evaluations/{job_id}", response_model=EvaluationJobSchema, tags=["development"])
    def get_evaluation_job(job_id: str, authorization: str | None = Header(default=None)) -> EvaluationJobSchema:
        authorize_evaluation(authorization)
        job = application.state.evaluation_jobs.get(job_id)
        if job is None:
            raise HTTPException(404, detail={"code": "evaluation_job_not_found", "message": "No evaluation job has that ID."})
        return EvaluationJobSchema(**job)

    return application
