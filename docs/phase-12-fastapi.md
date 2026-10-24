# Phase 12 — FastAPI integration

The API entry point is `backend/campusreg/main.py`; the route factory is
`campusreg.api.app:create_app`. Routes call `MultiDocumentRAGEngine`,
`ConversationRAGService`, and `ConflictDetector` from the shared package.
Pydantic schemas are in `campusreg/api/schemas.py`.

The API provides `/api/health`, `/api/ready`, PDF upload/indexing and document
management, question answering, exact evidence lookup, conversation lifecycle,
conflict reports, a bounded `/api/knowledge-graph` provenance view, and
development-only evaluation jobs. Interactive API
documentation is available at `/docs`.

The database defaults to `data/campusreg.sqlite3`; uploads and indexes do not
go under the source package. CORS accepts only origins configured through
`CAMPUSREG_CORS_ORIGINS` (default local development origins). Upload count,
file size, extension, MIME type, and PDF header are checked. Clients cannot
provide local paths. Model keys are read only by the backend provider. A
generation failure returns a structured non-answer with a failed HTTP status.

Evaluation jobs are disabled by default. To enable the development endpoints,
set `CAMPUSREG_ENABLE_DEV_EVALUATION=true` and a high-entropy
`CAMPUSREG_DEV_TOKEN` on the backend; the client must use a Bearer token.
Only a configured local dataset path is used, at most one job runs at a time,
and output files remain in ignored `evaluation/results/`.

Run locally from the repository root:

```powershell
python -m pip install -e ".\backend[all]"
$env:CAMPUSREG_CORS_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"
python -m uvicorn campusreg.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

The test suite injects fake embedding, generation, query-rewrite, and conflict
providers. Those mocks are not used by the normal application factory.
