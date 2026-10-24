# Current feature coverage

This matrix describes the checkout as it exists now. “Not implemented” means the UI or backend does not simulate a successful feature.

| Feature | Current implementation | Verification | Status and limits |
|---|---|---|---|
| Shared RAG, PDF ingestion, SQLite persistence and page citations | `backend/campusreg/rag/`, `backend/campusreg/ingestion/`, `backend/campusreg/storage/` | `python -m pytest backend/tests -q` | Implemented for text PDFs; official policy coverage depends on uploaded verified sources. |
| PRAG local preprocessing | `backend/campusreg/rag/privacy.py`, `backend/campusreg/ingestion/service.py`, `backend/campusreg/rag/retrieval.py` | Privacy unit tests in the backend suite | Partial: deterministic patterns cover listed formats only; this is not general PII recognition or a zero-leak guarantee. Legacy indexes are blocked pending re-index. |
| Rule-evasion preflight and prompt-injection instruction | `backend/campusreg/rag/guardrails.py`, `backend/campusreg/rag/pipeline.py`, `backend/campusreg/rag/conversation.py` | Guardrail and API tests in the backend suite | Partial: phrase-based screen, no reviewed safety benchmark; paraphrases can be missed. |
| Gemma model registry and application router | No router or OpenRouter adapter is present. The current generation adapter is Google GenAI with one configured model ID. | No routing tests | Not implemented. |
| Jev / TypeSafe classification | No adapter or API integration is present. | No Jev tests or benchmark | Not implemented. |
| Regulatory knowledge graph | `backend/campusreg/rag/knowledge_graph.py` and `/api/knowledge-graph` expose document → page → passage provenance only. | Existing API graph tests | Partial: no extracted regulatory entities, obligations, deadlines, exceptions or semantic relation edges. |
| Hybrid retrieval | Dense retrieval is the default; BM25/RRF code is configurable in the shared core. | Retrieval tests in the backend suite | Available as a retrieval option; no current official held-out result is claimed. |
| Feature quality, safety and cost evaluation | Existing evaluation infrastructure supports reviewed datasets. No feature benchmark was run in this change. | No measured feature metrics | Not tested; no accuracy, leakage, recall, routing-quality or cost claim is made. |
| Frontend | Existing Next.js workspace uses API-backed documents, chat, evidence, provenance graph and evaluation views; redaction summary is API-backed. | `npm test -- --run`, lint and build | The model-routing, Jev, semantic regulatory graph and redaction-processing controls are not represented as enabled features. |

## Actual software-check results

On 2026-10-24, the backend suite completed with 98 passed and one upstream Starlette deprecation warning. Frontend Vitest completed with 8 passed; ESLint and `next build` completed successfully. These are software checks, not policy-answer accuracy or privacy-leakage measurements. No official-policy evaluation set or real institutional PDF was used for this feature validation.
