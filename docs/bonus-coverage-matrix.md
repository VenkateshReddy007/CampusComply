# CampusReg delivery and bonus coverage

Status definitions: **complete** means the code path and its software checks
are present; **partial** means implementation exists but the required real
data or quality measurement is absent; **not tested** means the required
validation has not run. Software tests do not establish policy correctness.

| Requirement | Implementation | Test command | Actual result | Supporting record | Limitation | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Shared PDF RAG core and cited answers | `backend/campusreg/rag/` | `python -m pytest tests -q` from `backend/` | 81 passed, 1 warning (8.38s) | Backend software suite; see evaluation report | No official campus PDFs in checkout | Partial |
| Multiple PDFs, persistent metadata, original page citations | `ingestion/`, `storage/`, `rag/pipeline.py` | `python -m pytest tests -q` from `backend/` | Included in suite: 81 passed, 1 warning | `docs/phase-3-multi-document.md`; synthetic API ingestion smoke passed for 2 PDFs | Software fixtures only | Partial |
| Held-out evaluation with at least 15 human-reviewed questions | `backend/campusreg/evaluation/`, `evaluation/` | `python -m campusreg.evaluation.dataset` validation command | Not run: no official reviewed dataset | `evaluation/README.md` | Expected answers/labels require official sources and reviewers | Not tested |
| Chunk/top-k/embedding experiments | `retrieval_experiments.py` | `campusreg-retrieval-experiments ...` | Not run | `evaluation/README.md` | Requires fixed corpus and reviewed development/validation set | Not tested |
| Paired RAG versus no-RAG | `paired.py` | `campusreg-paired-evaluation --smoke` / `--full` | Full run not run; see test logs for smoke | `evaluation/README.md` | Full run requires API credentials and reviewed questions | Partial |
| Relevance threshold and refusal | `rag/evidence_gate.py` | `python -m pytest tests/test_evidence_gate.py -q` | See evaluation report | `evaluation/README.md` | No calibrated production policy | Partial |
| Contextual follow-up questions | `rag/conversation.py`, SQLite | `python -m pytest tests/test_conversation.py -q` | See evaluation report | `docs/phase-8-conversations.md` | Synthetic/mock evaluation only | Partial |
| Gemma 26B A4B versus 31B | `model_comparison.py` | `campusreg-model-comparison --smoke` / `--full` | Full model comparison not run | `docs/phase-9-model-comparison.md` | No API calls or human review performed | Not tested |
| Dense versus hybrid retrieval | `rag/retrieval.py`, `hybrid_comparison.py` | `python -m pytest tests/test_hybrid_retrieval.py -q` | See evaluation report | `docs/phase-10-hybrid-retrieval.md` | No held-out campus retrieval results | Partial |
| Cross-document conflict detection with two passages | `rag/conflicts.py`, `/api/conflicts` | `python -m pytest tests -q` from `backend/` | Synthetic contract/API coverage included in 81 passing tests | `docs/phase-11-conflict-detection.md` | Lexical candidates; no real labelled set or authority policy | Partial |
| Typed FastAPI backend | `backend/campusreg/api/` | `python -m pytest tests -q` from `backend/` | 81 passed, 1 warning; local health and readiness returned HTTP 200 | `docs/phase-12-fastapi.md` | No production auth, tenancy, or deployment | Partial |
| Responsive Next.js interface | `frontend/` | `npm test`, `npm run lint`, `npm run typecheck`, `npm run build` | 5 tests passed; lint, type-check, and build passed | Frontend source and tests | Requires running API; generation needs operator key | Partial |
| Final end-to-end run with official PDFs and manual citation review | Not available in this checkout | Manual showcase procedure | Blocked: no official corpus, evaluation labels, or API key | `docs/demo-script.md` | Must be completed by project owners | Not tested |

No benchmark result in this matrix is inferred from code presence or mock
fixtures. Software command output is recorded in `docs/evaluation-report.md`;
policy-quality measurements remain unavailable without verified official
documents and reviewed question labels.
