# CampusReg architecture

## Shared RAG core

`backend/campusreg/` is the single implementation used by local Python code
and `notebooks/02_rag_starter_kit.ipynb`. It has no Colab or Gradio imports.
The notebook remains a UI/reference client. Typed `RAGSettings` provides the
notebook baseline and environment overrides.

```text
verified PDF bytes
    -> pypdf physical pages
    -> overlapping word chunks with document/page/chunk identity
    -> SentenceTransformer vectors + persistent SQLite storage
    -> dense cosine or BM25 + Reciprocal Rank Fusion retrieval
    -> optional configuration-bound evidence gate
    -> Google GenAI Gemma generation from the current retrieved passages
    -> structured answer, citations, and exact retrieved evidence
```

`SQLiteDocumentStore` stores source PDFs, metadata, pages, chunks, vectors,
conversation records, and messages. Index replacement and document deletion
are transactional. Dense scores are cosine similarity; BM25 and RRF scores
retain explicit method diagnostics. None are probabilities of factual
correctness. Citation syntax validation maps an answer citation to a retrieved
source; human review remains necessary to determine whether the source
supports the claim.

`ConversationRAGService` persists separate conversation histories. A bounded
query-rewriting step can resolve references in a follow-up, after which the
system retrieves current passages again. Historical assistant answers are
never included as official evidence in final answer generation.

`ConflictDetector` is a separate core service. Candidate pairs are compared
through an injected comparator; the Gemma adapter requests a structured
classification, but the output is only returned after literal quote/scope and
verified date checks against indexed passages and document metadata. Reports
carry both source passages and require human review. No authority hierarchy is
implemented.

## API and frontend

`backend/campusreg/api/app.py` exposes typed FastAPI routes which call the
shared engine, conversation service, SQLite store, and conflict service. It
does not contain retrieval or generation algorithms. Uploaded source bytes
and the database default to `data/campusreg.sqlite3`, outside the package.
The local API key is read server-side. Development evaluation routes require
an opt-in flag and backend bearer token. CORS origins are explicit and
environment-controlled.

`frontend/` is a Next.js App Router client written in TypeScript. Its API
client models the Pydantic response contracts. Answer text, citations,
passages, documents, conflicts, and evaluation metrics come from the API; the
frontend has no production mock answers or sample policy records. Synthetic
PDFs can be generated locally for software-only checks and are ignored by
Git.

## Evaluation

`backend/campusreg/evaluation/` contains reviewed-dataset loading, metrics,
cacheable predictions, evidence-gate calibration, retrieval experiments,
paired no-RAG/RAG evaluation, fixed-evidence Gemma comparison, and dense/BM25/
RRF comparison. Full benchmark commands are explicit; imports and application
startup do not run evaluations. Smoke tests use clearly marked synthetic
fixtures and mocked model providers.

The final dataset must contain at least 15 human-reviewed questions across
development, validation, and held-out test splits. Expected claims, answer
correctness, unsupported claims, citation support, evidence support, and
refusal correctness require human review. Synthetic fixture text must not be
copied into the official evaluation corpus.

## Configuration and boundaries

Baseline settings are Gemma `gemma-4-26b-a4b-it`, embedding model
`sentence-transformers/all-MiniLM-L6-v2`, 120-word chunks, 30-word overlap,
top-k 3, temperature 0.2, and dense retrieval. Hybrid retrieval defaults to
BM25 `k1=1.5`, `b=0.75`, candidate count 20, and RRF constant 60. Thresholds
must be calibrated separately for the embedding, retrieval, chunking, and
corpus configuration used; score cutoffs are not confidence probabilities.

API credentials are read from process environment and are not written to
results. OCR, production authentication/authorization, institution tenancy,
deployment hardening, and retention controls remain outside the implemented
scope. No official campus corpus or reviewed benchmark dataset is checked in,
so no campus-policy accuracy or real-data conflict metric is available from
this repository.

See [the backend guide](../backend/README.md) and phase notes for details on
[multi-document ingestion](phase-3-multi-document.md),
[contextual conversations](phase-8-conversations.md),
[fixed-evidence model comparison](phase-9-model-comparison.md), and
[hybrid retrieval](phase-10-hybrid-retrieval.md).
