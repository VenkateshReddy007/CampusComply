# CampusReg shared RAG core

The package in this directory is the shared implementation used by the
application and the Colab notebook. The core contains no Google Colab or
Gradio imports. It preserves the starter notebook's PDF page extraction,
word-based overlapping chunks, normalized SentenceTransformer embeddings,
cosine-similarity dense retrieval, and Google GenAI Gemma generation.

## Install locally

From the repository root, install the package and all integrations:

```powershell
python -m pip install -e ".\backend[all]"
```

For the test suite only, use `python -m pip install -e ".\backend[test]"`.
The `all` extra includes the embedding model runtime, Google GenAI SDK, and
Gradio notebook UI. Model weights are downloaded by Sentence Transformers on
first use; they are not bundled in this repository.
The `api` extra also installs the Google GenAI SDK required by the FastAPI
generation provider, so an API-only installation can serve Gemma requests.

Set `GOOGLE_API_KEY` in the process environment (or in a local ignored `.env`
file loaded by your chosen runner). The SDK provider accepts `GOOGLE_API_KEY`,
`GEMMA_API_KEY`, or the legacy Colab secret name `gemma_API_Key`. The key is
read at provider creation and is never logged or included in results.

The shared core refuses requests with no recognized campus or education
topic before retrieval or generation. This is a deterministic topic boundary,
not a general-purpose intent classifier; phrase campus context explicitly
when a question could refer to another domain. In-scope answers still require
retrieved evidence and valid document/page citations.

## Use the shared core

```python
from campusreg.config import RAGSettings
from campusreg.rag import GoogleGenAIProvider, RAGEngine, SentenceTransformerEmbedder

settings = RAGSettings.baseline()
embedder = SentenceTransformerEmbedder(settings.embedding_model)
generator = GoogleGenAIProvider.from_env()
rag = RAGEngine(settings=settings, embedder=embedder, generator=generator)

rag.load_pdf("official-handbook.pdf")
result = rag.answer("What does the handbook say about this topic?")
if result.status.value == "answered":
    print(result.answer)
for source in result.sources:
    print(source.source_name, source.page, source.text)
```

`AnswerResult` always separates the generated answer, retrieved evidence, and
status. An abstention is distinct from a generation failure. On generation
failure, `answer` is `None`, while the failure code and the passages that were
sent to the model remain available as structured fields. A generated response
without at least one citation matching a retrieved document and page is
rejected with status `invalid_response`; this check does not prove that every
claim in an otherwise cited response is fully supported.

## Configuration

`RAGSettings.baseline()` selects the original notebook values:

- Gemma: `gemma-4-26b-a4b-it`
- embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- chunk size: 120 words
- overlap: 30 words
- top-k: 3
- temperature: 0.2
- relevance threshold: unset, matching the starter's lack of a threshold

`RAGSettings.from_env()` reads `GEMMA_MODEL`, `EMBED_MODEL_NAME`,
`CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `TEMPERATURE`, optional
`RELEVANCE_THRESHOLD`, `MAX_PDF_SIZE_BYTES` (20 MiB by default),
`RETRIEVAL_METHOD`, `RETRIEVAL_CANDIDATE_COUNT`, and `RRF_CONSTANT`. The
default method is dense. Dense-only retrieval scores are cosine similarities;
hybrid retrieval uses BM25 and rank-based RRF scores. Thresholds are
configuration-specific retrieval cutoffs, not probabilities of correctness.
Leave the threshold unset to preserve baseline retrieval.

## Persistent multi-document ingestion

The shared core also provides `MultiDocumentRAGEngine` backed by
`SQLiteDocumentStore`. Uploaded PDF bytes, document metadata, extracted page
text, chunks and float32 vectors live together in a local SQLite database.
Use an ignored path such as `data/campusreg.sqlite3`; `.gitignore` excludes
SQLite files. Source PDFs remain available in the database so an index can be
rebuilt after a restart or embedding-model change.

```python
from campusreg.config import RAGSettings
from campusreg.rag import GoogleGenAIProvider, MultiDocumentRAGEngine, SentenceTransformerEmbedder
from campusreg.storage import SQLiteDocumentStore

settings = RAGSettings.from_env()
embedder = SentenceTransformerEmbedder(settings.embedding_model)
generator = GoogleGenAIProvider.from_env()
store = SQLiteDocumentStore("data/campusreg.sqlite3")
rag = MultiDocumentRAGEngine(settings, embedder, generator, store)

with open("verified-source.pdf", "rb") as pdf_file:
    upload = rag.ingest_pdf(pdf_file.read(), "verified-source.pdf")
result = rag.answer("Ask about the uploaded source")
for source in result.sources:
    print(source.citation, source.text)
```

The example filename is a placeholder, not a project source. Production uploads
must be verified official documents. `ingest_pdf` rejects non-PDF content,
files over the configured size limit, malformed PDFs, encrypted PDFs and PDFs
with no extractable text. OCR is not included. Duplicate bytes return the
existing document record by SHA-256. Use `reindex_document(document_id)` to
rebuild one source or `delete_document(document_id)` to remove it and its
dependent page/chunk/vector rows. Index replacement and deletion are
transactional. Reindex failure keeps the previous indexed passages available.

Each page is stored with its physical one-based PDF page number. Chunks and
retrieved sources carry the document ID, filename, chunk ID and original page
text. Multi-document citations include filename, stable document ID and page,
for example `[filename.pdf, document <id>, p. 4]`. `AnswerResult.sources` are
the exact passages placed in the generation prompt. Each score is labeled
with its retrieval method and diagnostics; none is a calibrated correctness
probability.

## Hybrid retrieval and conversations

Set `RETRIEVAL_METHOD=hybrid` to fuse dense and BM25 candidate ranks with RRF.
`RAGSettings` defaults to 20 candidates and RRF constant 60; BM25 uses
`k1=1.5` and `b=0.75`. The retrieved source object exposes its method-specific
score and diagnostics. See [Phase 10](../docs/phase-10-hybrid-retrieval.md)
for the held-out comparison command.

`ConversationRAGService` adds persistent isolated histories and bounded
reference resolution for follow-ups. It rewrites the retrieval query, then
retrieves current evidence again; prior assistant answers are not answer
evidence. See [Phase 8](../docs/phase-8-conversations.md).

## Use in Google Colab

Check out the repository in the Colab runtime, then open
`notebooks/02_rag_starter_kit.ipynb` from that checkout. Its setup cell finds
the checkout root, installs `backend[notebook]` in editable mode, and imports
the same `campusreg` modules used locally. Add a Google AI Studio key to
Colab Secrets under `GOOGLE_API_KEY` (or `gemma_API_Key` for compatibility);
the notebook reads it into the runtime environment without displaying it or
writing it to disk. Upload verified official source PDFs before asking
policy questions; the starter's synthetic sample syllabus is not used. The
notebook Gradio app launches with public sharing disabled by default; its
baseline used `share=True`.

The original Colab notebook code is retained separately as
`notebooks/02_rag_starter_kit_baseline.ipynb`. Its saved outputs are cleared
in the repository copy because a saved output from a Colab secret lookup was
token-like and could not safely be ruled out as sensitive. The reference also
neutralizes the expression that would echo that value and labels the embedded
sample policy text as synthetic and reference-only.
