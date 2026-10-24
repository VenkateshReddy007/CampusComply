# Baseline audit

## Evidence and audit history

The initial Phase 1 workspace did not contain the starter notebook. The user later supplied `C:\Users\ADMIN\Downloads\02_rag_starter_kit (1).ipynb`; its code cells were inspected without executing the notebook. Saved outputs were inspected only for metadata and credential-like patterns. The original download was left untouched.

The upstream [problem statement](https://github.com/mithun-srinivas/HacktoberFest-HackDay/blob/main/problem-statements/02_rag_application.md) describes the Colab starter and its goals. This section records the actual supplied notebook implementation rather than inferring behavior from the upstream description.

## Actual notebook dependencies and execution flow

The installation cell runs `pip install -q pypdf sentence-transformers "gradio>=6,<7" google-genai`. The notebook later imports NumPy; it is available transitively in the Colab environment/model dependencies but is not explicitly installed by that cell. There was no lock file or pinned dependency set.

The notebook runs these stages in order:

1. Set the baseline model, chunk, retrieval, prompt, and temperature constants.
2. Read a Colab secret, initialize a Google GenAI client when a key is present, and otherwise select a demonstration mode.
3. Load `SentenceTransformer` for dense embeddings.
4. Define an in-notebook sample syllabus with invented course details and policies. It is synthetic demonstration content, not a verified campus source.
5. Create a global knowledge-base dictionary, extract PDF page text, chunk pages, embed chunks, and rank question embeddings against stored vectors.
6. Build a prompt from the retrieved passages, call `generate()`, and return `(reply, found)` from `answer()`.
7. Launch a Gradio upload/question interface with `share=True`, including the retrieved passages in the UI.

The notebook also makes a test generation request during its initialization cell if a Colab key is available. The notebook was not executed during this audit or refactor; no live Gemma request was made by this work.

## Baseline functions and representation

| Component | Actual behavior in the supplied notebook |
| --- | --- |
| `read_pdf(path)` | Uses `pypdf.PdfReader`; returns one string per page using `page.extract_text() or ""`. It does not OCR scanned/image-only PDFs. |
| `split_into_chunks(pages)` | Splits each page independently with `str.split()`, starts at step `max(1, CHUNK_SIZE - CHUNK_OVERLAP)`, joins each window with spaces, and keeps chunks only when they contain more than five words. Each chunk contains `page` and `text`. |
| `build_knowledge_base(pages, name)` | Chunks pages; raises `ValueError` when no chunks survive; encodes chunk text with SentenceTransformer and `normalize_embeddings=True`; mutates the module-global `knowledge_base` dictionary and returns a formatted status string. |
| `retrieve(question, top_k=TOP_K)` | Encodes and normalizes the question; computes `knowledge_base["vectors"] @ question_vector`; sorts descending and returns up to top-k chunk dicts with a float `score`. No relevance threshold is applied. |
| `generate(user_text, system_text, temperature, demo_reply)` | Calls `client.models.generate_content` with the selected Gemma ID, system instruction, temperature, and `thinking_level="minimal"`. It returns text strings for API errors and empty responses rather than raising a typed error. If no key is configured, it returns `demo_reply`. |
| `answer(question)` | Retrieves chunks, formats `[Page N]` context, calls `generate()`, and returns `(reply, found)`. The source list is preserved even if `reply` is an error string. |
| `knowledge_base` | A mutable global dictionary with shape `{"name": None, "chunks": [], "vectors": None}`. The document name is stored at the top level; each chunk has page/text only. There is no multi-document isolation or persistence. |
| Citations and evidence | The prompt asks for page-only citations such as `(p. 2)`. It does not pass the document name in the generation context. The UI displays each retrieved page, similarity score, and passage text. Citation correctness is not programmatically checked. |
| Gradio/API configuration | Uses `gr.File(..., type="filepath")`, callbacks to load/ask, and `demo.launch(share=True)`. It can read a Colab secret or environment variables. It has no authentication or server API layer. |

## Exact baseline settings

These values are present in the supplied notebook code. They are preserved as the selectable defaults in `RAGSettings.baseline()`; they are not claimed to be optimal.

| Setting | Notebook value |
| --- | --- |
| `CHUNK_SIZE` | `120` words |
| `CHUNK_OVERLAP` | `30` words |
| `TOP_K` | `3` |
| `EMBED_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` |
| `GEMMA_MODEL` | `gemma-4-26b-a4b-it` |
| `TEMPERATURE` | `0.2` |

The notebook uses normalized vectors plus their dot product, which is cosine similarity for unit-length vectors. It does not configure a relevance threshold.

## Reuse and refactoring decisions

The shared core retains pypdf page extraction, the word-window chunking algorithm and short-fragment rule, normalized SentenceTransformer embeddings, dense cosine ranking, top-k behavior, the Gemma model/configuration call, and context-only answer instructions. It improves document metadata by carrying the filename and one-based PDF page on every chunk and source result. It also rejects generated text unless it contains at least one citation matching retrieved document/page metadata; this presence check does not establish that every individual claim is fully supported.

Refactoring was needed to remove the global knowledge-base state, Colab/Gradio imports from core, demonstration answers, string-formatted provider failures, and page-only citations. The core now uses typed settings and injected interfaces, with a structured answer status, optional answer text, error information, and actual retrieved evidence. Generation errors return `answer=None`, never a normal policy answer.

The original notebook is retained as `notebooks/02_rag_starter_kit_baseline.ipynb`, with saved outputs and execution counts cleared, a prominent warning that its embedded sample policy text is synthetic/reference-only, and the standalone secret-display expression neutralized. The working `notebooks/02_rag_starter_kit.ipynb` installs and imports the shared package, keeps Gradio in notebook code, omits the synthetic sample policy document, and uses the package for all RAG operations. It defaults to `share=False` rather than the baseline's `share=True`, so document uploads are not automatically exposed through a public Gradio link.

## Credential and synthetic-data findings

One saved output belongs to a cell that reads a Colab secret and contained a long token-like value. It did not match the common Google API-key pattern, but it could not safely be ruled out as sensitive. Its value is not reproduced here. Saved outputs were therefore cleared from the repository notebook copies. The baseline reference also neutralizes the expression that would have echoed that secret on execution. The original file in Downloads was not changed. The supplied notebook reads credentials from Colab Secrets/environment variables and does not hardcode a key; its old standalone cell could nevertheless persist the returned value in notebook output if executed.

The notebook embeds a synthetic sample syllabus. It is omitted from the shared-core notebook so it cannot be mistaken for official policy evidence. It was not copied into package tests or application data.

## Files and dependency gaps

- `docs/source_problem_statement.md` is not present; the upstream statement was checked online.
- Before Phase 2 there was no backend package, dependency manifest, test suite, frontend, or application source in this workspace.
- The starter's dependency cell did not pin versions and omitted an explicit NumPy install despite later importing NumPy.
- The package manifest now declares pypdf and NumPy as core dependencies; SentenceTransformers, Google GenAI, and Gradio are optional integrations.

## Commands and test limits

Notebook inspection read JSON/code and scanned output metadata only; it did not execute notebook cells. No Gemma or embedding-model API/quota calls were made. Phase 2 unit tests use fake embedders and generators; see the phase report for the exact command and result.
