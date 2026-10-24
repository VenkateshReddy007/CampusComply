# CampusReg evaluation dataset and methodology

## Retrieval-only experiments

`retrieval-experiment.config.json` declares the fixed 18-run grid: two
embedding models, three word chunk sizes, 30-word overlap, and three values
of `top_k`. The runner imports chunking and cosine retrieval from the shared
`campusreg` core. It runs development and validation questions only; it does
not call Gemma or include generation settings. It rebuilds one index per
embedding model, chunk size, overlap, and corpus fingerprint, and reuses that
index for the three `top_k` variants. Query vectors are cached separately for
each embedding model and dataset version, and model-specific index/query
dimensions must agree before similarity is computed.

The selected configuration is the highest complete validation Recall@3, then
MRR@3; ties use lower mean query-embedding-plus-vector-search latency, lower
vector-index bytes, and experiment ID. The rule is recorded in the plan and
each run. It does not inspect held-out test questions. Incomplete or failed
configurations cannot win. Latency excludes model loading and the first warm-up
call; indexing time includes chunking and document embedding. Memory reporting
includes exact vector-index bytes and model parameter bytes when exposed by the
embedding runtime, but not process peak RSS or non-parameter runtime overhead.

To run against reviewed policy material, first prepare `questions.jsonl`
containing the complete reviewed challenge dataset and a local corpus manifest.
Each manifest document has `document_id`, `filename`, relative or absolute
`path`, and the exact PDF `sha256`; paths in the manifest are resolved from the
manifest directory. The runner verifies hashes and reads physical pages from
those PDFs. It will reject missing expected document IDs or page references.
The runner only selects development and validation rows; test rows are
excluded from both metrics and selection.

From the repository root, install the core with the embedding extra and run:

```powershell
python -m pip install -e "backend[embeddings]"
campusreg-retrieval-experiments --dataset .\evaluation\questions.jsonl --corpus-manifest .\evaluation\corpus-manifest.json
```

Equivalent module invocation:

```powershell
python -m campusreg.evaluation.retrieval_experiments --dataset evaluation/questions.jsonl --corpus-manifest evaluation/corpus-manifest.json
```

Outputs go under ignored `evaluation/results/retrieval-experiments/` by
default: `results.csv`, `experiment_config.json`, `best_configuration.json`,
`failure_cases.jsonl`, `interpretation.md`, resumable `observations.jsonl`, and
model/corpus-specific caches. A changed question set, PDF hash/content, model
ID, chunk size, overlap, or algorithm revision gets a different cache key.
Successful per-question observations are reused after interruption; failed
ones are retried. The default 18-configuration plan and selection rule are
versioned in source control. No result or winner is valid until the runner has
actually completed against the intended reviewed dataset and matching PDF
manifest.

## Controlled no-RAG versus RAG generation

Use `campusreg-paired-evaluation --smoke` for the synthetic mocked smoke test.
It makes no network calls. The full benchmark is a separate explicit command:

```powershell
campusreg-paired-evaluation --full --dataset .\evaluation\questions.jsonl --corpus-manifest .\evaluation\corpus-manifest.json --output-dir .\evaluation\results\paired
```

Each question is sent to the no-RAG condition as question text only; regulation
passages are passed only to the RAG condition. Both records retain the same
question, expected answer, model ID, and generation configuration. Raw paired
records are cached by question and run fingerprint. Completed outputs can be
reviewed by editing their nullable `human_labels`; a rerun rebuilds the CSV and
report from cache without another call. Labels require a reviewer and a
timezone-aware `reviewed_at`. Record answer correctness, unsupported claims,
citation/evidence support, refusal correctness, and a comparison tag such as
`rag_helped`, `rag_misleading_evidence`, or `rag_incomplete_evidence`. Citation
syntax and mapping are deterministic checks only; they do not establish source
support or factual correctness. The generated report leaves unreviewed metrics
unreported and only gives latency percentiles when at least five calls exist.

## Evidence gate calibration

Review every validation RAG output before calibration. For each answered
validation row, set `human_labels.answer_correct` and
`human_labels.unsupported_claims_present`; for unanswerable rows, review
`human_labels.appropriate_refusal`. Every populated label requires a reviewer
and a timezone-aware timestamp. Calibrate each retrieval/corpus configuration
independently, with the acceptable risk limits chosen explicitly:

```powershell
campusreg-evidence-gate calibrate --predictions .\evaluation\results\paired\paired_predictions.jsonl --output-dir .\evaluation\results\gate --max-false-accept-rate 0.05 --max-unsupported-answer-rate 0.10
```

Those limits are policy trade-offs supplied by the operator, not measured
probabilities. Within both limits the calibrator selects the threshold with
highest validation coverage, then answer correctness, then lower false
refusals. It writes a distinct score cutoff for each embedding/retrieval
configuration and corpus revision. If no cutoff meets both limits, that
configuration receives no deployable policy. Retrieval scores, including RRF
scores, are always labeled as scores rather than probabilities.

After freezing the policies, score the test split without changing thresholds:

```powershell
campusreg-evidence-gate test --calibration .\evaluation\results\gate\evidence_gate_calibration.json --predictions .\evaluation\results\paired-test\paired_predictions.jsonl --output-dir .\evaluation\results\gate-test
```

At runtime, load one policy with `EvidenceGate.from_file(...)` and inject it
into `answer`, `RAGEngine`, or `MultiDocumentRAGEngine`. A rejected gate returns
a structured abstention before Gemma is called, with the retrieved passages,
score, threshold, configuration ID, and refusal reason preserved for review.

This directory defines the evaluation format and tooling contract. It does not
contain a final campus-policy question set or reported benchmark. No expected
campus-policy answer has been authored for this phase. Add questions only after
reviewing the relevant verified official documents; use their stored document
IDs and physical PDF page numbers.

`questions.template.jsonl` is one placeholder record, not an evaluation
question. Copy it, replace every placeholder, and mark a record `ready` only
after human review. Keep synthetic policy text out of development, validation,
and held-out test datasets. Synthetic fixtures belong only in isolated
automated software tests and must say `SYNTHETIC` in the fixture text/name.

## Dataset size and categories

The framework accepts at least 15 ready questions and is intended for a
manually reviewed set of roughly 30–40. Include a balanced mix of:

- `direct_factual`
- `multiple_passages`
- `multiple_documents`
- `exceptions_or_effective_dates`
- `unanswerable`
- `follow_up`
- `potential_conflict`

Follow-up records can include prior user/assistant turns in
`conversation_context`. `expected_pages` is a set of `{document_id,
page_number}` references to the original PDFs. For answerable records, the
expected answer, source IDs/pages, expected claims, and official-source
verification require human review. Unanswerable records have `answerable: false`
and null/empty expected policy evidence. `expected_conflict` is reviewed
independently of whether a model detects one. A positive conflict label needs
two separately reviewed `expected_claims`; both may cite passages on the same
physical PDF page, so the record preserves passage distinctions without
inventing separate page numbers.

`dataset.schema.json` describes each JSONL question; the loader also enforces
semantic checks such as both pages for a reviewed conflict and conversation
history for a follow-up. `prediction.schema.json` describes each raw output row
and its nullable human review labels. `validate_challenge_dataset()` checks
that a completed set has at least 15 reviewed rows and has development,
validation, and held-out test coverage. It makes no model calls.

`split` is one of `development`, `validation`, or `test`. Put related questions
and near-duplicate phrasings in the same split. Choose the held-out test set
before tuning and do not use its metrics to select thresholds, prompts, model
settings, or retrieval parameters. Use development/validation results for
iteration. The runner evaluates only the requested split and never tunes a
setting.

## Metrics and their labels

Retrieval is reported separately from answer quality:

- **Retrieval Recall@k and MRR@k** use human-reviewed expected document/page
  references. If `expected_pages` is present, relevance is measured at the
  document/page level; otherwise expected document IDs are used. Unanswerable
  records have no retrieval denominator.
- **Answer correctness** is measured only from a human label in the raw
  prediction (`human_labels.answer_correct`) with a recorded reviewer and
  review timestamp. No string overlap score or LLM judge is treated as ground
  truth.
- **Citation existence** says whether the answer contains a citation label.
  **Citation validity** says whether every cited label maps to a retrieved
  source. Neither means that the cited passage supports the associated claim.
- **Citation support** and **evidence support** are separate human labels in
  `human_labels.citation_support` and `human_labels.evidence_support`.
  Reviewers check claim-to-passage support against the official PDF; a label
  without reviewer metadata is rejected by the metrics code.
- **Appropriate refusal** is reported from a human review label. A separate,
  reproducible `abstention_behavior_match` checks whether an answerable
  record was answered and an unanswerable record was abstained on. It is a
  behavior check, not a substitute for human judgment about a particular
  refusal.
- **Potential-conflict detection** is scored against `expected_conflict` only
  when a conflict detector is supplied and returns a label. Its precision,
  recall, F1 and accuracy are reported separately.
- **Latency and failed requests** include the measured answer call duration,
  failure count/rate and conflict-detector failures. Generation errors and
  thrown request exceptions count as failed requests; an invalid generated
  answer is reported by status but is not an API transport failure.

Review predictions after a run by filling the nullable `human_labels` fields
in its `predictions_<split>.jsonl`. Re-run the same evaluation to recompute
metrics: cached predictions are reused, including those labels. If the system,
index, prompt, or model settings change, update `run_config` so its fingerprint
changes and old predictions are not reused.

## Running an evaluation

Build and review the dataset first, then connect the shared engine without
making an evaluation request at import time:

```python
from campusreg.config import RAGSettings
from campusreg.evaluation import EvaluationRunner, load_dataset
from campusreg.rag import GoogleGenAIProvider, MultiDocumentRAGEngine, SentenceTransformerEmbedder
from campusreg.storage import SQLiteDocumentStore

settings = RAGSettings.from_env()
embedder = SentenceTransformerEmbedder(settings.embedding_model)
generator = GoogleGenAIProvider.from_env()
rag = MultiDocumentRAGEngine(
    settings, embedder, generator, SQLiteDocumentStore("data/campusreg.sqlite3")
)
questions = load_dataset("evaluation/questions.jsonl", split="development")
run = EvaluationRunner().run(
    questions,
    answerer=rag.answer,
    output_dir="evaluation/results/development",
    split="development",
    retrieval_k=5,
    run_config={
        "gemma_model": settings.gemma_model,
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": settings.top_k,
        "temperature": settings.temperature,
        "index_revision": "REPLACE_WITH_A_RECORDED_INDEX_REVISION",
        "prompt_revision": "REPLACE_WITH_A_RECORDED_PROMPT_REVISION"
    },
)
```

Calling `run()` may make model requests and consume API quota. The framework
does not run automatically. Tests inject deterministic local fakes. The
`run_config` must identify the corpus/index and evaluation-relevant code or
prompt revision as well as model settings; it is hashed into each cache key.

Each run writes raw predictions and retrieved passage text to
`predictions_<split>.jsonl`, aggregate and per-category metrics to
`metrics_<split>.json`, and flattened metrics to `metrics_<split>.csv`.
Prediction rows follow `prediction.schema.json`. Predictions are flushed after each question so an
interrupted run can resume. Results belong under the ignored
`evaluation/results/` path and must not be committed with uploaded documents.

No accuracy value is available until a real reviewed dataset has been run and
the resulting predictions and metrics have been inspected. This phase creates
the framework only; it does not create or report benchmark results.

## Contextual follow-up evaluation

Use the same reviewed `follow_up` records and their `conversation_context` with
`EvaluationRunner.run(..., conversation_service=service)`. The runner seeds
each record's bounded test transcript into a fresh isolated conversation and
asks the original current question. It records the original question and the
rewritten retrieval query. Review answer correctness, citation support, and
evidence support with human labels. See [the conversation implementation and
limitations](../docs/phase-8-conversations.md). No reviewed official
multi-turn dataset or measured follow-up result is included yet.

## Fixed-evidence Gemma model comparison

Run the two-model smoke test with a mocked provider:

```powershell
python -m campusreg.evaluation.model_comparison --smoke
```

For a real comparison, use the explicit `--full` command documented in
[Phase 9](../docs/phase-9-model-comparison.md). It first retrieves fixed
passages once per question, then provides the exact same passages and prompt
to both specified model IDs. Results are cached and written as raw JSONL,
side-by-side CSV, and a markdown report under `evaluation/results/`. Model
correctness, unsupported claims, citation support, evidence support, and
refusal quality remain human-reviewed labels. A failed model makes the run
incomplete and is recorded without substitution. This checkout has no
reviewed official question set or corpus, so no model-quality metrics are
available.

## Hybrid retrieval comparison

Run the mocked synthetic smoke test with:

```powershell
python -m campusreg.evaluation.hybrid_comparison --smoke
```

The explicit full command compares dense, BM25-only, and dense+BM25 RRF on the
same held-out test questions and verified corpus:

```powershell
python -m campusreg.evaluation.hybrid_comparison --full --dataset ..\evaluation\questions.jsonl --corpus-manifest ..\evaluation\corpus-manifest.json
```

BM25 uses `k1=1.5`, `b=0.75`; fusion uses ranks only, with a recorded RRF
constant. Its retrieval score is not a correctness probability. Outputs are
stored in `evaluation/results/hybrid-comparison/`; see [the Phase 10
methodology](../docs/phase-10-hybrid-retrieval.md). The full evaluation has not
been run because reviewed official questions and PDFs are absent.
