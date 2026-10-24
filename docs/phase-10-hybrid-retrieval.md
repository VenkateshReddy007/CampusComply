# Phase 10: hybrid keyword and semantic retrieval

## Retrieval implementation

Dense cosine retrieval remains the default and public baseline. `RAGSettings`
can select `retrieval_method="hybrid"`; the answer pipeline then uses both
dense candidates and BM25 keyword candidates, fused by Reciprocal Rank Fusion:

```text
RRF(chunk) = sum(1 / (rrf_constant + one_based_rank))
```

BM25 uses Unicode case-folded word tokens, `k1=1.5`, and `b=0.75`. The
defaults are 20 candidates per ranker, final `top_k` from settings, and RRF
constant 60. No raw cosine and BM25 scores are added together. Retrieval
results use the same structured source record (document ID/name, physical page,
chunk ID, original text, method, method-specific score, and diagnostics).
Hybrid scores are explicitly tagged as RRF scores, not probabilities.

Dense-only, BM25-only, and hybrid RRF can be compared as ablations. Optional
reranking is not enabled. Evidence-gate calibration includes retrieval method,
candidate count, RRF constant, and corpus revision in its configuration key,
so a dense calibration cannot silently transfer to hybrid retrieval.

## Held-out comparison

Smoke test (one synthetic held-out fixture, no network calls):

```powershell
python -m campusreg.evaluation.hybrid_comparison --smoke
```

Full evaluation validates the reviewed dataset and source manifest, then runs
only the held-out `test` split. It uses the same index, questions, chunking,
embedding model, and final top-k across all three retrieval methods:

```powershell
python -m campusreg.evaluation.hybrid_comparison --full --dataset ..\evaluation\questions.jsonl --corpus-manifest ..\evaluation\corpus-manifest.json
```

`--candidate-count` and `--rrf-constant` select declared experimental
parameters. Do not tune them on the held-out set. Outputs contain raw JSONL,
CSV, experiment configuration, and a markdown report. Metrics are retrieval
Recall@k, MRR@k, and latency; no generation is involved.

## Results and limitations

Unit tests verify exact-number/deadline keyword retrieval, dense baseline
preservation, RRF diagnostics and source metadata, and a synthetic ablation.
These fixtures are not campus regulations. The checkout has no reviewed
official held-out questions or source corpus; therefore, no real Recall@k,
MRR@k, latency comparison, or claim that hybrid is better is available.
BM25 tokenization and the fixed `k1`/`b` values may not fit every document
format; any changes require a new recorded experiment configuration. The
reviewed held-out set should cover exact rule names, numeric values and
deadlines so keyword retrieval has a fair, human-labeled test.
