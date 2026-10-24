# Phase 9: Gemma model quality and speed comparison

## Method

`campusreg.evaluation.model_comparison` compares exactly
`gemma-4-26b-a4b-it` (the baseline) and `gemma-4-31b-it`. It constructs a
single fixed evidence set per question with the shared retrieval pipeline;
both model calls receive the same evidence bytes, citation labels, question,
system instruction, and temperature. Only the `model` argument changes.
There is no silent model fallback. Each row stores model ID, question ID,
timestamp, status, latency, output, citation structure checks, evidence, and
nullable human-review labels. Completed rows are cached per question, model,
evidence fingerprint, and run configuration.

The CSV pairs results side-by-side. The report separates structural citation
validity from human-reviewed citation support and factual correctness. It
reports latency percentiles only after at least five calls per model and
marks the comparison incomplete if any expected model/question call fails.
Use `--force` to retry cached failures or outputs deliberately.

## Commands

The smoke mode uses two clearly marked synthetic questions and a mocked
provider; it has no network access:

```powershell
python -m campusreg.evaluation.model_comparison --smoke
```

The full comparison is explicit and requires an already reviewed dataset,
verified PDF manifest, configured Google GenAI credentials, and the embedding
extra. Select the split explicitly so development/validation or held-out test
is run intentionally:

```powershell
python -m pip install -e ".[embeddings,google]"
python -m campusreg.evaluation.model_comparison --full --dataset ..\evaluation\questions.jsonl --corpus-manifest ..\evaluation\corpus-manifest.json --split test
```

Use `--output-dir` to select a result directory and `--force` to bypass cached
results. The command makes external model calls only after it is explicitly
invoked with `--full`.

## Results and limitations

The smoke tests verify identical prompt/evidence/settings, persistence, cache
reuse, and failure handling. They are not Gemma performance measurements.
There is no reviewed official question dataset or PDF corpus in this checkout,
so no real comparison, quality score, or model winner is reported. Model
availability, request speed, and answer quality remain unmeasured until the
full command runs and outputs receive human review.
