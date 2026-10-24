# Feature evaluation report

## Status

The requested controlled comparison of baseline RAG, hybrid RAG, graph augmentation, adaptive Gemma routing and Jev classification is **incomplete**. No reviewed development/validation/test labels for these features and no cost-accounting workload are configured in this checkout. No live model calls or benchmark runs were performed for this report.

## What was checked

Automated software tests use synthetic fixtures and mocked model providers. They verify local redaction for supported labelled fields, that redacted text is used for indexing/prompts, that unsupported legacy indexes are blocked until re-index, that clear rule-evasion requests stop before retrieval/model calls, and that the existing API/UI regressions still pass.

Actual commands and results on 2026-10-24:

- `python -m pytest backend/tests -q` — 98 passed; one upstream Starlette `python_multipart` deprecation warning.
- `npm test -- --run` — 8 passed.
- `npm run lint` — passed.
- `npm run build` — passed.

These counts measure software behavior only. They are not correctness, citation-support, unsafe-acceptance, false-refusal, redaction-recall, leakage, routing-quality, graph-quality, latency, or cost results.

## Required measurements not yet available

| Requested measure | Result |
|---|---|
| Correctness and citation support against official policy labels | Not measured |
| Unsafe-request refusal / false-refusal rate on reviewed labels | Not measured |
| Redaction recall / false-redaction rate | Not measured; synthetic cases only |
| Sensitive leakage into prompts, indexes and logs | Targeted canary assertions pass; no comprehensive leakage audit or zero-leak claim |
| Gemma routing decisions / context failures | Not measured; model router not implemented |
| Jev quality and invocation overhead | Not measured; Jev integration not implemented |
| Regulatory graph retrieval / relationship errors | Not measured; only source provenance graph exists |
| Graph construction, request latency and total workload cost | Not measured |
| Same-question baseline/hybrid/graph/routing comparison | Not run |

## Completion criteria

Create human-reviewed labels over the same versioned official documents for development/validation and a held-out test split. Record raw predictions and provider usage only when returned. Keep retrieval and answer metrics separate, manually review citation entailment, and calculate ingestion plus indexing plus graph/classification plus generation and hosting costs over a declared workload. Do not tune against the held-out split. Until that work is complete, feature quality and total cost remain unverified.
