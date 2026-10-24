# Evaluation report

## Dataset and policy-quality results

No official institutional PDF corpus and no human-reviewed campus question
set are included in this workspace. The challenge requirement for at least 15
reviewed questions, manually verified citations, and a measured percentage
answered correctly is therefore **not met yet**. No campus-policy accuracy,
retrieval Recall@k/MRR, refusal-quality score, model comparison, or real-data
conflict precision/recall is reported.

The evaluation template and runner support development, validation, and held-
out test splits, but synthetic fixtures are deliberately isolated to software
tests and are not used as policy evaluation data. The Gemma API is not called
by automated tests.

## Conflict fixture contract check

The conflict unit test uses six explicitly labelled synthetic pairs and a
deterministic test comparator. It checks that the service preserves the six
labels, rejects a forged quote, verifies dates/scopes, and calculates precision
and recall from labelled predictions. The resulting perfect fixture assertion
is a test-double contract check, not a measurement of model or production
performance.

## Software checks

Commands run locally on 2026-10-24:

```powershell
# From backend/
python -m pytest tests -q

# From frontend/
npm test
npm run lint
npm run typecheck
npm run build
```

Actual results:

- Backend: `81 passed, 1 warning in 8.38s`. The warning is a Starlette
  `PendingDeprecationWarning` about importing `python_multipart` instead of
  `multipart`.
- Frontend: `npm test` — 5 tests passed.
- Frontend: `npm run lint` — passed.
- Frontend: `npm run typecheck` — passed.
- Frontend: `npm run build` — passed; Next.js 16.4.3 generated the `/` and
  `/_not-found` routes.
- Local smoke: the web root, `/api/health`, and `/api/ready` returned HTTP 200.
  Readiness reported `degraded` because no model API key was configured.
- Local ingestion smoke: the two clearly labelled synthetic PDFs indexed;
  each contained 1 extracted page, 1 chunk, and a 384-dimensional embedding.
  They are excluded from Git and are not policy-evaluation data.
- No live Gemma request was made. Question generation and model-assisted
  conflict comparison therefore remain unverified in this local run.

The API integration suite uses synthetic PDFs, fake embeddings, and mock
generation/comparison providers. It checks upload/indexing, exact source
passages, citations, abstention, conversations, conflict responses, missing
model configuration, and error/non-answer separation. It does not test Google
availability, real embeddings on official PDFs, legal interpretation, or
claim-to-source support by a human reviewer.

## Reproduction and reporting

Use the commands in `evaluation/README.md` only after supplying official
source PDFs, a corpus manifest, and a human-reviewed dataset. Keep test split
questions untouched during tuning. Record config/model/corpus revision,
timestamps, raw predictions, and review labels. Do not copy synthetic test
metrics into the public policy-quality score.
