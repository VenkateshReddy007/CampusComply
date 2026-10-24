# Failure analysis and current risks

This document describes observed software failure handling and unmeasured
policy-quality risks. It does not claim that any failure rate has been
measured on institutional documents.

| Failure mode | Current behavior | Evidence/coverage | Remaining limitation |
| --- | --- | --- | --- |
| Missing or unreadable PDF | Rejects invalid MIME/header/extension, size, malformed or image-only PDFs; marks ingestion failure | Synthetic ingestion and API tests | OCR is not implemented |
| No indexed documents | Question API returns a client conflict response | API integration tests | No official document set to verify institutional coverage |
| Unsupported question | Dense score cutoff or evidence gate can abstain; otherwise Gemma is instructed to abstain when context does not support the answer | Mocked refusal tests | No calibrated production threshold or human-reviewed refusal set |
| Generation unavailable/empty | Structured error with `answer: null`; never treated as an answer | Core and API mocked failure tests | Live provider reliability has not been exercised |
| Citation omitted or not retrieved | Core rejects the generation as invalid; API returns an error state without answer citations | Core tests | Citation syntax mapping is not the same as claim support |
| False citation from a conflict comparator | Quote, scope, and date validation fails closed to `insufficient_evidence`; report sources come from indexed passages | Synthetic conflict tests | Candidate generation can miss paraphrases; real precision/recall unknown |
| Supersession or date ambiguity | Report may say `possible_supersession`; both passages retained; no authority decision | Structured schema and mocked test | No verified institutional hierarchy/effective-date policy configured |
| Overlapping scopes or valid exception | Separate labels are available; both exact passages retained | Synthetic fixture contract tests | Classification needs qualified human review |
| Follow-up has ambiguous referent | Clarification response; fresh retrieval follows a resolved query | Conversation tests | Rewriter quality not measured on real multi-turn policy questions |
| Model key absent | Generation-dependent API actions fail with a structured configuration response | API configuration test | Local operator must configure the key for live generation |
| Off-topic question | The generation prompt asks Gemma to abstain outside campus compliance and the core recognizes the configured refusal | Out-of-scope core test | This is prompt-guided, not a formal domain classifier; no live model evaluation has been run |
| Graph implies an unsupported policy relation | Graph edges are limited to document/page/passage containment | Knowledge graph unit/API tests | It is a source provenance graph, not a semantic or policy dependency graph |
| Requested Gemma 2/3 routing | No route is advertised; the configured hosted provider currently uses Gemma 4 | Configuration inspection and current provider documentation | Gemma 2/3 require separately configured and tested inference endpoints plus routing evaluation |
| Key exposure | API readiness reports a boolean only; key is never part of response schema | Static code/test review | Production secret management and deployment still need review |
| Evaluation dataset absent | Protected development runner refuses to start when its configured dataset is absent/empty | Dataset and API route contract | No actual evaluation job can produce policy metrics yet |

## Priority follow-up

1. Obtain two or more current official regulations and verify provenance,
   versions, effective dates, and authority hierarchy with the institution.
2. Have campus reviewers build at least 15 questions (preferably 30–40),
   page-level expected evidence, answerability, conflicts, and held-out splits.
3. Run retrieval and threshold calibration on development/validation data only.
4. Independently review model answers, citation support, refusals, and conflict
   false positives/negatives; preserve raw outputs and failure cases.
5. Add deployment authentication, tenancy, retention/deletion controls, and
   security review before processing personal or institution-confidential data.
