# Phase 11 — Cross-document conflict detection

## Implementation

`backend/campusreg/rag/conflicts.py` adds a separate service with a structured
report schema. It proposes deterministic cross-document candidate pairs from
shared non-stopword terms, then calls an injected `ConflictComparator`. The
Gemma adapter requests structured classifications; neither it nor the API
selects an authoritative regulation.

Before a report is returned, both document IDs/pages are taken from the actual
indexed passages, quote text must be a literal substring of its source,
scope excerpts must also be literal, and dates must match passage text or
verified document metadata. Failed or invalid comparisons become
`insufficient_evidence`; they cannot invent source references. Reports expose
both original passages and always require human review.

The status set separates potential contradiction, possible supersession,
different scopes, valid exception, non-conflicting near duplicates, and
insufficient evidence. The service has no source hierarchy or effective-date
policy that can determine authority.

`docs/conflict-report.schema.json` documents the output object.
`conflict_metrics()` computes labelled binary precision/recall and returns
false-positive/false-negative IDs.

## Synthetic test cases and measured test behavior

`backend/tests/test_conflicts.py` has explicitly labelled synthetic pairs for
all six requested classes. Mock comparison output is exercised against the
exact pair; a second test verifies that a fabricated quote and an unverified
date are suppressed. The precision/recall assertion is a software contract
check against a deterministic labelled fake comparator. It is **not** an
estimate of Gemma or production conflict-detection quality.

No official policies or independently human-labelled institutional conflict
set is present. Production precision/recall has not been measured. Lexical
candidate generation can miss paraphrases with little word overlap; results
must be treated as possible issues and reviewed by a qualified person.
