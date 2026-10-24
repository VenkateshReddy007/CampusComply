# CampusReg request guardrails

## Intended use

CampusReg answers campus and education policy questions only from retrieved indexed evidence. Unsupported questions abstain, and factual answers require document/page citations. Retrieved PDF text is untrusted data: the generation system instruction explicitly says to ignore instructions embedded in source passages. Source text cannot authorize tool use, change policy, or request secrets.

## Local preflight

Before retrieval, embeddings or generation, a deterministic request screen blocks clear actionable requests to exploit or abuse rule loopholes, evade/bypass/circumvent requirements, reverse-engineer rules for an unfair advantage, or manipulate exceptions/procedures. Blocked requests receive a fixed safe redirection to documented explanation, legitimate appeals/clarification, or fair rule improvement. Conversation turns are screened before query rewriting as well.

The phrase rules deliberately permit ordinary rule understanding, appeals, ambiguity review and fairness-improvement requests. This is a narrow baseline; it is not a general safety classifier and may miss paraphrases or misclassify ambiguous requests. No classifier is currently used for enforcement, and no Jev integration or measured policy-risk benchmark is present.

## Evidence and output handling

An evidence gate may abstain before generation. Generation/provider errors and uncited or invalidly cited output remain structured failures rather than normal policy answers. Citation validation checks that labels match retrieved source labels; it does not independently prove that every claim is entailed by its cited passage. Human review remains necessary for consequential interpretations and possible cross-document conflicts.

## Test status

Synthetic unit and API tests cover clear evasion attempts, safe appeal/clarification/fairness requests, blocking before embedder/generator calls, and the shared conversation path. These tests are not a reviewed operational dataset and do not establish perfect protection. See the test command recorded in the phase completion report.
