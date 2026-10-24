# Phase 8: contextual follow-up questions

## Implementation

`ConversationRAGService` stores isolated conversation records and timestamped
user/assistant messages in the same SQLite store as document metadata. Each
assistant row retains the original question, the standalone retrieval query,
answer status, and the evidence passages used for that answer. Clear removes
messages while preserving the conversation ID; delete cascades to all messages.

Follow-up detection is deliberately conservative and limited to common
reference forms. `GemmaQueryRewriter` receives only a bounded recent transcript
and the current question, under instructions that history is for reference
resolution only. It does not receive evidence as facts and does not generate a
policy answer. A resolved standalone query is sent through retrieval again;
the final answer receives only the current question and the newly retrieved
evidence. Prior assistant messages are never included in answer evidence.

If there is no history, the rewriter is unavailable, rewriting fails, or the
rewriter cannot resolve a query, the service returns a structured abstention
without calling the policy answerer. Generation failures remain structured
failures. The evaluation runner accepts `conversation_service=` and seeds each
follow-up test question's reviewed context into a fresh isolated conversation;
the stored original question and rewritten query remain in raw predictions.

## Tests

Run from `backend/`:

```powershell
python -m pytest tests/test_conversation.py -q
```

Tests cover resolved and ambiguous follow-ups, no-history clarification, topic
changes, conversation isolation, clear/delete, unanswerable follow-ups,
bounded history without sources, and the evaluation runner's follow-up path.
All policy strings in these tests are explicitly synthetic fixtures.

## Evaluation and limitations

For reviewed multi-turn questions, pass a configured `ConversationRAGService`
to `EvaluationRunner.run(..., conversation_service=service)`. Review
`human_labels.answer_correct`, `citation_support`, and `evidence_support` using
the normal evaluation schema. Synthetic tests verify plumbing only; they do
not measure follow-up accuracy. No reviewed official multi-turn set or
benchmark output is present, so no quality result is reported.

Follow-up detection uses a small phrase list; implicit or multilingual
references may not trigger rewriting. The rewriter's ability to resolve
ambiguity is not itself a factual evidence decision. Relevance gating and
fresh retrieval still govern the answer path. SQLite stores conversation
content locally; deployments must provide appropriate access and retention
controls.
