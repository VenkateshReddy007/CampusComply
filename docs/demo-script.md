# Demo script and readiness

## What can be demonstrated in this checkout

The browser app can run locally at `http://localhost:3000` against the FastAPI
service at `http://127.0.0.1:8000`. Mocked integration tests exercise normal
answer/citation rendering, abstention, multiple PDFs, conversation isolation,
source inspection, conflict comparison, upload validation, and generation
failure handling.

To exercise the upload and source library using only fake content, run:

```powershell
python scripts/create_synthetic_demo_pdfs.py
```

Upload `data/synthetic_demo/synthetic-alpha.pdf` and
`synthetic-beta.pdf`. Both documents are marked as synthetic and contain
invented example values. Never present these as campus rules. Without a
backend model key, the UI correctly reports that answer generation and
model-assisted conflict comparison need configuration; test coverage for
those paths uses injected test doubles.

## Official-data showcase requested by Problem Statement 2A

The requested live showcase is **not ready** because this workspace has no
verified institutional PDFs, reviewed expected answers, or user-provided
Google AI Studio API key. Do not claim that synthetic content satisfies the
official-data deliverable. Once those inputs are provided:

1. Start the API and frontend using `README.md`; verify readiness and API CORS.
2. Upload at least two verified current official regulations and record their
   provenance, effective dates, and version metadata.
3. Ask a reviewed direct factual question, open each citation, and compare the
   displayed passage to the original PDF page.
4. Ask a reviewed multi-passage / multi-document question and inspect all
   retrieved passages actually sent to Gemma.
5. Ask a question absent from the PDFs; verify structured abstention without
   an unsupported answer.
6. Ask a follow-up; verify its rewritten retrieval query and fresh source
   passages while keeping it in the same conversation.
7. Inspect a potential conflict; show both original passages and state that
   a human must determine scope, supersession, and authority.
8. Show only measured experiment output produced from the reviewed dataset.
   Do not create or narrate benchmark figures in advance.

Manually check source name, physical page number, and claim support on the
held-out test set before presenting any percentage answered correctly.
