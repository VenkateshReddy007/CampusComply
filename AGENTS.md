PROJECT: CampusReg — Campus Regulations Assistant, problem statement 2A.

MISSION:
Complete all required deliverables and all applicable bonus features
from the Hacktoberfest RAG Application with Gemma problem statement.

NON-NEGOTIABLE RULES:

1. Reuse the working Google Colab RAG baseline. Do not rewrite
   functioning stages without a documented reason.

2. Maintain one shared Python RAG implementation. Colab evaluation
   and the FastAPI application must import the same core modules.

3. Never invent campus regulations, attendance requirements,
   deadlines, hostel rules, source documents, citations or test results.

4. Only use actual uploaded or verified official source documents
   for factual campus-policy answers. Clearly label synthetic data
   as synthetic and use it only for software tests.

5. Every substantive answer must cite document name and PDF page,
   and expose the actual retrieved evidence.

6. Implement abstention when evidence is inadequate. Similarity scores
   are not calibrated probabilities of factual correctness.

7. Conflict detection must show both supporting passages and must
   not arbitrarily decide which regulation is authoritative.

8. Keep API keys server-side. Never print, commit, hardcode, or return
   secrets in source code, test fixtures, logs, notebook outputs,
   browser responses, screenshots, or documentation.

9. No fake benchmark metrics, hardcoded production answers, mock
   features represented as finished functionality, or fabricated data.

10. Make changes phase by phase. Read existing files before modifying
    them. Do not replace working implementations without justification.

11. Write automated tests for every feature. Run relevant tests after
    each phase, fix failures, and report the actual results.

12. Do not launch expensive model evaluations or consume API quota
    without an explicit request. Use mocked model calls in unit tests.

13. Keep clear documentation of architecture, assumptions, model
    settings, benchmark methodology, limitations and failure cases.

14. End every phase with:
    - files changed;
    - implementation summary;
    - tests and commands actually run;
    - actual test results;
    - known limitations;
    - next phase readiness.
