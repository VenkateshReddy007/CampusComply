# CampusReg privacy processing: scope and limits

## Current data path

PDF bytes are parsed locally with the existing `pypdf` ingestion path. A deterministic local redaction pass runs on extracted page text before chunking, vectorization, SQLite page/chunk persistence, retrieval, or passage prompts. The same redaction utility runs on user questions before embedding or generation and on bounded chat history before query rewriting. No identity-to-placeholder map is created. Summaries contain only category counts and physical PDF page numbers.

The original PDF is retained in the existing SQLite `original_pdf` BLOB for explicit re-indexing. It is not sent to the generation provider. SQLite files and uploaded data must remain on a protected local/backend volume; the current store does not encrypt the BLOB or database itself. Existing indexes are marked with a redaction version. Retrieval, evidence and graph endpoints refuse to serve indexes from before this version until they are explicitly re-indexed.

## Detector coverage

The current local rules target common, explicitly labelled student/employee IDs, labelled dates of birth, email addresses, common India/US phone formats, labelled addresses/account IDs, labelled title-cased names and labelled signatures. Rule labels and values are replaced together in some categories; person labels are retained. Repeated matched values use a stable category token within one document or request.

This implementation is not a general PII recognizer. It can miss unlabelled people, atypical names, international phone formats, free-form addresses, identifiers without labels, sensitive content embedded in images, handwriting, PDF metadata, and categories not configured in its rules. OCR is not implemented. A clean count of zero means only that these detectors did not match; it is not proof that the document contains no sensitive information. False positives can also remove relevant text. Human review is required for high-risk source material.

## External processing boundary

The configured Gemma provider receives the sanitized question and sanitized retrieved passages. Conversation rewriting receives sanitized bounded history. Local SentenceTransformer inference receives sanitized chunk/query text; model weights may be downloaded separately, but document text is not sent to that model host by this code. Provider failures are normalized without including prompt text or credentials in the returned error.

## Test coverage and residual risks

Synthetic automated tests cover labelled names, IDs, email and phone values; retention of policy section/percentage/date text; placeholder consistency; page-number preservation; sanitized embedding and generation inputs; safe summaries; and blocking/re-indexing of old indexes. These tests do not establish zero leakage or production-grade detector recall. No real institutional PDF was used for privacy evaluation in this change.

The uploaded original remains available to the backend process and local database administrator. File-system permissions, backups, retention and secure deletion remain deployment responsibilities. Do not upload documents unless authorized to process them.
