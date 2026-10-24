# Phase 3: Multi-document ingestion and citations

## Implementation

`MultiDocumentRAGEngine` composes the shared chunking, embedding, retrieval and
generation modules with `PDFIngestor` and `SQLiteDocumentStore`. No new RAG
algorithm or policy content was added. The SQLite database stores document
metadata, original uploaded PDF bytes, every extracted page's text and physical
one-based page number, and each chunk with its document ID, filename, chunk ID,
page, original text and normalized float32 vector.

Document upload metadata includes a UUID, original filename, SHA-256 content
hash, optional source URL, optional ISO effective date, document type, upload
status, ingestion status, file size, page/chunk counts and embedding settings.
Duplicate content hashes return the existing record. The default upload limit
is 20 MiB and is configurable with `MAX_PDF_SIZE_BYTES`. URLs are metadata only;
the application does not fetch them or verify that they are official.

New indexing stores a failed status if parsing, text extraction, chunking or
embedding fails. Re-indexing rebuilds one document from its stored PDF and
replaces its pages/chunks/vectors in one SQLite transaction. The old indexed
rows remain queryable while re-indexing and are kept if replacement fails.
Document deletion uses cascading foreign keys inside a transaction.

Retrieval loads all currently indexed chunks into one vector matrix while
preserving per-row source metadata. All indexed records must use the configured
embedding model and share a vector dimension. A mismatch raises an explicit
compatibility error; re-index affected documents with one model before asking
questions. Answer prompts and returned evidence share the same source passage
strings. Multi-document citations contain both filename and document UUID plus
the physical page number.

## PDF handling and limitations

- Only `.pdf` filenames with a PDF signature are accepted.
- Encrypted and malformed PDFs are rejected with typed ingestion errors.
- PDFs with no extracted text are rejected. OCR is not implemented. PDFs with
  some blank pages retain those pages in storage but produce no chunks for them.
  Textless pages in mixed scanned/text PDFs are therefore not covered by the
  index and must not be treated as having been read.
- Source URLs and effective dates are provided by the uploader; they are not
  independently verified.
- The database contains original source files and should remain local and
  ignored. Back it up securely if the deployment requires durable documents.
- Citation validation checks that an answer contains a retrieved citation. It
  cannot prove that every claim in generated prose is supported.

## Local use

From the repository root, install with `python -m pip install -e ".\backend[all]"`
or install the package/test extras as described in `backend/README.md`. Store the
database at an ignored path such as `data/campusreg.sqlite3`. The same package
and engine can be imported in Colab after installing `backend[notebook]` from
the checkout.

Synthetic PDF fixtures in `backend/tests/test_multidocument.py` are generated
inside tests and labelled synthetic. They are not campus regulations and are
not included in production data.
