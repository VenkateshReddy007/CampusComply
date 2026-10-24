import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import HomePage from "@/app/page";
import { api, responseFromError } from "@/lib/api";
import type { DocumentRecord, KnowledgeGraph } from "@/lib/contracts";

vi.mock("@/lib/api", () => ({
  api: {
    health: vi.fn(),
    listDocuments: vi.fn(),
    createConversation: vi.fn(),
    getConversation: vi.fn(),
    ask: vi.fn(),
    knowledgeGraph: vi.fn(),
  },
  messageFromQuestion: vi.fn(),
  responseFromError: vi.fn(),
}));

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.health).mockResolvedValue({ status: "ok" });
  vi.mocked(api.listDocuments).mockResolvedValue([]);
  vi.mocked(api.createConversation).mockResolvedValue({ conversation_id: "synthetic-chat-id" });
  vi.mocked(api.getConversation).mockRejectedValue(new Error("No saved chat"));
  vi.mocked(api.knowledgeGraph).mockResolvedValue({ nodes: [], edges: [], truncated: false, scope_note: "Source provenance only." });
  vi.mocked(responseFromError).mockReturnValue({
    status: "generation_error", answer: null, citations: [], supporting_passages: [], abstained: false,
    conflicts: [], conflict_check_status: "not_requested", conversation_id: null,
    retrieval_query: null, error: { code: "missing_api_configuration", message: "Set the backend model API key before asking a question." },
  });
});

const indexedFixture: DocumentRecord = {
  id: "synthetic-document-id", filename: "synthetic-policy-fixture.pdf", content_hash: "fixture-hash",
  source_url: null, effective_date: null, document_type: "unknown", upload_status: "stored",
  ingestion_status: "indexed", size_bytes: 100, page_count: 1, chunk_count: 1,
  embedding_model: "fixture-model", embedding_dimension: 2, last_error_code: null,
  redaction_summary: { counts_by_category: {}, counts_by_page: {}, total: 0 },
  redaction_version: 1,
  created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z",
};

describe("CampusReg workspace", () => {
  it("shows an official-document empty state without inventing a policy answer", async () => {
    render(<HomePage />);

    expect(await screen.findByRole("heading", { name: "Start with an official source" })).toBeInTheDocument();
    expect(screen.getByText(/Upload a text-based campus regulation PDF/)).toBeInTheDocument();
    expect(screen.queryByText(/attendance required/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Gemma key needed/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/generation is not configured/i)).not.toBeInTheDocument();
  });

  it("shows a recoverable backend-offline message", async () => {
    vi.mocked(api.health).mockRejectedValueOnce(new Error("API unreachable synthetic test"));

    render(<HomePage />);

    await waitFor(() => expect(screen.getByText("Backend connection unavailable")).toBeInTheDocument());
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });

  it("navigates to the document library", async () => {
    render(<HomePage />);
    await screen.findByRole("heading", { name: "Start with an official source" });

    fireEvent.click(screen.getByRole("button", { name: /Regulations/ }));

    expect(screen.getByRole("heading", { name: "Regulations & documents" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Upload PDFs" })).toBeInTheDocument();
  });

  it("shows a model configuration error only after the user asks", async () => {
    vi.mocked(api.listDocuments).mockResolvedValue([indexedFixture]);
    vi.mocked(api.ask).mockRejectedValue(new Error("Set the backend model API key before asking a question."));
    render(<HomePage />);

    const textbox = await screen.findByRole("textbox", { name: "Ask about your indexed regulations" });
    expect(screen.getByText(/Synthetic fixture files are for software checks only/)).toBeInTheDocument();
    expect(screen.getByText(/Question starters · replace bracketed text/)).toBeInTheDocument();
    expect(screen.queryByText(/Gemma key needed/i)).not.toBeInTheDocument();
    fireEvent.change(textbox, { target: { value: "Which requirements appear in the indexed documents?" } });
    fireEvent.click(screen.getByRole("button", { name: "Ask" }));

    expect(await screen.findByText("missing api configuration")).toBeInTheDocument();
    expect(screen.getByText("Set the backend model API key before asking a question.")).toBeInTheDocument();
  });

  it("loads the source provenance graph from the API", async () => {
    const graph: KnowledgeGraph = {
      nodes: [
        { id: "document:doc", type: "document", label: "policy-fixture.pdf", document_id: "doc", document_name: "policy-fixture.pdf", page: null, chunk_id: null, citation: null, text: null },
        { id: "page:doc:1", type: "page", label: "PDF page 1", document_id: "doc", document_name: "policy-fixture.pdf", page: 1, chunk_id: null, citation: null, text: null },
        { id: "passage:chunk", type: "passage", label: "Passage chunk", document_id: "doc", document_name: "policy-fixture.pdf", page: 1, chunk_id: "chunk", citation: "[policy-fixture.pdf, document doc, p. 1]", text: "Source-backed passage excerpt." },
      ],
      edges: [
        { source: "document:doc", target: "page:doc:1", relation: "contains_page" },
        { source: "page:doc:1", target: "passage:chunk", relation: "contains_passage" },
      ],
      truncated: false,
      scope_note: "Connections show document provenance only.",
    };
    vi.mocked(api.knowledgeGraph).mockResolvedValue(graph);
    render(<HomePage />);
    await screen.findByRole("heading", { name: "Start with an official source" });
    fireEvent.click(screen.getByRole("button", { name: "Knowledge graph" }));

    expect(await screen.findByText("Source-backed passage excerpt.")).toBeInTheDocument();
    expect(screen.getByText(/\[policy-fixture\.pdf, document doc, p\. 1\]/)).toBeInTheDocument();
    expect(screen.getByText("Connections show document provenance only.")).toBeInTheDocument();
  });

  it("shows only redaction categories, counts and source pages", async () => {
    vi.mocked(api.listDocuments).mockResolvedValue([{
      ...indexedFixture,
      redaction_summary: {
        counts_by_category: { EMAIL: 1, STUDENT_ID: 1 },
        counts_by_page: { "2": { EMAIL: 1, STUDENT_ID: 1 } },
        total: 2,
      },
    }]);
    render(<HomePage />);
    await screen.findByRole("heading", { name: "Start with an official source" });
    fireEvent.click(screen.getByRole("button", { name: /Regulations/ }));

    expect(await screen.findByText(/Privacy preprocessing · 2 replacements/)).toBeInTheDocument();
    expect(screen.getByText(/EMAIL: 1 · STUDENT_ID: 1 · PDF page 2/)).toBeInTheDocument();
    expect(screen.queryByText(/Jane Doe|jane@example\.edu|S123456/)).not.toBeInTheDocument();
  });
});
