import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import HomePage from "@/app/page";
import { api } from "@/lib/api";

vi.mock("@/lib/api", () => ({
  api: {
    health: vi.fn(),
    readiness: vi.fn(),
    listDocuments: vi.fn(),
    createConversation: vi.fn(),
    getConversation: vi.fn(),
  },
  messageFromQuestion: vi.fn(),
  responseFromError: vi.fn(),
}));

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.health).mockResolvedValue({ status: "ok" });
  vi.mocked(api.readiness).mockResolvedValue({ status: "degraded", model_api_key_configured: false, embedding_dependency_available: true, database_configured: true });
  vi.mocked(api.listDocuments).mockResolvedValue([]);
  vi.mocked(api.createConversation).mockResolvedValue({ conversation_id: "synthetic-chat-id" });
  vi.mocked(api.getConversation).mockRejectedValue(new Error("No saved chat"));
});

describe("CampusReg workspace", () => {
  it("shows an official-document empty state without inventing a policy answer", async () => {
    render(<HomePage />);

    expect(await screen.findByRole("heading", { name: "Start with an official source" })).toBeInTheDocument();
    expect(screen.getByText(/Upload a text-based campus regulation PDF/)).toBeInTheDocument();
    expect(screen.queryByText(/attendance required/i)).not.toBeInTheDocument();
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
});
