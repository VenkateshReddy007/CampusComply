import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, api, responseFromError } from "@/lib/api";

afterEach(() => vi.unstubAllGlobals());

describe("CampusReg API client", () => {
  it("sends questions to the conversation endpoint and preserves returned evidence", async () => {
    const payload = {
      status: "answered", answer: "SYNTHETIC TEST answer", citations: ["[fixture.pdf, document d1, p. 1]"],
      supporting_passages: [{ document_id: "d1", document_name: "fixture.pdf", page: 1, chunk_id: "c1", citation: "[fixture.pdf, document d1, p. 1]", text: "SYNTHETIC TEST evidence", retrieval_method: "dense", score: 0.7, diagnostics: {} }],
      abstained: false, conflicts: [], conflict_check_status: "available_on_conflict_endpoint", conversation_id: "chat1", retrieval_query: "synthetic question", error: null,
    };
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(payload), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    const response = await api.ask("chat1", "SYNTHETIC TEST question");

    expect(fetchMock.mock.calls[0][0]).toContain("/api/conversations/chat1/questions");
    expect(response.supporting_passages[0].text).toBe("SYNTHETIC TEST evidence");
    expect(response.citations).toHaveLength(1);
  });

  it("does not convert failed requests into normal answers", async () => {
    const failure = new ApiError("provider unavailable", 503, {
      status: "generation_error", answer: null, citations: [], supporting_passages: [],
      abstained: false, conflicts: [], conflict_check_status: "not_requested", conversation_id: null,
      retrieval_query: null, error: { code: "generation_failed", message: "provider unavailable" },
    });

    const response = responseFromError(failure);

    expect(response.status).toBe("generation_error");
    expect(response.answer).toBeNull();
    expect(response.citations).toEqual([]);
  });
});
