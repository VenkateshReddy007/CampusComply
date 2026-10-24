import type { ConflictReport, ConversationMessage, DocumentRecord, EvaluationJob, KnowledgeGraph, QuestionResponse, UploadItem } from "@/lib/contracts";

const API_URL = (process.env.NEXT_PUBLIC_CAMPUSREG_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(message: string, public readonly status: number, public readonly payload: unknown) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, { ...init, cache: "no-store" });
  } catch {
    throw new ApiError("CampusReg API is unreachable. Start the backend and retry.", 0, null);
  }
  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const payload = body as { detail?: { message?: string }; error?: { message?: string } } | null;
    const message = payload?.detail?.message || payload?.error?.message || `Request failed (${response.status}).`;
    throw new ApiError(message, response.status, body);
  }
  return body as T;
}

export const api = {
  health: () => request<{ status: string }>("/api/health"),
  readiness: () => request<{ status: string; model_api_key_configured: boolean; embedding_dependency_available: boolean; database_configured: boolean }>("/api/ready"),
  listDocuments: () => request<DocumentRecord[]>("/api/documents"),
  uploadDocuments: (files: File[]) => {
    const body = new FormData();
    files.forEach((file) => body.append("files", file));
    return request<{ items: UploadItem[] }>("/api/documents", { method: "POST", body });
  },
  deleteDocument: (id: string) => request<{ deleted: boolean }>(`/api/documents/${encodeURIComponent(id)}`, { method: "DELETE" }),
  reindexDocument: (id: string) => request<DocumentRecord>(`/api/documents/${encodeURIComponent(id)}/reindex`, { method: "POST" }),
  createConversation: () => request<{ conversation_id: string }>("/api/conversations", { method: "POST" }),
  getConversation: (id: string) => request<{ conversation_id: string; messages: ConversationMessage[] }>(`/api/conversations/${encodeURIComponent(id)}`),
  clearConversation: (id: string) => request<{ cleared: boolean }>(`/api/conversations/${encodeURIComponent(id)}/messages`, { method: "DELETE" }),
  ask: (conversationId: string, question: string) => request<QuestionResponse>(`/api/conversations/${encodeURIComponent(conversationId)}/questions`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question }),
  }),
  conflicts: () => request<{ reports: ConflictReport[]; human_review_required: boolean; authority_decided: boolean }>("/api/conflicts"),
  knowledgeGraph: () => request<KnowledgeGraph>("/api/knowledge-graph"),
  startEvaluation: (split: string, token: string) => request<EvaluationJob>("/api/development/evaluations", {
    method: "POST", headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
    body: JSON.stringify({ split, force: false }),
  }),
  getEvaluation: (id: string, token: string) => request<EvaluationJob>(`/api/development/evaluations/${encodeURIComponent(id)}`, {
    headers: { Authorization: `Bearer ${token}` },
  }),
};

export function messageFromQuestion(question: string, response: QuestionResponse): ConversationMessage {
  return {
    id: Date.now(), conversation_id: response.conversation_id || "", role: "assistant",
    content: response.answer || "", created_at: new Date().toISOString(),
    original_question: question, retrieval_query: response.retrieval_query,
    answer_status: response.status, sources: response.supporting_passages,
  };
}

export function responseFromError(error: unknown): QuestionResponse {
  if (error instanceof ApiError && error.payload && typeof error.payload === "object") {
    const body = error.payload as Partial<QuestionResponse>;
    if (typeof body.status === "string") {
      return {
        status: body.status as QuestionResponse["status"], answer: body.answer ?? null,
        citations: body.citations ?? [], supporting_passages: body.supporting_passages ?? [],
        abstained: body.abstained ?? false, conflicts: body.conflicts ?? [],
        conflict_check_status: body.conflict_check_status ?? "not_requested",
        conversation_id: body.conversation_id ?? null, retrieval_query: body.retrieval_query ?? null,
        error: body.error ?? { code: "request_failed", message: error.message },
      };
    }
  }
  return {
    status: "generation_error", answer: null, citations: [], supporting_passages: [],
    abstained: false, conflicts: [], conflict_check_status: "not_requested",
    conversation_id: null, retrieval_query: null,
    error: { code: error instanceof ApiError ? `http_${error.status}` : "request_failed", message: error instanceof Error ? error.message : "The request could not be completed." },
  };
}
