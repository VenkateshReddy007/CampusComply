export type Source = {
  document_id: string;
  document_name: string;
  page: number;
  chunk_id: string;
  citation: string;
  text: string;
  retrieval_method: string;
  score: number;
  diagnostics: Record<string, unknown>;
};

export type QuestionResponse = {
  status: "answered" | "abstained" | "generation_error" | "invalid_response";
  answer: string | null;
  citations: string[];
  supporting_passages: Source[];
  abstained: boolean;
  conflicts: Record<string, unknown>[];
  conflict_check_status: string;
  conversation_id: string | null;
  retrieval_query: string | null;
  error: { code: string; message: string } | null;
};

export type DocumentRecord = {
  id: string;
  filename: string;
  content_hash: string;
  source_url: string | null;
  effective_date: string | null;
  document_type: string;
  upload_status: string;
  ingestion_status: string;
  size_bytes: number;
  page_count: number;
  chunk_count: number;
  embedding_model: string | null;
  embedding_dimension: number | null;
  last_error_code: string | null;
  created_at: string;
  updated_at: string;
};

export type UploadItem = {
  filename: string;
  status: string;
  duplicate: boolean;
  document: DocumentRecord | null;
  error: { code: string; message: string } | null;
};

export type ConversationMessage = {
  id: number;
  conversation_id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
  original_question: string | null;
  retrieval_query: string | null;
  answer_status: string | null;
  sources: Source[];
};

export type ConflictReport = {
  subject: string;
  status: string;
  passage_a: { document_id: string; document_name: string; page: number; chunk_id: string; citation: string; text: string };
  passage_b: { document_id: string; document_name: string; page: number; chunk_id: string; citation: string; text: string };
  explanation: string;
  scope: { passage_a: string | null; passage_b: string | null };
  effective_dates: { passage_a: string | null; passage_b: string | null };
  limitations: string[];
  human_review_required: boolean;
};

export type EvaluationJob = {
  id: string;
  status: string;
  split: string;
  result: Record<string, unknown> | null;
  error: { code: string; message: string } | null;
};
