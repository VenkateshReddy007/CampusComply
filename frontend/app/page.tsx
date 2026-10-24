"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertCircle, ArrowRight, BookOpenText, Check, ChevronDown,
  CircleHelp, Clock3, FileText, FolderOpen, Gauge, LoaderCircle, MessageSquareText,
  Network, Plus, RefreshCw, Search, Send, ShieldAlert, ShieldCheck, SlidersHorizontal,
  Sparkles, Trash2, Upload, X,
} from "lucide-react";
import { api, messageFromQuestion, responseFromError } from "@/lib/api";
import type { ConflictReport, ConversationMessage, DocumentRecord, EvaluationJob, KnowledgeGraph, Source, UploadItem } from "@/lib/contracts";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";

type View = "chats" | "documents" | "knowledge" | "evaluation";
type UIMessage = {
  key: string;
  role: "user" | "assistant";
  content: string;
  status?: string | null;
  error?: { code: string; message: string } | null;
  sources?: Source[];
  citations?: string[];
  originalQuestion?: string;
  pending?: boolean;
};

const CONVERSATION_STORAGE_KEY = "campusreg.conversation.v1";

const NAV_ITEMS: { id: View; label: string; icon: typeof MessageSquareText }[] = [
  { id: "chats", label: "Chats", icon: MessageSquareText },
  { id: "documents", label: "Regulations", icon: BookOpenText },
  { id: "knowledge", label: "Knowledge graph", icon: Network },
  { id: "evaluation", label: "Evaluation", icon: Gauge },
];

const QUESTION_STARTERS = [
  "Which requirements are stated in the indexed documents?",
  "Which page states the deadline for [topic]?",
  "What exceptions are listed for [requirement]?",
  "Do the indexed documents give the same rule for [group]?",
];

function isSyntheticFixture(document: DocumentRecord) {
  return document.filename.toLowerCase().includes("synthetic") || document.document_type.toLowerCase() === "synthetic";
}

function sourceFromStored(value: unknown): Source[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is Source => Boolean(item && typeof item === "object" && "document_id" in item && "text" in item));
}

function toUIMessage(message: ConversationMessage): UIMessage {
  return {
    key: `server-${message.id}`,
    role: message.role,
    content: message.content,
    status: message.answer_status,
    sources: sourceFromStored(message.sources),
  };
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function citedText(text: string, sources: Source[], onSelect: (source: Source) => void) {
  const labels = [...sources].sort((a, b) => b.citation.length - a.citation.length);
  if (!labels.length) return text;
  const escaped = labels.map((source) => source.citation.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|");
  const parts = text.split(new RegExp(`(${escaped})`, "g"));
  return parts.map((part, index) => {
    const match = labels.find((source) => source.citation === part);
    return match ? (
      <button className="citation-link" key={`${part}-${index}`} onClick={() => onSelect(match)} type="button">{part}</button>
    ) : <span key={`${index}-${part.slice(0, 16)}`}>{part}</span>;
  });
}

export default function HomePage() {
  const [view, setView] = useState<View>("chats");
  const [apiStatus, setApiStatus] = useState<"checking" | "online" | "offline">("checking");
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [conversationId, setConversationId] = useState("");
  const [messages, setMessages] = useState<UIMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [pageError, setPageError] = useState("");
  const [uploadState, setUploadState] = useState("");
  const [uploadResults, setUploadResults] = useState<UploadItem[]>([]);
  const [documentFilter, setDocumentFilter] = useState("");
  const [selectedSource, setSelectedSource] = useState<Source | null>(null);
  const [conflicts, setConflicts] = useState<ConflictReport[]>([]);
  const [conflictLoading, setConflictLoading] = useState(false);
  const [evaluationToken, setEvaluationToken] = useState("");
  const [evaluationSplit, setEvaluationSplit] = useState("validation");
  const [evaluationJob, setEvaluationJob] = useState<EvaluationJob | null>(null);
  const [evaluationError, setEvaluationError] = useState("");
  const [knowledgeGraph, setKnowledgeGraph] = useState<KnowledgeGraph | null>(null);
  const [knowledgeGraphError, setKnowledgeGraphError] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);
  const messageEndRef = useRef<HTMLDivElement>(null);

  const refreshDocuments = useCallback(async () => {
    const records = await api.listDocuments();
    setDocuments(records);
  }, []);

  const startNewConversation = useCallback(async () => {
    setPageError("");
    try {
      const created = await api.createConversation();
      setConversationId(created.conversation_id);
      setMessages([]);
      window.localStorage.setItem(CONVERSATION_STORAGE_KEY, created.conversation_id);
    } catch (error) {
      setPageError(error instanceof Error ? error.message : "A new conversation could not be created.");
    }
  }, []);

  useEffect(() => {
    let active = true;
    async function initialize() {
      try {
        await api.health();
        if (!active) return;
        setApiStatus("online");
        const existing = window.localStorage.getItem(CONVERSATION_STORAGE_KEY);
        const [records, existingHistory] = await Promise.all([
          api.listDocuments(),
          existing ? api.getConversation(existing).catch(() => null) : Promise.resolve(null),
        ]);
        if (!active) return;
        setDocuments(records);
        if (existing && existingHistory) {
          setConversationId(existing);
          setMessages(existingHistory.messages.map(toUIMessage));
          return;
        }
        window.localStorage.removeItem(CONVERSATION_STORAGE_KEY);
        const created = await api.createConversation();
        if (!active) return;
        setConversationId(created.conversation_id);
        window.localStorage.setItem(CONVERSATION_STORAGE_KEY, created.conversation_id);
      } catch (error) {
        if (!active) return;
        setApiStatus("offline");
        setPageError(error instanceof Error ? error.message : "Could not connect to the API.");
      }
    }
    void initialize();
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (view !== "knowledge") return;
    let active = true;
    void api.knowledgeGraph().then((graph) => {
      if (active) setKnowledgeGraph(graph);
    }).catch((error: unknown) => {
      if (active) setKnowledgeGraphError(error instanceof Error ? error.message : "The source graph could not be loaded.");
    });
    return () => { active = false; };
  }, [view]);

  useEffect(() => {
    const behavior = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
    messageEndRef.current?.scrollIntoView({ behavior, block: "end" });
  }, [messages]);

  const visibleDocuments = useMemo(() => {
    const query = documentFilter.trim().toLowerCase();
    return documents.filter((record) => !query || `${record.filename} ${record.document_type} ${record.ingestion_status}`.toLowerCase().includes(query));
  }, [documents, documentFilter]);

  async function askQuestion(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const text = question.trim();
    if (!text || busy || !conversationId) return;
    setQuestion("");
    setPageError("");
    setBusy(true);
    const pendingKey = `pending-${Date.now()}`;
    setMessages((current) => [
      ...current,
      { key: `user-${pendingKey}`, role: "user", content: text },
      { key: pendingKey, role: "assistant", content: "", pending: true },
    ]);
    try {
      const response = await api.ask(conversationId, text);
      const assistant = messageFromQuestion(text, response);
      setMessages((current) => current.map((message) => message.key === pendingKey ? {
        key: `assistant-${assistant.created_at}`, role: "assistant", content: assistant.content,
        status: response.status, error: response.error, sources: response.supporting_passages,
        citations: response.citations,
      } : message));
      setSelectedSource(response.supporting_passages[0] || null);
    } catch (error) {
      const response = responseFromError(error);
      setMessages((current) => current.map((message) => message.key === pendingKey ? {
        key: `error-${Date.now()}`, role: "assistant", content: "", status: response.status,
        error: response.error, sources: response.supporting_passages, citations: [], originalQuestion: text,
      } : message));
    } finally {
      setBusy(false);
    }
  }

  async function handleUpload(files: FileList | null) {
    if (!files?.length) return;
    setUploadState(`Uploading and indexing ${files.length} PDF${files.length === 1 ? "" : "s"}…`);
    setUploadResults([]);
    setPageError("");
    try {
      const result = await api.uploadDocuments(Array.from(files));
      setUploadResults(result.items);
      await refreshDocuments();
      const failed = result.items.filter((item) => item.status === "failed").length;
      setUploadState(failed ? `${failed} file${failed === 1 ? "" : "s"} need attention.` : "Indexing finished.");
    } catch (error) {
      setUploadState("");
      setPageError(error instanceof Error ? error.message : "The upload could not be completed.");
    } finally {
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  async function deleteDocument(record: DocumentRecord) {
    if (!window.confirm(`Remove ${record.filename} and its local index?`)) return;
    try {
      await api.deleteDocument(record.id);
      setDocuments((current) => current.filter((document) => document.id !== record.id));
    } catch (error) {
      setPageError(error instanceof Error ? error.message : "Document could not be removed.");
    }
  }

  async function reindexDocument(record: DocumentRecord) {
    try {
      await api.reindexDocument(record.id);
      await refreshDocuments();
    } catch (error) {
      setPageError(error instanceof Error ? error.message : "Document could not be re-indexed.");
    }
  }

  async function clearChat() {
    if (!conversationId) return;
    try {
      await api.clearConversation(conversationId);
      setMessages([]);
      setSelectedSource(null);
    } catch (error) {
      setPageError(error instanceof Error ? error.message : "Conversation could not be cleared.");
    }
  }

  async function inspectConflicts() {
    setConflictLoading(true);
    setEvaluationError("");
    try {
      const result = await api.conflicts();
      setConflicts(result.reports);
    } catch (error) {
      setEvaluationError(error instanceof Error ? error.message : "Conflict comparison could not be completed.");
    } finally {
      setConflictLoading(false);
    }
  }

  async function startEvaluation() {
    setEvaluationError("");
    if (!evaluationToken.trim()) {
      setEvaluationError("Enter the development evaluation bearer token configured on the backend.");
      return;
    }
    try {
      const started = await api.startEvaluation(evaluationSplit, evaluationToken.trim());
      setEvaluationJob(started);
      const poll = window.setInterval(async () => {
        try {
          const next = await api.getEvaluation(started.id, evaluationToken.trim());
          setEvaluationJob(next);
          if (next.status === "completed" || next.status === "failed") window.clearInterval(poll);
        } catch (error) {
          window.clearInterval(poll);
          setEvaluationError(error instanceof Error ? error.message : "Could not read the evaluation job.");
        }
      }, 1500);
      window.setTimeout(() => window.clearInterval(poll), 120_000);
    } catch (error) {
      setEvaluationError(error instanceof Error ? error.message : "The evaluation job could not be started.");
    }
  }

  return (
    <main className="app-shell">
      <aside className="sidebar" aria-label="Main navigation">
        <div className="brand-lockup">
          <div className="brand-mark"><BookOpenText size={19} strokeWidth={1.8} /></div>
          <div><strong>CampusReg</strong><span>CampusComply</span></div>
        </div>
        <div className="workspace-label">WORKSPACE</div>
        <nav className="nav-list">
          {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
            <button key={id} type="button" onClick={() => { if (id === "knowledge") { setKnowledgeGraph(null); setKnowledgeGraphError(""); } setView(id); }} className={`nav-item ${view === id ? "active" : ""}`} aria-label={label} aria-current={view === id ? "page" : undefined}>
              <Icon size={17} strokeWidth={1.8} /><span>{label}</span>
              {id === "documents" && <span className="nav-count">{documents.length}</span>}
            </button>
          ))}
        </nav>
        <div className="sidebar-spacer" />
        <div className="sidebar-note">
          <ShieldCheck size={16} />
          <p>Answers stay tied to the passages indexed in this workspace.</p>
        </div>
        <div className="profile-row">
          <div className="avatar">CR</div>
          <div className="profile-copy"><strong>Campus workspace</strong><span>Local session</span></div>
          <ChevronDown size={15} className="muted-icon" />
        </div>
      </aside>

      <section className="main-column">
        <header className="topbar">
          <div className="breadcrumb"><span>CampusComply</span><ArrowRight size={13} /><strong>{NAV_ITEMS.find((item) => item.id === view)?.label}</strong></div>
          <div className="topbar-actions">
            <span className={`connection-pill ${apiStatus}`}><span className="connection-dot" />{apiStatus === "online" ? "API online" : apiStatus === "offline" ? "API offline" : "Connecting"}</span>
            {view === "chats" && <Button variant="secondary" size="sm" onClick={() => void startNewConversation()} disabled={apiStatus !== "online"}><Plus size={15} />New chat</Button>}
          </div>
        </header>

        <div className="content-scroll">
          {pageError && <div className="notice error-notice" role="alert"><AlertCircle size={17} /><span>{pageError}</span><button type="button" aria-label="Dismiss" onClick={() => setPageError("")}><X size={15} /></button></div>}
          {apiStatus === "offline" && <div className="offline-banner"><CircleHelp size={18} /><div><strong>Backend connection unavailable</strong><p>{pageError || "Start the FastAPI service, then retry."}</p></div><Button variant="secondary" size="sm" onClick={() => window.location.reload()}><RefreshCw size={14} />Retry</Button></div>}
          <input ref={fileRef} className="sr-only" type="file" accept="application/pdf,.pdf" multiple onChange={(event) => void handleUpload(event.target.files)} />

          {view === "chats" && (
            <div className="chat-view">
              <div className="page-heading chat-heading">
                <div><div className="eyebrow"><Sparkles size={14} /> EVIDENCE-FIRST ASSISTANT</div><h1>Ask the regulations.</h1><p>Get a clear answer grounded in your indexed documents, with the source passage beside it.</p></div>
                <Button variant="ghost" size="sm" onClick={() => void clearChat()} disabled={!messages.length}><Trash2 size={14} />Clear chat</Button>
              </div>
              {documents.some(isSyntheticFixture) && <div className="fixture-notice" role="note"><ShieldAlert size={16} /><span>Synthetic fixture files are for software checks only. They are not institutional policy and must not be used for compliance decisions.</span></div>}
              <div className="chat-layout">
                <div className="chat-main">
                  {documents.length === 0 && !messages.length ? (
                    <Card className="empty-state">
                      <div className="empty-icon"><FolderOpen size={23} /></div>
                      <h2>Start with an official source</h2>
                      <p>Upload a text-based campus regulation PDF. Answers use only indexed passages and include their original PDF page.</p>
                      <Button onClick={() => { setView("documents"); fileRef.current?.click(); }}><Upload size={15} />Add documents</Button>
                    </Card>
                  ) : (
                    <div className="message-list" aria-live="polite">
                      {messages.map((message) => (
                        <article className={`message-row ${message.role}`} key={message.key}>
                          <div className={`message-avatar ${message.role}`}>{message.role === "assistant" ? <BookOpenText size={15} /> : "Y"}</div>
                          <div className="message-body">
                            <div className="message-meta"><strong>{message.role === "assistant" ? "CampusReg" : "You"}</strong>{message.status && <Badge className={`status-badge ${message.status}`}>{message.status.replaceAll("_", " ")}</Badge>}</div>
                            {message.pending ? <div className="thinking"><LoaderCircle size={16} className="spin" />Retrieving passages and preparing a cited response…</div> : message.error ? (
                              <div className="message-error" role="alert"><AlertCircle size={15} /><div><strong>{message.error.code.replaceAll("_", " ")}</strong><p>{message.error.message}</p><Button variant="secondary" size="sm" onClick={() => setQuestion(message.originalQuestion || "")}>Review request</Button></div></div>
                            ) : message.status === "abstained" ? (
                              <div className="refusal-box"><ShieldAlert size={17} /><p>{message.content || "The indexed documents do not establish an answer to that question."}</p></div>
                            ) : message.role === "assistant" ? (
                              <>
                                <p className="answer-copy">{citedText(message.content, message.sources || [], setSelectedSource)}</p>
                                {!!message.sources?.length && <div className="message-source-strip"><span>{message.sources.length} supporting passage{message.sources.length === 1 ? "" : "s"}</span>{message.sources.slice(0, 3).map((source) => <button type="button" key={source.chunk_id} onClick={() => setSelectedSource(source)}>{source.document_name} · p. {source.page}</button>)}</div>}
                              </>
                            ) : <p className="user-copy">{message.content}</p>}
                          </div>
                        </article>
                      ))}
                      <div ref={messageEndRef} />
                    </div>
                  )}

                  <form className="composer" onSubmit={askQuestion}>
                    <label htmlFor="question" className="sr-only">Ask about your indexed regulations</label>
                    {!messages.length && documents.length > 0 && <div className="question-starters" aria-label="Question starters"><span>Question starters · replace bracketed text with a term from your source</span><div>{QUESTION_STARTERS.map((starter) => <button key={starter} type="button" onClick={() => setQuestion(starter)}>{starter}</button>)}</div></div>}
                    <Textarea id="question" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder={documents.length ? "Ask about a rule, deadline, exception, or requirement…" : "Add an official document to begin…"} maxLength={4000} disabled={!documents.length || busy || apiStatus !== "online"} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); event.currentTarget.form?.requestSubmit(); } }} />
                    <div className="composer-footer"><span><ShieldCheck size={13} />Answers cite the current indexed evidence</span><Button size="sm" type="submit" disabled={!question.trim() || !documents.length || busy || !conversationId}><Send size={14} />Ask</Button></div>
                  </form>
                  <div className="composer-hint">Enter to send · Shift + Enter for a new line</div>
                </div>

                <aside className="evidence-sidebar" aria-label="Retrieved evidence">
                  <div className="evidence-header"><div><div className="eyebrow small">SOURCE PANEL</div><h2>Evidence</h2></div><Button variant="ghost" size="icon" aria-label="Check document conflicts" title="Check cross-document conflicts" onClick={() => void inspectConflicts()} disabled={conflictLoading || documents.length < 2}>{conflictLoading ? <LoaderCircle className="spin" size={16} /> : <SlidersHorizontal size={16} />}</Button></div>
                  {selectedSource ? (
                    <div className="evidence-content">
                      <div className="source-meta"><span className="source-file-icon"><FileText size={15} /></span><div><strong>{selectedSource.document_name}</strong><span>PDF page {selectedSource.page}</span></div></div>
                      <Badge className="source-id">{selectedSource.document_id.slice(0, 14)}</Badge>
                      <blockquote>{selectedSource.text}</blockquote>
                      <div className="evidence-details"><span>Passage ID</span><code>{selectedSource.chunk_id}</code><span>Retrieval method</span><code>{selectedSource.retrieval_method}</code></div>
                      <p className="evidence-note">This is the indexed passage supplied to the answer generator.</p>
                    </div>
                  ) : (
                    <div className="evidence-empty"><FileText size={20} /><p>Retrieved passages appear here with their original PDF page.</p></div>
                  )}
                  <div className="evidence-footer"><span><span className="teal-dot" />{documents.length} indexed source{documents.length === 1 ? "" : "s"}</span><button type="button" onClick={() => setView("documents")}>Manage <ArrowRight size={13} /></button></div>
                </aside>
              </div>
              {!!conflicts.length && <ConflictPanel reports={conflicts} />}
              {evaluationError && view === "chats" && <div className="notice error-notice" role="alert"><AlertCircle size={16} /><span>{evaluationError}</span></div>}
            </div>
          )}

          {view === "documents" && (
            <div className="document-view">
              <div className="page-heading"><div><div className="eyebrow"><BookOpenText size={14} /> SOURCE LIBRARY</div><h1>Regulations &amp; documents</h1><p>Manage PDFs indexed for this workspace. Use official documents for factual policy answers.</p></div><Button onClick={() => fileRef.current?.click()} disabled={apiStatus !== "online"}><Upload size={15} />Upload PDFs</Button></div>
              <div className="upload-drop" onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); void handleUpload(event.dataTransfer.files); }}>
                <div className="upload-icon"><Upload size={17} /></div><div><strong>Drop PDF files here</strong><span>Text-based PDFs, up to 20 MB each. Scanned PDFs need OCR and are not indexed.</span></div><Button variant="secondary" size="sm" onClick={() => fileRef.current?.click()}>Browse files</Button>
              </div>
              {uploadState && <div className="upload-status" aria-live="polite">{uploadState.includes("…") ? <LoaderCircle size={15} className="spin" /> : <Check size={15} />}{uploadState}</div>}
              {!!uploadResults.length && <div className="upload-results">{uploadResults.map((item, index) => <div className={`upload-result ${item.status}`} key={`${item.filename}-${index}`}><span>{item.status === "indexed" ? <Check size={15} /> : <AlertCircle size={15} />}</span><div><strong>{item.filename}</strong><p>{item.error?.message || (item.duplicate ? "Duplicate content detected; existing index reused." : `${item.document?.page_count ?? 0} pages indexed`)}</p>{item.document && <RedactionSummaryLabel document={item.document} />}</div></div>)}</div>}
              <div className="list-heading"><div><h2>Indexed sources</h2><span>{documents.length} document{documents.length === 1 ? "" : "s"}</span></div><div className="filter-input"><Search size={15} /><Input aria-label="Filter documents" placeholder="Filter documents" value={documentFilter} onChange={(event) => setDocumentFilter(event.target.value)} /></div></div>
              <div className="document-list">
                {visibleDocuments.length ? visibleDocuments.map((record) => <DocumentRow key={record.id} record={record} onDelete={() => void deleteDocument(record)} onReindex={() => void reindexDocument(record)} />) : <div className="list-empty">{documents.length ? "No documents match this filter." : "No documents are indexed yet."}</div>}
              </div>
              <div className="privacy-note"><ShieldCheck size={16} /><p>Uploaded PDF bytes and their indexes are stored by the backend in its local database. Only upload sources you are authorized to process.</p></div>
            </div>
          )}

          {view === "knowledge" && (
            <div className="knowledge-view">
              <div className="page-heading"><div><div className="eyebrow"><Network size={14} /> SOURCE PROVENANCE</div><h1>Knowledge graph</h1><p>Trace each indexed passage to its document and original PDF page.</p></div></div>
              <KnowledgeGraphView graph={knowledgeGraph} loading={knowledgeGraph === null && !knowledgeGraphError} error={knowledgeGraphError} />
            </div>
          )}

          {view === "evaluation" && (
            <div className="evaluation-view">
              <div className="page-heading"><div><div className="eyebrow"><Gauge size={14} /> MEASURED RESULTS</div><h1>Evaluation workspace</h1><p>Start a protected development run against the reviewed dataset configured on the backend.</p></div></div>
              <div className="eval-grid">
                <Card><CardHeader><h2>Run a dataset split</h2><p>Jobs are restricted to the backend’s configured reviewed dataset and require a development bearer token.</p></CardHeader><CardContent>
                  <label className="field-label" htmlFor="split">Dataset split</label><select id="split" className="select-control" value={evaluationSplit} onChange={(event) => setEvaluationSplit(event.target.value)}><option value="development">Development</option><option value="validation">Validation</option><option value="test">Held-out test</option></select>
                  <label className="field-label" htmlFor="dev-token">Development bearer token</label><Input id="dev-token" type="password" autoComplete="off" value={evaluationToken} onChange={(event) => setEvaluationToken(event.target.value)} placeholder="Configured only on the backend" />
                  <p className="field-hint">The model API key stays on the backend and is never requested here.</p>
                  <Button className="eval-run-button" onClick={() => void startEvaluation()} disabled={apiStatus !== "online" || evaluationJob?.status === "running" || evaluationJob?.status === "queued"}><Gauge size={15} />Start evaluation</Button>
                </CardContent></Card>
                <Card><CardHeader><h2>Run status</h2><p>Metrics appear only after the backend completes a real run.</p></CardHeader><CardContent>
                  {evaluationJob ? <div className="job-status"><Badge className={`status-badge ${evaluationJob.status}`}>{evaluationJob.status}</Badge><p>Split: <strong>{evaluationJob.split}</strong></p><code>{evaluationJob.id}</code>{evaluationJob.error && <div className="message-error compact"><AlertCircle size={15} /><span>{evaluationJob.error.message}</span></div>}{evaluationJob.result && <MetricsSummary result={evaluationJob.result} />}</div> : <div className="evidence-empty"><Clock3 size={20} /><p>No development evaluation job has been started in this session.</p></div>}
                  {evaluationError && <div className="notice error-notice" role="alert"><AlertCircle size={16} /><span>{evaluationError}</span></div>}
                </CardContent></Card>
              </div>
              <div className="evaluation-note"><ShieldAlert size={17} /><p>There is no official campus dataset in this checkout. Synthetic fixtures exercise software behavior only and are not policy evaluation results.</p></div>
            </div>
          )}
        </div>
        <footer className="app-footer"><span>CampusReg · Evidence tied to indexed sources</span><span>Local development workspace</span></footer>
      </section>
    </main>
  );
}

function DocumentRow({ record, onDelete, onReindex }: { record: DocumentRecord; onDelete: () => void; onReindex: () => void }) {
  const fixture = isSyntheticFixture(record);
  return (
    <article className="document-row">
      <div className="document-type-icon"><FileText size={18} /></div>
      <div className="document-main"><div className="document-name-line"><strong>{record.filename}</strong><Badge className={`status-badge ${record.ingestion_status}`}>{record.ingestion_status}</Badge>{record.redaction_version < 1 && <Badge className="status-badge failed">Re-index required</Badge>}{fixture && <Badge className="fixture-label">Synthetic fixture · not policy</Badge>}</div><p>{record.document_type} · {formatBytes(record.size_bytes)} · {record.page_count} pages · {record.chunk_count} passages</p><span className="document-detail">Added {new Date(record.created_at).toLocaleDateString()} · SHA-256 {record.content_hash.slice(0, 12)}</span><RedactionSummaryLabel document={record} /></div>
      <div className="document-actions"><Button variant="ghost" size="icon" title="Re-index document" aria-label={`Re-index ${record.filename}`} onClick={onReindex}><RefreshCw size={15} /></Button><Button variant="ghost" size="icon" title="Delete document" aria-label={`Delete ${record.filename}`} onClick={onDelete}><Trash2 size={15} /></Button></div>
    </article>
  );
}

function RedactionSummaryLabel({ document }: { document: DocumentRecord }) {
  if (document.redaction_version < 1) return <span className="document-detail privacy-summary">Privacy preprocessing has not run · re-index before asking questions</span>;
  const summary = document.redaction_summary;
  const total = summary?.total ?? 0;
  if (!total) return <span className="document-detail privacy-summary">Privacy preprocessing · no supported labelled fields detected</span>;
  const categories = Object.entries(summary.counts_by_category ?? {})
    .filter(([, count]) => Number.isFinite(count) && count > 0)
    .map(([category, count]) => `${category}: ${count}`);
  const pages = Object.keys(summary.counts_by_page ?? {}).sort((a, b) => Number(a) - Number(b));
  return <span className="document-detail privacy-summary">Privacy preprocessing · {total} replacement{total === 1 ? "" : "s"}{categories.length ? ` · ${categories.join(" · ")}` : ""}{pages.length ? ` · PDF page${pages.length === 1 ? "" : "s"} ${pages.join(", ")}` : ""}</span>;
}

function KnowledgeGraphView({ graph, loading, error }: { graph: KnowledgeGraph | null; loading: boolean; error: string }) {
  if (loading) return <div className="graph-state" role="status"><LoaderCircle className="spin" size={18} />Loading source relationships…</div>;
  if (error) return <div className="notice error-notice" role="alert"><AlertCircle size={16} />{error}</div>;
  if (!graph?.nodes.length) return <div className="graph-state"><FileText size={20} /><p>Index a text-based policy PDF to build its source graph.</p></div>;

  const documents = graph.nodes.filter((node) => node.type === "document");
  const nodesById = new Map(graph.nodes.map((node) => [node.id, node]));
  return <section className="knowledge-graph" aria-label="Document and passage provenance graph">
    <p className="graph-scope">{graph.scope_note}</p>
    {graph.truncated && <p className="graph-truncated" role="status">Showing the first 500 indexed passages. Narrow the source set to inspect a smaller graph.</p>}
    {documents.map((document) => {
      const pageIds = graph.edges.filter((edge) => edge.source === document.id && edge.relation === "contains_page").map((edge) => edge.target);
      return <article className="graph-document" key={document.id}>
        <div className="graph-node document-node"><BookOpenText size={16} /><div><span>DOCUMENT</span><strong>{document.document_name}</strong></div></div>
        {pageIds.map((pageId) => {
          const page = nodesById.get(pageId);
          if (!page) return null;
          const passageIds = graph.edges.filter((edge) => edge.source === page.id && edge.relation === "contains_passage").map((edge) => edge.target);
          return <div className="graph-page" key={page.id}>
            <div className="graph-node page-node"><FileText size={15} /><strong>{page.label}</strong></div>
            <div className="graph-passages">{passageIds.map((passageId) => {
              const passage = nodesById.get(passageId);
              return passage ? <article className="graph-node passage-node" key={passage.id}>
                <span className="graph-passage-title">{passage.label} · {passage.citation}</span>
                <blockquote>{passage.text}</blockquote>
              </article> : null;
            })}</div>
          </div>;
        })}
      </article>;
    })}
  </section>;
}

function ConflictPanel({ reports }: { reports: ConflictReport[] }) {
  return <section className="conflict-panel" aria-label="Potential document conflicts"><div className="conflict-panel-heading"><div><div className="eyebrow warning-eyebrow"><ShieldAlert size={14} /> HUMAN REVIEW REQUIRED</div><h2>Potential differences across sources</h2></div><Badge className="conflict-count">{reports.length} report{reports.length === 1 ? "" : "s"}</Badge></div><p className="conflict-disclaimer">These comparisons flag possible inconsistencies. They do not determine which document is authoritative.</p>
    {reports.map((report, index) => <article className="conflict-report" key={`${report.passage_a.chunk_id}-${report.passage_b.chunk_id}-${index}`}><div className="conflict-report-title"><strong>{report.subject}</strong><Badge className="conflict-status">{report.status.replaceAll("_", " ")}</Badge></div><p>{report.explanation}</p><div className="conflict-sources"><ConflictSource passage={report.passage_a} scope={report.scope.passage_a} /><ConflictSource passage={report.passage_b} scope={report.scope.passage_b} /></div>{report.limitations.length > 0 && <p className="conflict-limit">{report.limitations.join(" ")}</p>}</article>)}
  </section>;
}

function ConflictSource({ passage, scope }: { passage: ConflictReport["passage_a"]; scope: string | null }) {
  return <div className="conflict-source"><div><FileText size={14} /><strong>{passage.document_name}</strong><Badge>PDF p. {passage.page}</Badge></div><blockquote>{passage.text}</blockquote>{scope && <span className="scope-label">Scope evidence: “{scope}”</span>}</div>;
}

function MetricsSummary({ result }: { result: Record<string, unknown> }) {
  const metrics = result.metrics;
  if (!metrics || typeof metrics !== "object") return <p className="field-hint">Completed; metrics were not included in the job response.</p>;
  return <pre className="metrics-output">{JSON.stringify(metrics, null, 2)}</pre>;
}
