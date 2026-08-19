"use client";

import { useEffect, useRef, useState, type ChangeEvent, type DragEvent } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  BookOpen, FileText, MessageSquare, Pin, Pencil, Trash2, Plus,
  Paperclip, Mic, ArrowUp, Square, Menu, Globe, Download, ListChecks, ScrollText,
  UploadCloud, X, RotateCcw, FileDown,
} from "lucide-react";
import { api, type QuizQuestion, type Source, type Usage, type Stats, type Doc, type QuotaSnapshot, type PaperLayout } from "@/lib/api";

type Msg = {
  role: "user" | "assistant";
  content: string;
  type?: "quiz" | "paper";
  quiz?: QuizQuestion[];
  quizAnswers?: Record<number, string>;
  quizDone?: boolean;
  sources?: Source[];
  confidence?: number;
  latency?: number;
  web?: boolean;
  model?: string; // which AI model answered (empty/primary = normal)
};

type Convo = {
  id: string;
  title: string;
  messages: Msg[];
  pinned?: boolean;
  kind?: "chat" | "paper";
  pattern?: string;       // attached format sample text (paper conversations)
  patternName?: string;
  patternId?: string;     // stored .docx sample — export clones the real file
  patternLayout?: PaperLayout; // visual spec from a photo sample
  refs?: string[];        // exam-paper reference documents ([] / unset = all materials)
};

const MD =
  "text-[15px] leading-relaxed text-gray-800 [&_h1]:text-lg [&_h1]:font-semibold [&_h1]:mt-3 [&_h1]:text-gray-900 [&_h2]:text-base [&_h2]:font-semibold [&_h2]:mt-3 [&_h2]:text-gray-900 [&_h3]:font-medium [&_h3]:mt-3 [&_h3]:text-gray-900 [&_p]:my-2 [&_ul]:list-disc [&_ul]:pl-5 [&_ul]:my-2 [&_ol]:list-decimal [&_ol]:pl-5 [&_ol]:my-2 [&_li]:my-1 [&_strong]:font-semibold [&_strong]:text-gray-900 [&_code]:bg-indigo-50 [&_code]:px-1.5 [&_code]:py-0.5 [&_code]:rounded [&_code]:text-indigo-700 [&_hr]:border-gray-200 [&_hr]:my-3 [&_a]:text-indigo-600 [&_a]:underline";

/* ── Illomra logo: open book with a rising spark of 'ilm' (knowledge) ── */
function Logo({ size = 34 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 40 40" aria-label="Illomra">
      <defs>
        <linearGradient id="illomra-g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#6366f1" />
          <stop offset="1" stopColor="#8b5cf6" />
        </linearGradient>
      </defs>
      <rect x="1.5" y="1.5" width="37" height="37" rx="11" fill="url(#illomra-g)" />
      <path d="M10.5 27.5c3.4-1.8 6.6-1.8 9.5-.2V15.2c-2.9-1.6-6.1-1.6-9.5.2z" fill="#fff" opacity=".96" />
      <path d="M29.5 27.5c-3.4-1.8-6.6-1.8-9.5-.2V15.2c2.9-1.6 6.1-1.6 9.5.2z" fill="#fff" opacity=".78" />
      <circle cx="20" cy="9.6" r="2.3" fill="#fbbf24" />
    </svg>
  );
}

function fmtNum(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1000) return (n / 1000).toFixed(1) + "K";
  return String(n);
}

function shortName(s: string, max = 26): string {
  return s.length > max ? s.slice(0, max - 1) + "…" : s;
}

function fmtEta(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.round((seconds % 3600) / 60);
  return h > 0 ? `${h}h ${m}m` : `${m}m`;
}

function makeConvo(kind: "chat" | "paper" = "chat"): Convo {
  const id = typeof crypto !== "undefined" && crypto.randomUUID ? crypto.randomUUID() : String(Date.now());
  return { id, title: kind === "paper" ? "New paper" : "New chat", messages: [], kind };
}

/* ── Quiz card: answers persist on the message, so a reload shows the finished quiz ── */
function QuizCard({ questions, saved, savedDone, onAnswer, onComplete, onRetry, onPractice }: {
  questions: QuizQuestion[];
  saved: Record<number, string>;
  savedDone: boolean;
  onAnswer: (answers: Record<number, string>) => void;
  onComplete: () => void;
  onRetry: () => void;
  onPractice: (missed: string[]) => void;
}) {
  const answers = saved;
  const done = savedDone;
  const score = questions.reduce((n, q, i) => n + (answers[i] === q.answer ? 1 : 0), 0);
  const missed = questions.filter((q, i) => answers[i] !== q.answer).map((q) => q.question);
  return (
    <div className="space-y-4">
      {questions.map((q, i) => (
        <div key={i}>
          <div className="text-sm font-medium mb-2 text-gray-900">{i + 1}. {q.question}</div>
          <div className="space-y-1.5">
            {Object.entries(q.options).map(([letter, text]) => {
              const chosen = answers[i] === letter;
              const correct = q.answer === letter;
              let cls = "border-gray-300 text-gray-700 hover:border-indigo-400 hover:bg-indigo-50/50";
              if (done) {
                if (correct) cls = "border-green-500 bg-green-50 text-green-800";
                else if (chosen) cls = "border-red-400 bg-red-50 text-red-700";
                else cls = "border-gray-200 text-gray-400";
              } else if (chosen) cls = "border-indigo-500 bg-indigo-50 text-indigo-900";
              return (
                <button key={letter} disabled={done} onClick={() => onAnswer({ ...answers, [i]: letter })} className={`w-full text-left text-sm px-3 py-2 rounded-lg border transition ${cls}`}>
                  <span className="font-medium mr-1.5">{letter})</span>{text}
                </button>
              );
            })}
          </div>
          {done && q.explanation && <div className="text-xs text-gray-500 mt-1.5">💡 {q.explanation}</div>}
        </div>
      ))}
      {!done ? (
        <button onClick={onComplete} disabled={Object.keys(answers).length < questions.length} className="w-full py-2 rounded-lg bg-indigo-600 text-white font-medium text-sm hover:bg-indigo-500 disabled:opacity-40 transition">Submit answers</button>
      ) : (
        <div className="space-y-3">
          <div className="text-center">
            <div className="text-2xl font-bold text-indigo-600">{score} / {questions.length}</div>
            <div className="text-xs text-gray-500">{Math.round((score / questions.length) * 100)}% correct</div>
          </div>
          <div className="flex flex-wrap gap-2 justify-center">
            <button onClick={onRetry} className="inline-flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border border-gray-300 text-gray-600 hover:border-indigo-400 hover:text-indigo-700 transition">
              <RotateCcw size={13} /> Try again
            </button>
            {missed.length > 0 && (
              <button onClick={() => onPractice(missed)} className="inline-flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg bg-indigo-600 text-white hover:bg-indigo-500 transition">
                <ListChecks size={13} /> Practice my mistakes
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default function Home() {
  const [convos, setConvos] = useState<Convo[]>([]);
  const [currentId, setCurrentId] = useState("");
  const [hydrated, setHydrated] = useState(false);

  const [connected, setConnected] = useState<boolean | null>(null);
  const [quota, setQuota] = useState<QuotaSnapshot | null>(null);
  const [docs, setDocs] = useState<Doc[]>([]);
  const [scopeSource, setScopeSource] = useState(""); // "" = all materials
  const [stats, setStats] = useState<Stats>({ documents: 0, chunks: 0 });
  const [input, setInput] = useState("");
  const [thinking, setThinking] = useState(false);
  const [thinkingNote, setThinkingNote] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [url, setUrl] = useState("");
  const [showAdd, setShowAdd] = useState(false);
  const [usage, setUsage] = useState<Usage | null>(null);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [quizLoading, setQuizLoading] = useState(false);
  const [quizCount, setQuizCount] = useState(5);
  const [patternLoading, setPatternLoading] = useState(false);
  const [listening, setListening] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [dragOver, setDragOver] = useState(false);

  const recognitionRef = useRef<{ stop: () => void; start: () => void } | null>(null);
  const voiceBaseRef = useRef("");
  const chatEndRef = useRef<HTMLDivElement>(null);
  const scrollBoxRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const saveTimerRef = useRef<number | null>(null);
  // Token batching: streaming chunks are buffered and flushed every ~80ms so the
  // markdown doesn't re-render (and visibly "vibrate") on every tiny token.
  const pendingTokensRef = useRef("");
  const flushTimerRef = useRef<number | null>(null);

  /* ── Persistence (debounced; survives streaming without re-serializing every token) ── */
  // Canonical client-only localStorage hydration — must run in a mount effect
  // (localStorage doesn't exist during SSR), so the strict hooks rule is off here.
  // NOTE: storage keys keep the old "studymind_" prefix on purpose — renaming
  // them would silently wipe existing users' chats.
  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => {
    let loaded: Convo[] = [];
    try {
      loaded = JSON.parse(localStorage.getItem("studymind_convos") || "[]");
    } catch {
      loaded = [];
    }
    const savedId = localStorage.getItem("studymind_current") || "";
    if (!Array.isArray(loaded) || loaded.length === 0) {
      const c = makeConvo();
      loaded = [c];
      setCurrentId(c.id);
    } else {
      setCurrentId(loaded.some((c) => c.id === savedId) ? savedId : loaded[0].id);
    }
    setConvos(loaded);
    setHydrated(true);
  }, []);
  /* eslint-enable react-hooks/set-state-in-effect */

  useEffect(() => {
    if (!hydrated) return;
    if (saveTimerRef.current) window.clearTimeout(saveTimerRef.current);
    saveTimerRef.current = window.setTimeout(() => {
      try {
        localStorage.setItem("studymind_convos", JSON.stringify(convos));
        localStorage.setItem("studymind_current", currentId);
      } catch {
        flashError("Browser storage is full — delete some old chats, or use Export chats.");
      }
    }, 400);
    return () => {
      if (saveTimerRef.current) window.clearTimeout(saveTimerRef.current);
    };
  }, [convos, currentId, hydrated]);

  async function refreshDocs() {
    try {
      const [d, s] = await Promise.all([api.documents(), api.stats()]);
      setDocs(d.documents);
      setStats(s);
    } catch {
      // backend not up yet — ignore
    }
  }

  async function refreshQuota() {
    try {
      setQuota(await api.quota());
    } catch {
      // backend not up yet — ignore
    }
  }

  // Mount-only backend handshake (async — state lands after the fetch resolves).
  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => {
    api.health().then(() => setConnected(true)).catch(() => setConnected(false));
    refreshDocs();
    refreshQuota();
  }, []);
  /* eslint-enable react-hooks/set-state-in-effect */

  const current = convos.find((c) => c.id === currentId);
  const messages = current?.messages ?? [];
  const pattern = current?.pattern ?? "";
  const patternName = current?.patternName ?? "";
  const isPaper = current?.kind === "paper";
  // Derived: if the scoped document was deleted, behave as "all materials".
  const effectiveScope = scopeSource && docs.some((d) => d.source === scopeSource) ? scopeSource : "";

  // Auto-scroll — instant (not animated) while streaming so the view doesn't
  // judder, and only when the user is already near the bottom (never fight a
  // reader who scrolled up).
  const lastLen = messages.length ? messages[messages.length - 1].content.length : 0;
  useEffect(() => {
    const box = scrollBoxRef.current;
    if (box) {
      const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 160;
      if (!nearBottom && streaming) return;
    }
    chatEndRef.current?.scrollIntoView({ behavior: streaming ? "auto" : "smooth" });
  }, [messages.length, lastLen, thinking, quizLoading, summaryLoading, streaming]);

  function flashError(m: string) {
    setError(m);
    window.setTimeout(() => setError(null), 6000);
  }

  function updateConvo(id: string, patch: Partial<Convo>) {
    setConvos((prev) => prev.map((c) => (c.id === id ? { ...c, ...patch } : c)));
  }

  function setMessages(updater: (m: Msg[]) => Msg[]) {
    setConvos((prev) =>
      prev.map((c) => {
        if (c.id !== currentId) return c;
        const msgs = updater(c.messages);
        const firstUser = msgs.find((x) => x.role === "user");
        const title = (c.title === "New chat" || c.title === "New paper") && firstUser ? firstUser.content.slice(0, 42) : c.title;
        return { ...c, messages: msgs, title };
      }),
    );
  }

  function newChat() {
    const c = makeConvo("chat");
    setConvos((prev) => [c, ...prev]);
    setCurrentId(c.id);
    setSidebarOpen(false);
  }

  function newPaper() {
    const c = makeConvo("paper");
    setConvos((prev) => [c, ...prev]);
    setCurrentId(c.id);
    setSidebarOpen(false);
  }

  function togglePin(id: string) {
    setConvos((prev) => prev.map((c) => (c.id === id ? { ...c, pinned: !c.pinned } : c)));
  }

  function startRename(c: Convo) {
    setRenamingId(c.id);
    setRenameValue(c.title);
  }

  function commitRename() {
    if (renamingId) {
      const title = renameValue.trim() || "Untitled";
      setConvos((prev) => prev.map((c) => (c.id === renamingId ? { ...c, title } : c)));
    }
    setRenamingId(null);
  }

  function deleteChat(id: string) {
    const next = convos.filter((c) => c.id !== id);
    if (next.length === 0) {
      const c = makeConvo();
      setConvos([c]);
      setCurrentId(c.id);
      return;
    }
    setConvos(next);
    if (id === currentId) setCurrentId(next[0].id);
  }

  function exportChats() {
    const blob = new Blob([JSON.stringify(convos, null, 2)], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "illomra-chats.json";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(link.href);
  }

  const contentReady = stats.chunks > 0;

  /* ── Uploading (multi-file + drag & drop) ── */
  async function uploadFiles(files: FileList | File[]) {
    const list = Array.from(files);
    if (!list.length) return;
    for (const file of list) {
      setBusy(`Indexing ${file.name}…`);
      try {
        await api.upload(file);
      } catch (err) {
        flashError(err instanceof Error ? err.message : `Upload failed: ${file.name}`);
      }
    }
    setBusy(null);
    await refreshDocs();
  }

  async function handleUpload(e: ChangeEvent<HTMLInputElement>) {
    if (e.target.files) await uploadFiles(e.target.files);
    e.target.value = "";
  }

  function handleDrop(e: DragEvent) {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files?.length) uploadFiles(e.dataTransfer.files);
  }

  async function handleUrl() {
    const target = url.trim();
    if (!target) return;
    setBusy("Reading link…");
    try {
      await api.link(target);
      await refreshDocs();
      setUrl("");
    } catch (err) {
      flashError(err instanceof Error ? err.message : "Could not read that link");
    } finally {
      setBusy(null);
    }
  }

  async function removeDoc(source: string) {
    try {
      await api.deleteDocument(source);
      await refreshDocs();
    } catch (err) {
      flashError(err instanceof Error ? err.message : "Could not delete that document");
    }
  }

  async function clearKnowledge() {
    if (!window.confirm("Remove ALL your materials? This can't be undone.")) return;
    try {
      await api.reset();
      setScopeSource("");
      await refreshDocs();
    } catch (err) {
      flashError(err instanceof Error ? err.message : "Could not clear your materials");
    }
  }

  /* ── Chat ── */
  function toggleRef(sourceName: string) {
    if (!current) return;
    const refs = current.refs ?? [];
    const next = refs.includes(sourceName) ? refs.filter((r) => r !== sourceName) : [...refs, sourceName];
    updateConvo(current.id, { refs: next });
  }

  async function send(question: string) {
    const q = question.trim();
    if (!q || thinking || streaming) return;
    const history = messages.map((m) => ({ role: m.role, content: m.content }));
    setMessages((m) => [...m, { role: "user", content: q }]);
    setInput("");
    setThinking(true);
    setThinkingNote("");
    setStreaming(true);
    abortRef.current = new AbortController();
    let started = false;
    const msgType: Msg["type"] = isPaper ? "paper" : undefined;

    const appendPending = () => {
      const chunk = pendingTokensRef.current;
      if (!chunk) return;
      pendingTokensRef.current = "";
      setMessages((m) => {
        const copy = [...m];
        const last = copy[copy.length - 1];
        if (last && last.role === "assistant") copy[copy.length - 1] = { ...last, content: last.content + chunk };
        return copy;
      });
    };
    const flushNow = () => {
      if (flushTimerRef.current) {
        window.clearTimeout(flushTimerRef.current);
        flushTimerRef.current = null;
      }
      appendPending();
    };
    try {
      await api.chatStream(
        // Paper mode uses its own reference list (chips); main chat uses the scope dropdown.
        { question: q, history, pattern: isPaper ? pattern : "", source: isPaper ? "" : effectiveScope, refs: isPaper ? (current?.refs ?? []).filter((r) => docs.some((d) => d.source === r)) : [] },
        {
          onToken: (tok) => {
            if (!started) {
              // First token renders immediately so the thinking dots vanish fast.
              started = true;
              setThinking(false);
              setThinkingNote("");
              setMessages((m) => [...m, { role: "assistant", content: tok, type: msgType }]);
              return;
            }
            // Batch the rest: collect tokens, paint at most ~12x/second.
            pendingTokensRef.current += tok;
            if (flushTimerRef.current == null) {
              flushTimerRef.current = window.setTimeout(() => {
                flushTimerRef.current = null;
                appendPending();
              }, 80);
            }
          },
          onNotice: (note) => setThinkingNote(note),
          onDone: (meta) => {
            flushNow();
            if (meta.error) {
              flashError(meta.error);
              if (!started) setMessages((m) => [...m, { role: "assistant", content: "⚠️ " + meta.error }]);
              return;
            }
            setUsage(meta.usage ?? null);
            refreshQuota();
            // Empty/blank answer: still show a real bubble instead of a dead turn.
            if (!started) {
              started = true;
              setThinking(false);
              setMessages((m) => [...m, { role: "assistant", content: "I couldn't find an answer in your material for that. Try rephrasing, or change \"Answering from\" below.", type: msgType, sources: meta.sources, confidence: meta.confidence, latency: meta.latency_ms, web: meta.web_used, model: meta.model }]);
              return;
            }
            setMessages((m) => {
              const copy = [...m];
              const last = copy[copy.length - 1];
              if (last && last.role === "assistant") copy[copy.length - 1] = { ...last, sources: meta.sources, confidence: meta.confidence, latency: meta.latency_ms, web: meta.web_used, model: meta.model };
              return copy;
            });
          },
        },
        abortRef.current.signal,
      );
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        // user pressed stop — keep whatever streamed
      } else {
        flashError(err instanceof Error ? err.message : "Chat failed");
        if (!started) setMessages((m) => [...m, { role: "assistant", content: "⚠️ Something went wrong — is the backend running?" }]);
      }
    } finally {
      flushNow();
      setThinking(false);
      setThinkingNote("");
      setStreaming(false);
      abortRef.current = null;
    }
  }

  function stopStreaming() {
    abortRef.current?.abort();
  }

  async function genSummary() {
    if (thinking || streaming || summaryLoading) return;
    setSummaryLoading(true);
    try {
      const res = await api.summary(effectiveScope);
      setMessages((m) => [...m, { role: "assistant", content: "## 📋 Study notes\n\n" + (res.summary || "") }]);
    } catch (err) {
      flashError(err instanceof Error ? err.message : "Couldn't build study notes");
    } finally {
      setSummaryLoading(false);
      refreshQuota();
    }
  }

  async function genQuiz(topic = "") {
    if (thinking || streaming || quizLoading) return;
    setQuizLoading(true);
    try {
      const res = await api.quiz(quizCount, effectiveScope, topic);
      const intro = topic ? "Round two — focused on what you missed:" : "Test yourself — from your own material:";
      setMessages((m) => [...m, { role: "assistant", type: "quiz", content: intro, quiz: res.questions, quizAnswers: {}, quizDone: false }]);
    } catch (err) {
      flashError(err instanceof Error ? err.message : "Couldn't build the quiz");
    } finally {
      setQuizLoading(false);
      refreshQuota();
    }
  }

  function patchMessage(index: number, patch: Partial<Msg>) {
    setMessages((m) => m.map((msg, i) => (i === index ? { ...msg, ...patch } : msg)));
  }

  async function handlePatternUpload(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file || !current) return;
    setPatternLoading(true);
    try {
      const res = await api.extract(file);
      updateConvo(current.id, { pattern: res.text || "", patternName: file.name, patternId: res.pattern_id || "", patternLayout: res.layout });
    } catch (err) {
      flashError(err instanceof Error ? err.message : "Could not read that paper");
    } finally {
      setPatternLoading(false);
      e.target.value = "";
    }
  }

  async function downloadPaper(text: string, fmt: string) {
    try {
      await api.exportPaper(text, fmt, current?.patternId || "", current?.patternLayout);
    } catch (err) {
      flashError(err instanceof Error ? err.message : "Download failed — is the backend running?");
    }
  }

  function toggleVoice() {
    const w = window as unknown as { SpeechRecognition?: new () => never; webkitSpeechRecognition?: new () => never };
    const SR = w.SpeechRecognition || w.webkitSpeechRecognition;
    if (!SR) {
      flashError("Voice input needs Chrome or Edge.");
      return;
    }
    if (listening) {
      recognitionRef.current?.stop();
      setListening(false);
      return;
    }
    const rec = new SR() as unknown as {
      lang: string;
      interimResults: boolean;
      continuous: boolean;
      onresult: (e: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void;
      onerror: (e: { error?: string }) => void;
      onend: () => void;
      start: () => void;
      stop: () => void;
    };
    rec.lang = "en-US";
    rec.interimResults = true;
    rec.continuous = false;
    voiceBaseRef.current = input ? input + " " : "";
    rec.onresult = (e) => {
      let t = "";
      for (let i = 0; i < e.results.length; i++) t += e.results[i][0].transcript;
      setInput(voiceBaseRef.current + t);
    };
    rec.onerror = (e) => {
      setListening(false);
      if (e?.error && e.error !== "aborted" && e.error !== "no-speech") flashError("Mic: " + e.error + (e.error === "not-allowed" ? " — allow microphone access" : ""));
    };
    rec.onend = () => setListening(false);
    recognitionRef.current = rec;
    setListening(true);
    try {
      rec.start();
    } catch {
      setListening(false);
    }
  }

  const ctxPct = usage ? Math.max(0.6, (usage.input_tokens / usage.context_window) * 100) : 0;
  const sessionTokens = usage ? usage.session_input + usage.session_output : 0;
  const generating = thinking || streaming || quizLoading || summaryLoading;

  return (
    <div className="h-dvh flex bg-[#f7f7f5] text-gray-800">
      {/* ── Sidebar ── */}
      <aside className={`${sidebarOpen ? "flex" : "hidden"} md:flex w-[268px] shrink-0 flex-col border-r border-gray-200 bg-white fixed md:static z-30 h-full`}>
        <div className="p-4 pb-3">
          <div className="flex items-center gap-2.5">
            <Logo size={34} />
            <div>
              <div className="font-semibold text-[17px] tracking-tight text-gray-900 leading-none">Illomra</div>
              <div className="mt-1 text-[10.5px] text-gray-400 leading-none">Your material. Your answers.</div>
            </div>
          </div>
        </div>
        <div className="px-3 grid grid-cols-2 gap-2">
          <button onClick={newChat} className="flex items-center justify-center gap-1.5 px-2 py-2 rounded-xl border border-gray-300 text-gray-700 hover:border-indigo-400 hover:bg-indigo-50/60 text-sm font-medium transition">
            <Plus size={14} /> Chat
          </button>
          <button onClick={newPaper} className="flex items-center justify-center gap-1.5 px-2 py-2 rounded-xl bg-indigo-600 text-white hover:bg-indigo-500 text-sm font-medium transition shadow-sm">
            <FileText size={14} /> Exam paper
          </button>
        </div>

        <div className="mt-3 px-2 flex-1 overflow-y-auto">
          {(() => {
            const row = (c: Convo) => (
              <div key={c.id} className={`group flex items-center gap-1.5 rounded-lg px-2 py-1.5 text-sm cursor-pointer transition ${c.id === currentId ? "bg-indigo-50 text-indigo-900" : "text-gray-600 hover:bg-gray-100"}`} onClick={() => { setCurrentId(c.id); setSidebarOpen(false); }}>
                <span className={`shrink-0 ${c.id === currentId ? "text-indigo-500" : "text-gray-400"}`}>{c.kind === "paper" ? <FileText size={13} /> : <MessageSquare size={13} />}</span>
                {renamingId === c.id ? (
                  <input
                    autoFocus
                    value={renameValue}
                    onChange={(e) => setRenameValue(e.target.value)}
                    onClick={(e) => e.stopPropagation()}
                    onKeyDown={(e) => { if (e.key === "Enter") commitRename(); if (e.key === "Escape") setRenamingId(null); }}
                    onBlur={commitRename}
                    className="flex-1 min-w-0 bg-white border border-indigo-400 rounded px-1 py-0.5 text-sm outline-none"
                  />
                ) : (
                  <span className="truncate flex-1 inline-flex items-center gap-1">{c.pinned && <Pin size={11} className="text-indigo-500 shrink-0" />}{c.title || "Untitled"}</span>
                )}
                <div className="opacity-100 md:opacity-0 md:group-hover:opacity-100 flex items-center gap-1.5 shrink-0">
                  <button onClick={(e) => { e.stopPropagation(); togglePin(c.id); }} title={c.pinned ? "Unpin" : "Pin"} aria-label={c.pinned ? "Unpin chat" : "Pin chat"} className="text-gray-400 hover:text-indigo-600"><Pin size={13} /></button>
                  <button onClick={(e) => { e.stopPropagation(); startRename(c); }} title="Rename" aria-label="Rename chat" className="text-gray-400 hover:text-gray-700"><Pencil size={13} /></button>
                  <button onClick={(e) => { e.stopPropagation(); deleteChat(c.id); }} title="Delete" aria-label="Delete chat" className="text-gray-400 hover:text-red-500"><Trash2 size={13} /></button>
                </div>
              </div>
            );
            const byPin = (a: Convo, b: Convo) => (b.pinned ? 1 : 0) - (a.pinned ? 1 : 0);
            const chats = convos.filter((c) => c.kind !== "paper").sort(byPin);
            const papers = convos.filter((c) => c.kind === "paper").sort(byPin);
            return (
              <>
                {chats.length > 0 && (
                  <>
                    <div className="px-2 text-[11px] uppercase tracking-wider text-gray-400 mb-1">Chats</div>
                    {chats.map(row)}
                  </>
                )}
                {papers.length > 0 && (
                  <>
                    <div className="px-2 text-[11px] uppercase tracking-wider text-gray-400 mb-1 mt-3 inline-flex items-center gap-1"><FileText size={10} /> Exam papers</div>
                    {papers.map(row)}
                  </>
                )}
              </>
            );
          })()}
        </div>

        {/* My materials — the document manager */}
        <div className="border-t border-gray-200 p-3 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-gray-700 inline-flex items-center gap-1.5"><BookOpen size={14} className="text-indigo-500" /> My materials</span>
            <button onClick={() => setShowAdd((v) => !v)} className="text-xs font-medium text-indigo-600 hover:text-indigo-500">{showAdd ? "Close" : "+ Add"}</button>
          </div>

          {docs.length > 0 ? (
            <div className="space-y-1 max-h-44 overflow-y-auto">
              {docs.map((d) => (
                <div key={d.source} className="group flex items-center gap-1.5 text-xs bg-gray-50 border border-gray-100 rounded-lg px-2 py-1.5">
                  <FileText size={12} className="shrink-0 text-gray-400" />
                  <span className="truncate flex-1 text-gray-700" title={`${d.source} · ${d.chunks} sections`}>{d.source}</span>
                  <button onClick={() => removeDoc(d.source)} title="Remove this material" aria-label="Remove this material" className="opacity-100 md:opacity-0 md:group-hover:opacity-100 text-gray-400 hover:text-red-500 shrink-0"><Trash2 size={12} /></button>
                </div>
              ))}
              <button onClick={clearKnowledge} className="w-full text-[11px] text-gray-400 hover:text-red-500 pt-1 text-left px-2">Clear all materials</button>
            </div>
          ) : (
            <div className="text-[11px] text-gray-400">Nothing yet — add your notes, books, or slides.</div>
          )}

          {showAdd && (
            <div className="space-y-2 pt-1 border-t border-gray-100">
              <label className={`block rounded-lg border border-dashed p-2 text-center cursor-pointer text-xs transition ${busy ? "opacity-50 pointer-events-none" : "border-gray-300 text-gray-600 hover:border-indigo-400 hover:bg-indigo-50/40"}`}>
                <input type="file" multiple accept=".pdf,.txt,.md,.csv,.docx,.pptx,.png,.jpg,.jpeg,.webp" onChange={handleUpload} className="hidden" disabled={!!busy} />
                <span className="inline-flex items-center gap-1.5 justify-center"><UploadCloud size={13} /> Upload files <span className="text-gray-400">(PDF/Word/PPT/photo)</span></span>
              </label>
              <div className="flex gap-1">
                <input value={url} onChange={(e) => setUrl(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") handleUrl(); }} placeholder="Paste any link or YouTube URL" disabled={!!busy} className="flex-1 rounded-lg bg-gray-50 border border-gray-300 outline-none px-2 py-1.5 text-xs placeholder:text-gray-400 focus:border-indigo-400" />
                <button onClick={handleUrl} disabled={!!busy || !url.trim()} className="px-2 rounded-lg bg-gray-800 text-white text-xs hover:bg-gray-700 disabled:opacity-40">Add</button>
              </div>
              {busy && <div className="text-[11px] text-indigo-600 animate-pulse">{busy}</div>}
            </div>
          )}
        </div>

        {/* Daily AI limit — how much free capacity is left across the model chain */}
        {quota && quota.totals.capacity > 0 && (() => {
          const used = quota.totals.used_today;
          const cap = quota.totals.capacity;
          const pct = Math.min(100, Math.round((used / cap) * 100));
          const barColor = pct >= 90 ? "bg-red-500" : pct >= 65 ? "bg-amber-500" : "bg-indigo-500";
          const exhausted = quota.models.filter((m) => m.status === "exhausted").length;
          return (
            <div className="border-t border-gray-200 px-3 py-2.5">
              <div className="flex justify-between text-[11px] text-gray-500 mb-1">
                <span>AI limit today</span>
                <span>≈ {used} / {cap}</span>
              </div>
              <div className="h-1.5 rounded-full bg-gray-100 overflow-hidden">
                <div className={`h-full rounded-full transition-all ${barColor}`} style={{ width: `${Math.max(1, pct)}%` }} />
              </div>
              <div className="mt-1 flex justify-between text-[10px] text-gray-400">
                <span>{exhausted > 0 ? `${exhausted} of ${quota.models.length} models at limit` : `${quota.models.length} models available`}</span>
                <span>resets in {fmtEta(quota.totals.resets_in_s)}</span>
              </div>
            </div>
          );
        })()}

        {/* Footer: export + session tokens (tucked away — not in the user's face) */}
        <div className="border-t border-gray-200 px-3 py-2 flex items-center justify-between text-[11px] text-gray-400">
          <button onClick={exportChats} className="inline-flex items-center gap-1 hover:text-gray-600" title="Download all chats as JSON"><FileDown size={11} /> Export chats</button>
          {usage && (
            <details className="relative">
              <summary className="cursor-pointer list-none hover:text-gray-600">{fmtNum(sessionTokens)} tok</summary>
              <div className="absolute bottom-6 right-0 w-44 rounded-lg border border-gray-200 bg-white p-2 shadow-lg z-40">
                <div className="flex justify-between mb-1 text-gray-500"><span>context</span><span>{fmtNum(usage.input_tokens)} / 1M</span></div>
                <div className="h-1 rounded-full bg-gray-100 overflow-hidden"><div className="h-full bg-indigo-500 rounded-full" style={{ width: `${ctxPct}%` }} /></div>
              </div>
            </details>
          )}
        </div>
      </aside>

      {sidebarOpen && <div className="fixed inset-0 bg-black/30 z-20 md:hidden" onClick={() => setSidebarOpen(false)} />}

      {/* ── Main ── */}
      <div className="flex-1 flex flex-col min-w-0">
        <header className="h-14 shrink-0 border-b border-gray-200 bg-white/80 backdrop-blur flex items-center justify-between px-4 gap-3">
          <div className="flex items-center gap-2 min-w-0">
            <button onClick={() => setSidebarOpen(true)} className="md:hidden text-gray-500 p-1.5" aria-label="Menu"><Menu size={20} /></button>
            <span className="truncate font-medium text-gray-900">{current?.title || "New chat"}</span>
            {isPaper ? (
              <span className="shrink-0 text-[10px] uppercase tracking-wide px-1.5 py-0.5 rounded bg-indigo-600 text-white">Exam paper</span>
            ) : contentReady ? (
              <span className="shrink-0 text-[10px] uppercase tracking-wide px-1.5 py-0.5 rounded bg-indigo-50 text-indigo-600 border border-indigo-100">From your material</span>
            ) : (
              <span className="shrink-0 text-[10px] uppercase tracking-wide px-1.5 py-0.5 rounded bg-gray-100 text-gray-500">General chat</span>
            )}
          </div>
          {connected === null ? (
            <span className="text-xs px-2 py-1 rounded-full bg-gray-100 text-gray-500">● connecting…</span>
          ) : connected === false ? (
            <span className="text-xs px-2 py-1 rounded-full bg-red-50 text-red-600">● offline</span>
          ) : (
            <span className="text-xs px-2 py-1 rounded-full bg-green-50 text-green-600">● online</span>
          )}
        </header>

        {error && <div className="fixed bottom-28 left-1/2 -translate-x-1/2 z-40 rounded-xl bg-red-600 text-white text-sm px-4 py-2 shadow-lg max-w-[90vw]">{error}</div>}

        <div ref={scrollBoxRef} className="flex-1 overflow-y-auto">
          <div className="max-w-3xl mx-auto px-4 py-6 space-y-5">
            {messages.length === 0 && (
              isPaper ? (
                /* ── Paper-mode empty state ── */
                <div className="min-h-[55vh] flex flex-col items-center justify-center text-center gap-4">
                  <div className="h-14 w-14 rounded-2xl bg-indigo-600 text-white grid place-items-center shadow-md shadow-indigo-200"><FileText size={26} /></div>
                  <div className="text-xl text-gray-900 font-semibold tracking-tight">Generate an exam paper from your material</div>
                  <ol className="text-sm text-gray-600 text-left space-y-1.5 max-w-md list-decimal pl-5">
                    <li>Attach a sample/past paper (file or photo) with <Paperclip size={12} className="inline" /> — it copies YOUR institute&apos;s exact format</li>
                    <li>Pick which <span className="text-gray-900 font-medium">references</span> to draw from below (or leave all)</li>
                    <li>Just say it in plain words — <span className="text-gray-900 font-medium">&quot;same pattern as the sample, 3 scenario-based questions from the neural networks part, 50 marks, with answer key&quot;</span></li>
                  </ol>
                  {!contentReady && <p className="text-xs text-gray-400">Tip: add your course material in <span className="text-indigo-600 font-medium">My materials</span> first.</p>}
                </div>
              ) : contentReady ? (
                /* ── Materials loaded: study-focused starters ── */
                <div className="min-h-[55vh] flex flex-col items-center justify-center text-center gap-4">
                  <Logo size={52} />
                  <div className="text-xl text-gray-900 font-semibold tracking-tight">Your material is ready. What do you want to learn?</div>
                  <div className="flex flex-wrap gap-2 justify-center max-w-md">
                    <button onClick={() => send("Summarize the key ideas")} className="text-xs px-3.5 py-2 rounded-full border border-gray-300 bg-white text-gray-700 hover:border-indigo-400 hover:text-indigo-700 transition shadow-sm">Summarize the key ideas</button>
                    <button onClick={() => send("Explain the hardest concept simply")} className="text-xs px-3.5 py-2 rounded-full border border-gray-300 bg-white text-gray-700 hover:border-indigo-400 hover:text-indigo-700 transition shadow-sm">Explain the hardest concept simply</button>
                    <button onClick={() => genQuiz()} className="text-xs px-3.5 py-2 rounded-full bg-indigo-600 text-white hover:bg-indigo-500 transition shadow-sm inline-flex items-center gap-1"><ListChecks size={12} /> Test me on it</button>
                  </div>
                  {effectiveScope && <p className="text-[11px] text-gray-400">Answering only from <span className="text-indigo-600 font-medium">{shortName(effectiveScope)}</span></p>}
                </div>
              ) : (
                /* ── First run: upload-first hero — this is the product, not a chatbot ── */
                <div className="min-h-[60vh] flex flex-col items-center justify-center gap-6">
                  <div className="flex flex-col items-center gap-2.5">
                    <Logo size={56} />
                    <div className="text-2xl font-semibold tracking-tight text-gray-900">Illomra</div>
                    <div className="text-sm text-gray-500 -mt-1.5">Your material. Your answers. Your exam papers.</div>
                  </div>
                  <label
                    onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                    onDragLeave={() => setDragOver(false)}
                    onDrop={handleDrop}
                    className={`w-full max-w-lg cursor-pointer rounded-2xl border-2 border-dashed p-8 text-center transition bg-white shadow-sm ${dragOver ? "border-indigo-500 bg-indigo-50" : "border-gray-300 hover:border-indigo-400"}`}
                  >
                    <input type="file" multiple accept=".pdf,.txt,.md,.csv,.docx,.pptx,.png,.jpg,.jpeg,.webp" onChange={handleUpload} className="hidden" disabled={!!busy} />
                    <UploadCloud size={32} className="mx-auto text-indigo-500 mb-3" />
                    <div className="text-gray-900 font-medium text-lg">Drop in your study material</div>
                    <div className="text-sm text-gray-500 mt-1">Textbooks, notes, slides, past papers — PDF, Word, PPT, even a <span className="text-gray-700 font-medium">photo of handwritten notes</span></div>
                    {busy && <div className="text-xs text-indigo-600 animate-pulse mt-3">{busy}</div>}
                  </label>

                  <div className="grid sm:grid-cols-3 gap-3 w-full max-w-lg">
                    <div className="rounded-xl border border-gray-200 bg-white p-3.5 shadow-sm">
                      <div className="flex items-center gap-1.5 text-gray-900 font-medium text-[13px]"><BookOpen size={14} className="text-indigo-500" /> Ask your material</div>
                      <p className="text-[11px] text-gray-500 mt-1 leading-relaxed">Every answer cited from YOUR books &amp; notes.</p>
                    </div>
                    <div className="rounded-xl border border-gray-200 bg-white p-3.5 shadow-sm">
                      <div className="flex items-center gap-1.5 text-gray-900 font-medium text-[13px]"><ListChecks size={14} className="text-indigo-500" /> Test yourself</div>
                      <p className="text-[11px] text-gray-500 mt-1 leading-relaxed">Quizzes that re-drill what you got wrong.</p>
                    </div>
                    <button onClick={newPaper} className="rounded-xl border border-indigo-200 bg-indigo-50/60 p-3.5 text-left hover:bg-indigo-50 transition shadow-sm">
                      <div className="flex items-center gap-1.5 text-indigo-700 font-medium text-[13px]"><FileText size={14} /> Exam papers →</div>
                      <p className="text-[11px] text-gray-500 mt-1 leading-relaxed">In your institute&apos;s exact format, with answer key.</p>
                    </button>
                  </div>

                  <p className="text-xs text-gray-400">…or just <button onClick={() => document.querySelector<HTMLTextAreaElement>("textarea")?.focus()} className="text-indigo-600 font-medium hover:underline">ask anything</button> without uploading.</p>
                </div>
              )
            )}

            {messages.map((m, i) => (
              <div key={i} className={m.role === "user" ? "flex justify-end" : "flex justify-start"}>
                <div className={m.role === "user" ? "max-w-[85%] rounded-2xl rounded-br-md bg-indigo-600 text-white px-4 py-2.5 text-[15px] whitespace-pre-wrap shadow-sm" : "max-w-full w-full"}>
                  {m.role === "assistant" ? (
                    <div className="rounded-2xl bg-white border border-gray-200 px-4 py-3 shadow-sm">
                      {m.type === "quiz" ? (
                        <>
                          <div className="text-sm text-gray-500 mb-3">{m.content}</div>
                          {m.quiz && (
                            <QuizCard
                              questions={m.quiz}
                              saved={m.quizAnswers ?? {}}
                              savedDone={m.quizDone ?? false}
                              onAnswer={(a) => patchMessage(i, { quizAnswers: a })}
                              onComplete={() => patchMessage(i, { quizDone: true })}
                              onRetry={() => patchMessage(i, { quizAnswers: {}, quizDone: false })}
                              onPractice={(missed) => genQuiz(missed.map((q) => q.slice(0, 90)).join(" ; "))}
                            />
                          )}
                        </>
                      ) : (
                        <div className={MD}>
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                        </div>
                      )}

                      {m.type === "paper" && m.content.length > 100 && !(streaming && i === messages.length - 1) && (
                        <div className="mt-3 flex flex-wrap gap-2 items-center border-t border-gray-100 pt-3">
                          <span className="text-xs text-gray-400 inline-flex items-center gap-1"><Download size={12} /> Download:</span>
                          <button onClick={() => downloadPaper(m.content, "docx")} className="text-xs px-3 py-1.5 rounded-lg bg-indigo-600 text-white hover:bg-indigo-500 transition font-medium">Word</button>
                          <button onClick={() => downloadPaper(m.content, "pdf")} className="text-xs px-3 py-1.5 rounded-lg border border-gray-300 text-gray-600 hover:border-indigo-400 hover:text-indigo-700 transition">PDF</button>
                          <button onClick={() => send("Generate a Variant B of the paper above: different questions of equal difficulty, identical format, sections, and marks distribution.")} className="text-xs px-3 py-1.5 rounded-lg border border-gray-300 text-gray-600 hover:border-indigo-400 hover:text-indigo-700 transition">Variant B</button>
                        </div>
                      )}

                      {m.web && <div className="mt-2 text-[11px] text-sky-600 inline-flex items-center gap-1"><Globe size={11} /> Searched the web</div>}

                      {m.model && quota && m.model !== quota.primary && (
                        <div className="mt-2 text-[11px] text-amber-600" title="The primary model was at its free limit, so a backup Gemini model answered — same material, same context.">⚡ Answered by backup model ({m.model})</div>
                      )}

                      {m.sources && m.sources.length > 0 && (
                        <details className="mt-2">
                          <summary className="cursor-pointer text-xs text-gray-400 hover:text-gray-600 select-none">From your material — {m.sources.length} source{m.sources.length > 1 ? "s" : ""}</summary>
                          <div className="mt-2 space-y-2">
                            {m.sources.map((s, j) => (
                              <div key={j} className="text-xs bg-gray-50 border border-gray-100 rounded-lg p-2">
                                <div className="text-indigo-600 mb-1">
                                  {s.source}{" "}
                                  {String(s.page).startsWith("http") ? (
                                    <a href={String(s.page)} target="_blank" rel="noopener noreferrer" className="underline hover:text-indigo-500">↗ open</a>
                                  ) : (
                                    <span className="text-gray-400">· {String(s.page)}</span>
                                  )}
                                </div>
                                <div className="text-gray-500 line-clamp-3">{s.content}</div>
                              </div>
                            ))}
                          </div>
                        </details>
                      )}
                    </div>
                  ) : (
                    m.content
                  )}
                </div>
              </div>
            ))}

            {(thinking || quizLoading || summaryLoading) && (
              <div className="flex justify-start">
                <div className="rounded-2xl bg-white border border-gray-200 px-4 py-3 text-gray-400 text-sm flex items-center gap-2.5 shadow-sm">
                  <span className="inline-flex gap-1 text-indigo-400">
                    <span className="animate-bounce">●</span>
                    <span className="animate-bounce [animation-delay:0.15s]">●</span>
                    <span className="animate-bounce [animation-delay:0.3s]">●</span>
                  </span>
                  {quizLoading && <span className="text-xs">Building your quiz from {effectiveScope ? shortName(effectiveScope) : "your material"}…</span>}
                  {summaryLoading && <span className="text-xs">Writing study notes from {effectiveScope ? shortName(effectiveScope) : "your material"}…</span>}
                  {thinkingNote && <span className="text-xs text-amber-600">{thinkingNote}</span>}
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>
        </div>

        {/* ── Input area ── */}
        <div className="shrink-0 border-t border-gray-200 bg-white p-3 pb-4">
          {/* Quick study actions + scope */}
          {contentReady && docs.length > 0 && !isPaper && (
            <div className="max-w-3xl mx-auto mb-2 flex flex-wrap items-center gap-2 text-xs">
              <span className="text-gray-400 shrink-0">Answering from:</span>
              <select value={effectiveScope} onChange={(e) => setScopeSource(e.target.value)} className="min-w-0 max-w-[45%] truncate bg-gray-50 border border-gray-300 rounded-lg px-2 py-1.5 text-gray-700 outline-none focus:border-indigo-400" aria-label="Answering from which material">
                <option value="">All my materials</option>
                {docs.map((d) => (
                  <option key={d.source} value={d.source}>{d.source}</option>
                ))}
              </select>
              <span className="flex-1" />
              <div className="inline-flex items-center rounded-lg border border-gray-300 overflow-hidden bg-white">
                <button onClick={() => genQuiz()} disabled={generating} className="inline-flex items-center gap-1 px-2.5 py-1.5 text-gray-600 hover:bg-indigo-50 hover:text-indigo-700 disabled:opacity-40 transition font-medium" title="Quiz yourself on your material"><ListChecks size={13} /> Test me</button>
                <select value={quizCount} onChange={(e) => setQuizCount(Number(e.target.value))} className="bg-gray-50 border-l border-gray-300 px-1 py-1.5 text-gray-500 outline-none" aria-label="Number of quiz questions">
                  <option value={5}>5</option>
                  <option value={10}>10</option>
                  <option value={15}>15</option>
                </select>
              </div>
              <button onClick={genSummary} disabled={generating} className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg border border-gray-300 bg-white text-gray-600 hover:bg-indigo-50 hover:text-indigo-700 disabled:opacity-40 transition font-medium" title="Structured study notes from your material"><ScrollText size={13} /> Study notes</button>
            </div>
          )}

          {/* Paper options */}
          {isPaper && (
            <div className="max-w-3xl mx-auto mb-2 flex flex-wrap items-center gap-x-3 gap-y-2 text-xs">
              {patternName ? (
                <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded-lg bg-indigo-50 text-indigo-700 border border-indigo-200" title={current?.patternId ? "Word sample — exports clone your file's exact header, footer and fonts" : current?.patternLayout ? "Photo sample — exports mirror its fonts and layout" : "Format sample attached"}>
                  <Paperclip size={11} /> {shortName(patternName, 22)}
                  {current?.patternId && <span className="text-[9px] uppercase tracking-wide text-indigo-500 font-semibold">exact</span>}
                  {!current?.patternId && current?.patternLayout && <span className="text-[9px] uppercase tracking-wide text-indigo-500 font-semibold">styled</span>}
                  <button onClick={() => current && updateConvo(current.id, { pattern: "", patternName: "", patternId: "", patternLayout: undefined })} className="hover:text-red-500" aria-label="Remove format"><X size={11} /></button>
                </span>
              ) : (
                <span className="text-gray-400 inline-flex items-center gap-1"><Paperclip size={11} /> {patternLoading ? "Reading format…" : "no format attached"}</span>
              )}
              {docs.length > 0 && (
                <span className="inline-flex items-center flex-wrap gap-1.5">
                  <span className="text-gray-400">References:</span>
                  {docs.map((d) => {
                    const refs = current?.refs ?? [];
                    const active = refs.length === 0 || refs.includes(d.source);
                    const explicit = refs.includes(d.source);
                    return (
                      <button
                        key={d.source}
                        onClick={() => toggleRef(d.source)}
                        title={active ? "Included — click to " + (explicit ? "exclude" : "narrow to specific documents") : "Excluded — click to include"}
                        className={`px-2 py-1 rounded-lg border transition max-w-[160px] truncate ${active ? "border-indigo-300 bg-indigo-50 text-indigo-700" : "border-gray-200 bg-white text-gray-400 line-through"}`}
                      >
                        {shortName(d.source, 20)}
                      </button>
                    );
                  })}
                  {(current?.refs ?? []).length === 0 && <span className="text-gray-300">(all)</span>}
                </span>
              )}
            </div>
          )}

          <div className="max-w-3xl mx-auto flex items-end gap-2">
            {isPaper && (
              <label title="Attach a format / past-paper sample" className={`h-12 w-12 shrink-0 grid place-items-center rounded-2xl border-2 cursor-pointer transition ${patternName ? "bg-indigo-50 border-indigo-300 text-indigo-600" : "bg-white border-gray-200 text-gray-500 hover:border-indigo-300 hover:text-indigo-600"}`}>
                <input type="file" accept=".pdf,.txt,.md,.csv,.docx,.pptx,.png,.jpg,.jpeg,.webp" onChange={handlePatternUpload} className="hidden" />
                <Paperclip size={18} />
              </label>
            )}
            <button onClick={toggleVoice} disabled={connected === false} title="Voice to text (Chrome/Edge)" className={`h-12 w-12 shrink-0 grid place-items-center rounded-2xl border-2 transition disabled:opacity-40 ${listening ? "bg-red-50 border-red-300 text-red-500 animate-pulse" : "bg-white border-gray-200 text-gray-500 hover:border-indigo-300 hover:text-indigo-600"}`} aria-label="Voice input">
              <Mic size={18} />
            </button>
            <div className="flex-1 flex items-end rounded-2xl border-2 border-gray-200 bg-white focus-within:border-indigo-400 focus-within:shadow-[0_0_0_4px_rgba(99,102,241,0.08)] transition shadow-sm">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(input); } }}
                placeholder={listening ? "Listening…" : isPaper ? "Describe the paper — e.g. '50-mark paper from chapter 3, Q1 scenario-based'" : contentReady ? "Ask about your material…" : "Ask me anything…"}
                disabled={connected === false}
                rows={1}
                className="flex-1 resize-none bg-transparent outline-none px-4 py-3 text-[15px] text-gray-900 placeholder:text-gray-400 disabled:opacity-50 max-h-40"
              />
              {streaming ? (
                <button onClick={stopStreaming} className="h-9 w-9 shrink-0 grid place-items-center rounded-xl bg-gray-800 text-white hover:bg-gray-700 transition m-1.5" aria-label="Stop generating">
                  <Square size={13} fill="currentColor" />
                </button>
              ) : (
                <button onClick={() => send(input)} disabled={connected === false || generating || !input.trim()} className="h-9 w-9 shrink-0 grid place-items-center rounded-xl bg-indigo-600 text-white hover:bg-indigo-500 disabled:opacity-30 transition m-1.5" aria-label="Send">
                  <ArrowUp size={17} strokeWidth={2.5} />
                </button>
              )}
            </div>
          </div>
          <div className="max-w-3xl mx-auto mt-1.5 text-center text-[10px] text-gray-300">
            Illomra answers from your material and says so when it can&apos;t.
          </div>
        </div>
      </div>

    </div>
  );
}
