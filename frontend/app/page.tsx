"use client";

import { useEffect, useRef, useState, type ChangeEvent } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api, type QuizQuestion, type Source, type Usage, type Stats, type Doc } from "@/lib/api";

type Msg = {
  role: "user" | "assistant";
  content: string;
  type?: "quiz" | "paper";
  quiz?: QuizQuestion[];
  sources?: Source[];
  confidence?: number;
  latency?: number;
  web?: boolean;
};

type Convo = { id: string; title: string; messages: Msg[]; pinned?: boolean; kind?: "chat" | "paper" };

const SUGGESTED = ["Summarize the key ideas", "Explain the hardest concept simply", "Give me an example"];
const GENERAL_SUGGESTED = ["Explain a tricky concept simply", "Help me make a study plan", "Give me practice questions on a topic"];

const MD =
  "text-[15px] leading-relaxed [&_h1]:text-lg [&_h1]:font-semibold [&_h1]:mt-3 [&_h2]:text-base [&_h2]:font-semibold [&_h2]:mt-3 [&_h3]:font-medium [&_h3]:mt-3 [&_p]:my-2 [&_ul]:list-disc [&_ul]:pl-5 [&_ul]:my-2 [&_ol]:list-decimal [&_ol]:pl-5 [&_ol]:my-2 [&_li]:my-1 [&_strong]:font-semibold [&_strong]:text-white [&_code]:bg-slate-700 [&_code]:px-1.5 [&_code]:py-0.5 [&_code]:rounded [&_code]:text-emerald-300 [&_hr]:border-slate-700 [&_hr]:my-3 [&_a]:text-emerald-400 [&_a]:underline";

function fmtNum(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1000) return (n / 1000).toFixed(1) + "K";
  return String(n);
}

function makeConvo(kind: "chat" | "paper" = "chat"): Convo {
  const id = typeof crypto !== "undefined" && crypto.randomUUID ? crypto.randomUUID() : String(Date.now());
  return { id, title: kind === "paper" ? "New paper" : "New chat", messages: [], kind };
}

function QuizCard({ questions }: { questions: QuizQuestion[] }) {
  const [answers, setAnswers] = useState<Record<number, string>>({});
  const [done, setDone] = useState(false);
  const score = questions.reduce((n, q, i) => n + (answers[i] === q.answer ? 1 : 0), 0);
  return (
    <div className="space-y-4">
      {questions.map((q, i) => (
        <div key={i}>
          <div className="text-sm font-medium mb-2">{i + 1}. {q.question}</div>
          <div className="space-y-1.5">
            {Object.entries(q.options).map(([letter, text]) => {
              const chosen = answers[i] === letter;
              const correct = q.answer === letter;
              let cls = "border-slate-700 hover:border-slate-500";
              if (done) {
                if (correct) cls = "border-emerald-500 bg-emerald-500/10 text-emerald-200";
                else if (chosen) cls = "border-red-500 bg-red-500/10 text-red-200";
                else cls = "border-slate-800 text-slate-500";
              } else if (chosen) cls = "border-emerald-500/60 bg-emerald-500/5";
              return (
                <button key={letter} disabled={done} onClick={() => setAnswers((a) => ({ ...a, [i]: letter }))} className={`w-full text-left text-sm px-3 py-2 rounded-lg border transition ${cls}`}>
                  <span className="font-medium mr-1.5">{letter})</span>{text}
                </button>
              );
            })}
          </div>
          {done && q.explanation && <div className="text-xs text-slate-500 mt-1.5">💡 {q.explanation}</div>}
        </div>
      ))}
      {!done ? (
        <button onClick={() => setDone(true)} disabled={Object.keys(answers).length < questions.length} className="w-full py-2 rounded-lg bg-emerald-500 text-slate-950 font-medium text-sm hover:bg-emerald-400 disabled:opacity-40 transition">Submit answers</button>
      ) : (
        <div className="text-center">
          <div className="text-2xl font-bold text-emerald-400">{score} / {questions.length}</div>
          <div className="text-xs text-slate-500">{Math.round((score / questions.length) * 100)}% correct</div>
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
  const [docs, setDocs] = useState<Doc[]>([]);
  const [scopeSource, setScopeSource] = useState(""); // "" = all documents
  const [stats, setStats] = useState<Stats>({ documents: 0, chunks: 0 });
  const [input, setInput] = useState("");
  const [thinking, setThinking] = useState(false);
  const [thinkingNote, setThinkingNote] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [url, setUrl] = useState("");
  const [showAdd, setShowAdd] = useState(false);
  const [usage, setUsage] = useState<Usage | null>(null);
  const [cost, setCost] = useState("");
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [quizLoading, setQuizLoading] = useState(false);
  const [examPattern, setExamPattern] = useState("");
  const [examName, setExamName] = useState("");
  const [examLoading, setExamLoading] = useState(false);
  const [listening, setListening] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");

  const recognitionRef = useRef<{ stop: () => void; start: () => void } | null>(null);
  const voiceBaseRef = useRef("");
  const chatEndRef = useRef<HTMLDivElement>(null);

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

  useEffect(() => {
    if (!hydrated) return;
    localStorage.setItem("studymind_convos", JSON.stringify(convos));
    localStorage.setItem("studymind_current", currentId);
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

  useEffect(() => {
    api.health().then(() => setConnected(true)).catch(() => setConnected(false));
    refreshDocs();
  }, []);

  // If the scoped document gets deleted, fall back to "all documents".
  useEffect(() => {
    if (scopeSource && !docs.some((d) => d.source === scopeSource)) setScopeSource("");
  }, [docs, scopeSource]);

  const current = convos.find((c) => c.id === currentId);
  const messages = current?.messages ?? [];

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, thinking]);

  useEffect(() => {
    setExamPattern("");
    setExamName("");
    setRenamingId(null);
  }, [currentId]);

  function flashError(m: string) {
    setError(m);
    window.setTimeout(() => setError(null), 6000);
  }

  function setMessages(updater: (m: Msg[]) => Msg[]) {
    setConvos((prev) =>
      prev.map((c) => {
        if (c.id !== currentId) return c;
        const msgs = updater(c.messages);
        const firstUser = msgs.find((x) => x.role === "user");
        const title = c.title === "New chat" && firstUser ? firstUser.content.slice(0, 42) : c.title;
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
    setConvos((prev) => {
      const next = prev.filter((c) => c.id !== id);
      if (next.length === 0) {
        const c = makeConvo();
        setCurrentId(c.id);
        return [c];
      }
      if (id === currentId) setCurrentId(next[0].id);
      return next;
    });
  }

  const contentReady = stats.chunks > 0;

  async function handleUpload(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setBusy(`Indexing ${file.name}…`);
    try {
      await api.upload(file);
      await refreshDocs();
    } catch (err) {
      flashError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setBusy(null);
      e.target.value = "";
    }
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
    if (!window.confirm("Remove ALL indexed documents? This can't be undone.")) return;
    try {
      await api.reset();
      setScopeSource("");
      await refreshDocs();
    } catch (err) {
      flashError(err instanceof Error ? err.message : "Could not clear the knowledge base");
    }
  }

  async function send(question: string) {
    const q = question.trim();
    if (!q || thinking) return;
    const history = messages.map((m) => ({ role: m.role, content: m.content }));
    setMessages((m) => [...m, { role: "user", content: q }]);
    setInput("");
    setThinking(true);
    setThinkingNote("");
    let started = false;
    try {
      await api.chatStream(
        q,
        history,
        current?.kind === "paper" ? examPattern : "",
        scopeSource,
        (tok) => {
          if (!started) {
            started = true;
            setThinking(false);
            setThinkingNote("");
            setMessages((m) => [...m, { role: "assistant", content: tok }]);
          } else {
            setMessages((m) => {
              const copy = [...m];
              const last = copy[copy.length - 1];
              if (last && last.role === "assistant") copy[copy.length - 1] = { ...last, content: last.content + tok };
              return copy;
            });
          }
        },
        (note) => setThinkingNote(note),
        (meta) => {
          if (meta.error) {
            flashError(meta.error);
            if (!started) setMessages((m) => [...m, { role: "assistant", content: "⚠️ " + meta.error }]);
            return;
          }
          setUsage(meta.usage ?? null);
          setCost(meta.cost ?? "");
          // Empty/blank answer: still show a real bubble instead of a dead turn.
          if (!started) {
            started = true;
            setThinking(false);
            setMessages((m) => [...m, { role: "assistant", content: "I couldn't find an answer in your material for that. Try rephrasing, or check the document scope above.", sources: meta.sources, confidence: meta.confidence, latency: meta.latency_ms, web: meta.web_used }]);
            return;
          }
          setMessages((m) => {
            const copy = [...m];
            const last = copy[copy.length - 1];
            if (last && last.role === "assistant") copy[copy.length - 1] = { ...last, sources: meta.sources, confidence: meta.confidence, latency: meta.latency_ms, web: meta.web_used };
            return copy;
          });
        },
      );
    } catch (err) {
      flashError(err instanceof Error ? err.message : "Chat failed");
      if (!started) setMessages((m) => [...m, { role: "assistant", content: "⚠️ Something went wrong — is the backend running?" }]);
    } finally {
      setThinking(false);
      setThinkingNote("");
    }
  }

  async function genSummary() {
    if (thinking) return;
    setSummaryLoading(true);
    try {
      const res = await api.summary(scopeSource);
      setMessages((m) => [...m, { role: "assistant", content: "## 📋 Summary\n\n" + (res.summary || "") }]);
    } catch (err) {
      flashError(err instanceof Error ? err.message : "Summary failed");
    } finally {
      setSummaryLoading(false);
    }
  }

  async function genQuiz() {
    if (thinking) return;
    setQuizLoading(true);
    try {
      const res = await api.quiz(5, scopeSource);
      setMessages((m) => [...m, { role: "assistant", type: "quiz", content: "Here's a quick quiz to test yourself:", quiz: res.questions }]);
    } catch (err) {
      flashError(err instanceof Error ? err.message : "Quiz failed");
    } finally {
      setQuizLoading(false);
    }
  }

  async function handlePatternUpload(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setExamLoading(true);
    try {
      const res = await api.extract(file);
      setExamPattern(res.text || "");
      setExamName(file.name);
    } catch (err) {
      flashError(err instanceof Error ? err.message : "Could not read that paper");
    } finally {
      setExamLoading(false);
      e.target.value = "";
    }
  }

  async function downloadPaper(text: string, fmt: string) {
    try {
      await api.exportPaper(text, fmt);
    } catch {
      flashError("Download failed — is the backend running?");
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

  return (
    <div className="h-screen flex bg-[#0b0f17] text-slate-200">
      {/* ── Sidebar ── */}
      <aside className={`${sidebarOpen ? "flex" : "hidden"} md:flex w-64 shrink-0 flex-col border-r border-slate-800 bg-[#080b11] fixed md:static z-30 h-full`}>
        <div className="p-3 flex items-center gap-2">
          <div className="h-8 w-8 rounded-lg bg-emerald-500/15 text-emerald-400 grid place-items-center">🎓</div>
          <div className="font-semibold">StudyMind</div>
        </div>
        <div className="px-3 grid grid-cols-2 gap-2">
          <button onClick={newChat} className="flex items-center justify-center gap-1.5 px-2 py-2 rounded-lg border border-slate-700 hover:border-emerald-500/50 hover:bg-slate-800/50 text-sm transition">
            <span className="text-base leading-none">＋</span> Chat
          </button>
          <button onClick={newPaper} className="flex items-center justify-center gap-1.5 px-2 py-2 rounded-lg border border-emerald-500/30 bg-emerald-500/10 text-emerald-300 hover:bg-emerald-500/20 text-sm transition">
            <span className="text-base leading-none">＋</span> Paper
          </button>
        </div>

        <div className="mt-3 px-2 flex-1 overflow-y-auto">
          <div className="px-2 text-[11px] uppercase tracking-wide text-slate-600 mb-1">Chats</div>
          {[...convos].sort((a, b) => (b.pinned ? 1 : 0) - (a.pinned ? 1 : 0)).map((c) => (
            <div key={c.id} className={`group flex items-center gap-1 rounded-lg px-2 py-1.5 text-sm cursor-pointer transition ${c.id === currentId ? "bg-slate-800 text-slate-100" : "text-slate-400 hover:bg-slate-800/50"}`} onClick={() => { setCurrentId(c.id); setSidebarOpen(false); }}>
              <span className="shrink-0 text-xs">{c.kind === "paper" ? "📝" : "💬"}</span>
              {renamingId === c.id ? (
                <input
                  autoFocus
                  value={renameValue}
                  onChange={(e) => setRenameValue(e.target.value)}
                  onClick={(e) => e.stopPropagation()}
                  onKeyDown={(e) => { if (e.key === "Enter") commitRename(); if (e.key === "Escape") setRenamingId(null); }}
                  onBlur={commitRename}
                  className="flex-1 min-w-0 bg-slate-900 border border-emerald-500/40 rounded px-1 py-0.5 text-sm outline-none"
                />
              ) : (
                <span className="truncate flex-1">{c.pinned && <span className="text-emerald-500 mr-0.5">📌</span>}{c.title || "Untitled"}</span>
              )}
              <div className="opacity-0 group-hover:opacity-100 flex items-center gap-1 shrink-0">
                <button onClick={(e) => { e.stopPropagation(); togglePin(c.id); }} title={c.pinned ? "Unpin" : "Pin"} className="text-slate-500 hover:text-emerald-400 text-xs">📌</button>
                <button onClick={(e) => { e.stopPropagation(); startRename(c); }} title="Rename" className="text-slate-500 hover:text-slate-200 text-xs">✎</button>
                <button onClick={(e) => { e.stopPropagation(); deleteChat(c.id); }} title="Delete" className="text-slate-500 hover:text-red-400 text-xs">🗑</button>
              </div>
            </div>
          ))}
        </div>

        {/* Knowledge — the document manager */}
        <div className="border-t border-slate-800 p-3 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm text-slate-300">📚 Knowledge {stats.chunks ? <span className="text-slate-600">· {stats.chunks} chunks</span> : null}</span>
            <button onClick={() => setShowAdd((v) => !v)} className="text-xs text-emerald-400 hover:text-emerald-300">{showAdd ? "Close" : "＋ Add"}</button>
          </div>

          {docs.length > 0 ? (
            <div className="space-y-1 max-h-44 overflow-y-auto">
              {docs.map((d) => (
                <div key={d.source} className="group flex items-center gap-1.5 text-xs bg-slate-800/40 rounded-lg px-2 py-1.5">
                  <span className="truncate flex-1" title={d.source}>{d.source}</span>
                  <span className="text-slate-600 shrink-0">{d.chunks}</span>
                  <button onClick={() => removeDoc(d.source)} title="Remove this document" className="opacity-0 group-hover:opacity-100 text-slate-500 hover:text-red-400 shrink-0">🗑</button>
                </div>
              ))}
              <button onClick={clearKnowledge} className="w-full text-[11px] text-slate-500 hover:text-red-400 pt-1">Clear all documents</button>
            </div>
          ) : (
            <div className="text-[11px] text-slate-600">No documents yet — add one below.</div>
          )}

          {showAdd && (
            <div className="space-y-2 pt-1 border-t border-slate-800/60">
              <label className={`block rounded-lg border border-dashed p-2 text-center cursor-pointer text-xs transition ${busy ? "opacity-50 pointer-events-none" : "border-slate-700 hover:border-emerald-500/50"}`}>
                <input type="file" accept=".pdf,.txt,.md,.csv,.docx,.pptx" onChange={handleUpload} className="hidden" disabled={!!busy} />
                📄 Upload file <span className="text-slate-500">(PDF/Word/PPT/…)</span>
              </label>
              <div className="flex gap-1">
                <input value={url} onChange={(e) => setUrl(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") handleUrl(); }} placeholder="Paste any link" disabled={!!busy} className="flex-1 rounded-lg bg-slate-800/60 border border-slate-700 outline-none px-2 py-1.5 text-xs placeholder:text-slate-600" />
                <button onClick={handleUrl} disabled={!!busy || !url.trim()} className="px-2 rounded-lg bg-slate-800 text-slate-200 text-xs hover:bg-slate-700 disabled:opacity-40">Add</button>
              </div>
              {busy && <div className="text-[11px] text-emerald-400 animate-pulse">{busy}</div>}
            </div>
          )}
        </div>

        {/* Usage */}
        {usage && (
          <div className="border-t border-slate-800 px-3 py-2 text-[11px] text-slate-500">
            <div className="flex justify-between mb-1"><span>context</span><span>{fmtNum(usage.input_tokens)} / 1M</span></div>
            <div className="h-1 rounded-full bg-slate-800 overflow-hidden"><div className="h-full bg-emerald-400 rounded-full" style={{ width: `${ctxPct}%` }} /></div>
            <div className="mt-1 flex justify-between"><span>session {fmtNum(sessionTokens)} tok</span><span>{cost}</span></div>
          </div>
        )}
      </aside>

      {sidebarOpen && <div className="fixed inset-0 bg-black/50 z-20 md:hidden" onClick={() => setSidebarOpen(false)} />}

      {/* ── Main ── */}
      <div className="flex-1 flex flex-col min-w-0">
        <header className="h-14 shrink-0 border-b border-slate-800 flex items-center justify-between px-4 gap-3">
          <div className="flex items-center gap-2 min-w-0">
            <button onClick={() => setSidebarOpen(true)} className="md:hidden text-slate-400 text-xl" aria-label="Menu">☰</button>
            <span className="truncate font-medium">{current?.title || "New chat"}</span>
          </div>
          <div className="flex items-center gap-2">
            <button onClick={genSummary} disabled={!contentReady || summaryLoading} className="text-xs px-3 py-1.5 rounded-lg border border-slate-700 text-slate-300 hover:border-emerald-500/50 disabled:opacity-40 transition">{summaryLoading ? "…" : "Summary"}</button>
            <button onClick={genQuiz} disabled={!contentReady || quizLoading} className="text-xs px-3 py-1.5 rounded-lg border border-slate-700 text-slate-300 hover:border-emerald-500/50 disabled:opacity-40 transition">{quizLoading ? "…" : "Quiz"}</button>
            {/* Paper/document generation now lives in the chat: attach a format with 📎 and just ask. */}
            {connected === false ? (
              <span className="text-xs px-2 py-1 rounded-full bg-red-500/10 text-red-400">● offline</span>
            ) : (
              <span className="text-xs px-2 py-1 rounded-full bg-emerald-500/10 text-emerald-400">● online</span>
            )}
          </div>
        </header>

        {error && <div className="fixed bottom-24 left-1/2 -translate-x-1/2 z-40 rounded-lg bg-red-600 text-white text-sm px-4 py-2 shadow-lg">{error}</div>}

        <div className="flex-1 overflow-y-auto">
          <div className="max-w-3xl mx-auto px-4 py-6 space-y-5">
            {messages.length === 0 && (
              <div className="h-[55vh] flex flex-col items-center justify-center text-center text-slate-500 gap-4">
                <div className="text-5xl">{current?.kind === "paper" ? "📝" : "🎓"}</div>
                <div className="text-lg text-slate-300">{current?.kind === "paper" ? "Generate a paper or document" : "What do you want to learn?"}</div>
                {current?.kind === "paper" ? (
                  <p className="text-sm max-w-sm">Attach a format sample with the 📎, then describe what you want — e.g. &quot;make a 10-MCQ paper in this format from chapter 3&quot;. Add source docs in 📚 Knowledge.</p>
                ) : contentReady ? (
                  <div className="flex flex-wrap gap-2 justify-center max-w-md">
                    {SUGGESTED.map((s) => (
                      <button key={s} onClick={() => send(s)} className="text-xs px-3 py-1.5 rounded-full border border-slate-700 text-slate-300 hover:border-emerald-500/50 hover:text-emerald-300 transition">{s}</button>
                    ))}
                  </div>
                ) : (
                  <div className="flex flex-col items-center gap-3">
                    <div className="flex flex-wrap gap-2 justify-center max-w-md">
                      {GENERAL_SUGGESTED.map((s) => (
                        <button key={s} onClick={() => send(s)} className="text-xs px-3 py-1.5 rounded-full border border-slate-700 text-slate-300 hover:border-emerald-500/50 hover:text-emerald-300 transition">{s}</button>
                      ))}
                    </div>
                    <p className="text-xs text-slate-600 max-w-sm">Ask me anything — or add a document in <span className="text-emerald-400">📚 Knowledge</span> to get answers grounded in your own material.</p>
                  </div>
                )}
              </div>
            )}

            {messages.map((m, i) => (
              <div key={i} className={m.role === "user" ? "flex justify-end" : "flex justify-start"}>
                <div className={m.role === "user" ? "max-w-[85%] rounded-2xl rounded-br-sm bg-emerald-500/15 border border-emerald-500/20 text-emerald-50 px-4 py-2.5 text-[15px] whitespace-pre-wrap" : "max-w-full w-full"}>
                  {m.role === "assistant" ? (
                    <div className="rounded-2xl bg-slate-800/40 border border-slate-700/50 px-4 py-3">
                      {m.type === "quiz" ? (
                        <>
                          <div className="text-sm text-slate-400 mb-3">{m.content}</div>
                          {m.quiz && <QuizCard questions={m.quiz} />}
                        </>
                      ) : (
                        <div className={MD}>
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                        </div>
                      )}

                      {m.type === "paper" && m.content.length > 100 && (
                        <div className="mt-3 flex flex-wrap gap-2 items-center">
                          <span className="text-xs text-slate-500">Download:</span>
                          <button onClick={() => downloadPaper(m.content, "pdf")} className="text-xs px-3 py-1.5 rounded-lg bg-emerald-500/15 border border-emerald-500/30 text-emerald-300 hover:bg-emerald-500/25 transition">⬇ PDF</button>
                          <button onClick={() => downloadPaper(m.content, "docx")} className="text-xs px-3 py-1.5 rounded-lg border border-slate-700 text-slate-300 hover:border-emerald-500/50 transition">⬇ Word</button>
                          <button onClick={() => downloadPaper(m.content, "pptx")} className="text-xs px-3 py-1.5 rounded-lg border border-slate-700 text-slate-300 hover:border-emerald-500/50 transition">⬇ PPTX</button>
                        </div>
                      )}

                      {/* match/latency meter removed — it was a misleading metric */}

                      {m.web && <div className="mt-2 text-[11px] text-sky-400">🌐 Searched the web</div>}

                      {m.sources && m.sources.length > 0 && (
                        <details className="mt-2">
                          <summary className="cursor-pointer text-xs text-slate-400 hover:text-slate-200 select-none">📚 {m.sources.length} source{m.sources.length > 1 ? "s" : ""}</summary>
                          <div className="mt-2 space-y-2">
                            {m.sources.map((s, j) => (
                              <div key={j} className="text-xs bg-slate-900/60 border border-slate-700/50 rounded-lg p-2">
                                <div className="text-emerald-400/90 mb-1">
                                  {s.source}{" "}
                                  {String(s.page).startsWith("http") ? (
                                    <a href={String(s.page)} target="_blank" rel="noopener noreferrer" className="underline hover:text-emerald-300">↗ open</a>
                                  ) : (
                                    <span className="text-slate-500">· {String(s.page)}</span>
                                  )}
                                </div>
                                <div className="text-slate-400 line-clamp-3">{s.content}</div>
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

            {thinking && (
              <div className="flex justify-start">
                <div className="rounded-2xl bg-slate-800/40 border border-slate-700/50 px-4 py-3 text-slate-400 text-sm flex items-center gap-2">
                  <span className="inline-flex gap-1">
                    <span className="animate-bounce">●</span>
                    <span className="animate-bounce [animation-delay:0.15s]">●</span>
                    <span className="animate-bounce [animation-delay:0.3s]">●</span>
                  </span>
                  {thinkingNote && <span className="text-xs text-amber-400/90">{thinkingNote}</span>}
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>
        </div>

        {/* Input */}
        <div className="shrink-0 border-t border-slate-800 p-3">
          {contentReady && docs.length > 0 && (
            <div className="max-w-3xl mx-auto mb-2 flex items-center gap-2 text-xs">
              <span className="text-slate-500 shrink-0">🎯 Scope:</span>
              <select value={scopeSource} onChange={(e) => setScopeSource(e.target.value)} className="min-w-0 max-w-[60%] truncate bg-slate-800/60 border border-slate-700 rounded-lg px-2 py-1 text-slate-200 outline-none focus:border-emerald-500/50">
                <option value="">All documents</option>
                {docs.map((d) => (
                  <option key={d.source} value={d.source}>{d.source}</option>
                ))}
              </select>
              {scopeSource && <span className="text-emerald-400/80 shrink-0">answers use only this document</span>}
            </div>
          )}
          {current?.kind === "paper" && examName && (
            <div className="max-w-3xl mx-auto mb-2 flex items-center gap-2 text-xs">
              <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded-lg bg-emerald-500/15 text-emerald-300 border border-emerald-500/20">
                📎 format: {examName}
                <button onClick={() => { setExamPattern(""); setExamName(""); }} className="hover:text-red-300" aria-label="Remove format">✕</button>
              </span>
            </div>
          )}
          <div className="max-w-3xl mx-auto flex items-end gap-2">
            {current?.kind === "paper" && (
              <label title="Attach a format / pattern sample" className={`h-11 w-11 shrink-0 grid place-items-center rounded-xl border cursor-pointer transition ${examName ? "bg-emerald-500/15 border-emerald-500/40 text-emerald-300" : "bg-slate-800/60 border-slate-700 text-slate-300 hover:border-emerald-500/50"}`}>
                <input type="file" accept=".pdf,.txt,.md,.csv,.docx,.pptx" onChange={handlePatternUpload} className="hidden" />
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21.44 11.05l-9.19 9.19a5 5 0 0 1-7.07-7.07l9.19-9.19a3.5 3.5 0 0 1 4.95 4.95l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" /></svg>
              </label>
            )}
            <button onClick={toggleVoice} disabled={connected === false} title="Voice to text (Chrome/Edge)" className={`h-11 w-11 shrink-0 grid place-items-center rounded-xl border transition disabled:opacity-40 ${listening ? "bg-red-500/20 border-red-500/50 text-red-300 animate-pulse" : "bg-slate-800/60 border-slate-700 text-slate-300 hover:border-emerald-500/50"}`} aria-label="Voice input">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z" /><path d="M19 10v1a7 7 0 0 1-14 0v-1" /><line x1="12" y1="18" x2="12" y2="22" /></svg>
            </button>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(input); } }}
              placeholder={listening ? "Listening…" : contentReady ? "Ask about your material…" : "Ask me anything…"}
              disabled={connected === false}
              rows={1}
              className="flex-1 resize-none rounded-xl bg-slate-800/60 border border-slate-700 focus:border-emerald-500/60 outline-none px-4 py-3 text-[15px] placeholder:text-slate-600 disabled:opacity-50 max-h-40"
            />
            <button onClick={() => send(input)} disabled={connected === false || thinking || !input.trim()} className="h-11 w-11 shrink-0 grid place-items-center rounded-xl bg-emerald-500 text-slate-950 hover:bg-emerald-400 disabled:opacity-40 disabled:hover:bg-emerald-500 transition" aria-label="Send">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M12 19V5M5 12l7-7 7 7" /></svg>
            </button>
          </div>
        </div>
      </div>

    </div>
  );
}
