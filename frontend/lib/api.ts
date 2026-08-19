// lib/api.ts — the single place the frontend talks to the FastAPI backend.
// Every function here maps to one endpoint in backend/api.py.

const BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
// Optional shared access token for deployed instances (matches backend APP_ACCESS_TOKEN).
const TOKEN = process.env.NEXT_PUBLIC_API_TOKEN || "";

function authHeaders(extra: Record<string, string> = {}): Record<string, string> {
  return TOKEN ? { ...extra, Authorization: `Bearer ${TOKEN}` } : extra;
}

export type Source = { content: string; source: string; page: string | number };

export type Usage = {
  input_tokens: number;
  output_tokens: number;
  context_window: number;
  session_input: number;
  session_output: number;
};

export type Stats = { documents: number; chunks: number };

export type Doc = { source: string; chunks: number };

export type QuizQuestion = {
  question: string;
  options: Record<string, string>;
  answer: string;
  explanation?: string;
};

export type StreamMeta = {
  sources?: Source[];
  confidence?: number;
  latency_ms?: number;
  web_used?: boolean;
  usage?: Usage;
  cost?: string;
  error?: string;
};

async function req(path: string, options?: RequestInit) {
  const res = await fetch(`${BASE}${path}`, {
    ...options,
    headers: authHeaders((options?.headers as Record<string, string>) || {}),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = await res.json();
      detail = data.detail || detail;
    } catch {
      // response wasn't JSON — keep statusText
    }
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  health: (): Promise<{ status: string; documents?: number; chunks?: number }> =>
    req("/health"),

  upload: (file: File): Promise<{ filename: string; chunks: number; stats: Stats }> => {
    const form = new FormData();
    form.append("file", file);
    return req("/upload", { method: "POST", body: form });
  },

  // Smart: sends any link; backend routes YouTube -> transcript, else -> web page.
  link: (url: string): Promise<{ chunks: number; stats: Stats }> =>
    req("/link", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    }),

  chatStream: async (
    params: { question: string; history: { role: string; content: string }[]; pattern: string; source: string; paperOpts: string },
    handlers: { onToken: (t: string) => void; onNotice: (n: string) => void; onDone: (meta: StreamMeta) => void },
    signal?: AbortSignal,
  ): Promise<void> => {
    const res = await fetch(`${BASE}/chat-stream`, {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        question: params.question,
        history: params.history,
        pattern: params.pattern,
        source: params.source,
        paper_opts: params.paperOpts,
      }),
      signal,
    });
    if (!res.ok) {
      let detail = "Stream failed";
      try {
        const data = await res.json();
        detail = data.detail || detail;
      } catch {
        // keep default
      }
      throw new Error(detail);
    }
    if (!res.body) throw new Error("Stream failed");
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, idx).trim();
        buf = buf.slice(idx + 1);
        if (!line) continue;
        try {
          const obj = JSON.parse(line);
          if (obj.token) handlers.onToken(obj.token as string);
          if (obj.notice) handlers.onNotice(obj.notice as string);
          if (obj.done) handlers.onDone(obj);
        } catch {
          // ignore a partial/bad line
        }
      }
    }
  },

  quiz: (num_questions: number, source = "", topic = ""): Promise<{ questions: QuizQuestion[] }> =>
    req("/quiz", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ num_questions, source, topic }),
    }),

  summary: (source = ""): Promise<{ summary: string }> =>
    req("/summary", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source }),
    }),

  stats: (): Promise<Stats> => req("/stats"),

  documents: (): Promise<{ documents: Doc[] }> => req("/documents"),

  deleteDocument: (source: string): Promise<{ status: string; stats: Stats }> =>
    req("/documents/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source }),
    }),

  reset: (): Promise<{ status: string }> => req("/reset", { method: "POST" }),

  extract: (file: File): Promise<{ text: string }> => {
    const form = new FormData();
    form.append("file", file);
    return req("/extract", { method: "POST", body: form });
  },

  exportPaper: async (text: string, format: string): Promise<void> => {
    const res = await fetch(`${BASE}/export`, {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ text, format }),
    });
    if (!res.ok) {
      let detail = "Export failed";
      try {
        const data = await res.json();
        detail = data.detail || detail;
      } catch {
        // keep default
      }
      throw new Error(detail);
    }
    const blob = await res.blob();
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `studymind-paper.${format}`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(link.href);
  },
};
