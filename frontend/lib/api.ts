// lib/api.ts — the single place the frontend talks to the FastAPI backend.
// Every function here maps to one endpoint in backend/api.py.

const BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type Source = { content: string; source: string; page: string | number };

export type Usage = {
  input_tokens: number;
  output_tokens: number;
  context_window: number;
  session_input: number;
  session_output: number;
};

export type ChatResult = {
  answer: string;
  sources: Source[];
  latency_ms: number;
  confidence: number;
  usage?: Usage;
  web_used?: boolean;
  cost?: string;
};

export type Stats = { documents: number; chunks: number };

export type Doc = { source: string; chunks: number };

export type QuizQuestion = {
  question: string;
  options: Record<string, string>;
  answer: string;
  explanation?: string;
};

async function req(path: string, options?: RequestInit) {
  const res = await fetch(`${BASE}${path}`, options);
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
  health: (): Promise<{ status: string; key_loaded: boolean; models: string[] }> =>
    req("/health"),

  upload: (file: File): Promise<{ filename: string; chunks: number; stats: Stats }> => {
    const form = new FormData();
    form.append("file", file);
    return req("/upload", { method: "POST", body: form });
  },

  youtube: (url: string): Promise<{ chunks: number; stats: Stats }> =>
    req("/youtube", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    }),

  // Smart: sends any link; backend routes YouTube -> transcript, else -> web page.
  link: (url: string): Promise<{ chunks: number; stats: Stats }> =>
    req("/link", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    }),

  chat: (question: string, history: { role: string; content: string }[] = [], pattern = "", source = ""): Promise<ChatResult> =>
    req("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history, pattern, source }),
    }),

  chatStream: async (
    question: string,
    history: { role: string; content: string }[],
    pattern: string,
    source: string,
    onToken: (t: string) => void,
    onNotice: (n: string) => void,
    onDone: (meta: { sources?: Source[]; confidence?: number; latency_ms?: number; web_used?: boolean; usage?: Usage; cost?: string; error?: string }) => void,
  ): Promise<void> => {
    const res = await fetch(`${BASE}/chat-stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history, pattern, source }),
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
          if (obj.token) onToken(obj.token as string);
          if (obj.notice) onNotice(obj.notice as string);
          if (obj.done) onDone(obj);
        } catch {
          // ignore a partial/bad line
        }
      }
    }
  },

  quiz: (num_questions: number, source = ""): Promise<{ questions: QuizQuestion[] }> =>
    req("/quiz", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ num_questions, source }),
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

  exam: (pattern_text: string, instructions: string, reference_text = "", reference_url = ""): Promise<{ paper: string }> =>
    req("/exam", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pattern_text, instructions, reference_text, reference_url }),
    }),

  exportPaper: async (text: string, format: string): Promise<void> => {
    const res = await fetch(`${BASE}/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, format }),
    });
    if (!res.ok) throw new Error("Export failed");
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
