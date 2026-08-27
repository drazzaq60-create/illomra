// lib/api.ts — the single place the frontend talks to the FastAPI backend.
// Every function here maps to one endpoint in backend/api.py.

const BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Access token for deployed instances (matches backend APP_ACCESS_TOKEN).
// Priority: what the user typed into the lock screen (localStorage) > build-time env.
function getToken(): string {
  try {
    const t = localStorage.getItem("illomra_token");
    if (t) return t;
  } catch {
    // SSR / storage blocked — fall through
  }
  return process.env.NEXT_PUBLIC_API_TOKEN || "";
}

export function setToken(t: string) {
  try {
    localStorage.setItem("illomra_token", t);
  } catch {
    // storage blocked — token will only last this page load
  }
}

export class AuthError extends Error {}

function authHeaders(extra: Record<string, string> = {}): Record<string, string> {
  const token = getToken();
  return token ? { ...extra, Authorization: `Bearer ${token}` } : extra;
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
  model?: string; // which model answered (fallback chain)
};

export type QuotaModel = {
  model: string;
  used_today: number;
  daily_limit: number;
  status: "ok" | "cooldown" | "exhausted";
  retry_in_s: number;
};

export type QuotaSnapshot = {
  primary: string;
  models: QuotaModel[];
  totals: { used_today: number; capacity: number; resets_in_s: number };
};

// Visual spec of an attached format sample (from a photo) — applied at export.
export type PaperLayout = {
  font?: string;             // "serif" | "sans"
  center_top_lines?: number; // how many opening lines are a centered title block
  footer?: string;
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
    if (res.status === 401) throw new AuthError(detail);
    throw new Error(detail);
  }
  return res.json();
}

export function clearToken() {
  try {
    localStorage.removeItem("illomra_token");
  } catch {
    // storage blocked
  }
}

export const api = {
  health: (): Promise<{ status: string; auth?: "google" | "token" | "open"; documents?: number; chunks?: number }> =>
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
    params: { question: string; history: { role: string; content: string }[]; pattern: string; source: string; refs?: string[] },
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
        sources: params.refs ?? [],
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

  quiz: (num_questions: number, source = "", topic = "", avoid: string[] = []): Promise<{ questions: QuizQuestion[] }> =>
    req("/quiz", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ num_questions, source, topic, avoid }),
    }),

  flashcards: (num_cards: number, source = ""): Promise<{ cards: { front: string; back: string }[] }> =>
    req("/flashcards", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ num_cards, source }),
    }),

  summary: (source = ""): Promise<{ summary: string }> =>
    req("/summary", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source }),
    }),

  stats: (): Promise<Stats> => req("/stats"),

  quota: (): Promise<QuotaSnapshot> => req("/usage"),

  // Server-side conversation store (chats follow the workspace, not one browser).
  getConvos: (): Promise<{ convos: unknown[]; current: string }> => req("/convos"),

  putConvos: (convos: unknown[], current: string): Promise<{ status: string }> =>
    req("/convos", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ convos, current }),
    }),

  documents: (): Promise<{ documents: Doc[] }> => req("/documents"),

  deleteDocument: (source: string): Promise<{ status: string; stats: Stats }> =>
    req("/documents/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source }),
    }),

  reset: (): Promise<{ status: string }> => req("/reset", { method: "POST" }),

  extract: (file: File): Promise<{ text: string; pattern_id?: string; layout?: PaperLayout }> => {
    const form = new FormData();
    form.append("file", file);
    return req("/extract", { method: "POST", body: form });
  },

  exportPaper: async (text: string, format: string, patternId = "", layout: PaperLayout | undefined = undefined): Promise<void> => {
    const res = await fetch(`${BASE}/export`, {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ text, format, pattern_id: patternId, layout: layout ?? {} }),
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
    link.download = `illomra-paper.${format}`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(link.href);
  },
};
