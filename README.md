# Illomra

**Answers from YOUR material — not the internet.**

Illomra is a study & teaching tool for students and teachers, built around
Retrieval-Augmented Generation (RAG):

- **Students** upload their own textbooks, notes, slides, or lecture links, then
  ask questions, get **source-cited answers grounded only in that material**,
  build study notes, and **test themselves** with quizzes that track mistakes
  and re-quiz on weak spots.
- **Teachers** attach a sample/past paper and generate **new exam papers in
  their institute's exact format** — with answer key & marking scheme,
  difficulty mix, total-marks control, and A/B variants — exported to Word/PDF.

That grounding + paper generation is what separates it from a generic
ChatGPT/Claude wrapper: it works from *your* documents, scoped per document,
and shows which source every answer came from.

Full-stack: **FastAPI** backend + **Next.js / React** frontend.

## Project structure
```
studymind/
├── backend/                 # FastAPI API (Python)
│   ├── api.py                  # endpoints, auth, rate limiting, export
│   ├── rag_engine.py           # the RAG brain: retrieval, prompts, Gemini
│   ├── requirements.txt        # dependency intent (loose pins)
│   ├── requirements.lock.txt   # exact pins — install THIS for reproducible builds
│   ├── Dockerfile              # deploy image (HF Spaces / Fly.io ready)
│   ├── tests/                  # pytest suite (no LLM calls needed)
│   ├── .env.example            # template — copy to .env and add your key
│   └── .env                    # GOOGLE_API_KEY (never committed)
└── frontend/                # Next.js + Tailwind app
    ├── app/page.tsx            # the whole UI
    └── lib/api.ts              # typed API client
```

## Tech stack
- **Backend:** FastAPI · LangChain · Google Gemini (`gemini-3.6-flash`, free tier) ·
  ChromaDB · HuggingFace MiniLM embeddings (local, free)
- **Frontend:** Next.js (App Router) · React · TypeScript · Tailwind CSS · lucide-react

## Run locally
**1. Backend**
```bash
cd backend
pip install -r requirements.lock.txt
# copy .env.example to .env and add your GOOGLE_API_KEY
uvicorn api:app --reload --port 8000
```
Backend → http://localhost:8000 (interactive API docs at `/docs`)

**2. Frontend**
```bash
cd frontend
npm install
npm run dev
```
Frontend → http://localhost:3000

**3. Tests**
```bash
cd backend
python -m pytest tests/ -v
```

## Configuration (backend `.env`)
| Variable | Required | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | yes | Gemini API key (free at aistudio.google.com) |
| `PERSIST_DIR` | no | Vector-store folder (default: `backend/chroma_db`; use `/data/chroma_db` on a host with a volume) |
| `CORS_ORIGINS` | no | Comma-separated allowed frontend origins (default `http://localhost:3000`) |
| `APP_ACCESS_TOKEN` | no | Shared access-code mode: every endpoint except `/health` requires `Authorization: Bearer <token>` |
| `GOOGLE_OAUTH_CLIENT_ID` | no | **Google Sign-In mode** (overrides the token): users sign in with Google and each account's documents + chats are fully isolated |

Frontend env (Vercel → Project Settings): `NEXT_PUBLIC_API_URL` (backend URL),
plus `NEXT_PUBLIC_GOOGLE_CLIENT_ID` (same client id) for Google mode, or
`NEXT_PUBLIC_API_TOKEN` (same value as `APP_ACCESS_TOKEN`) for token mode.

### Setting up Google Sign-In (one-time, ~10 min)
1. [console.cloud.google.com](https://console.cloud.google.com) → create/select a project.
2. **APIs & Services → OAuth consent screen** → External → fill app name + your email → add yourself as a test user (or publish).
3. **APIs & Services → Credentials → Create credentials → OAuth client ID → Web application.**
   Authorized JavaScript origins: `http://localhost:3000` and your deployed frontend URL.
4. Copy the Client ID into backend `GOOGLE_OAUTH_CLIENT_ID` and frontend `NEXT_PUBLIC_GOOGLE_CLIENT_ID`. Done — the app switches to Google auth automatically.

## Deploy (pilot-grade, ~$5/mo)
1. **Backend → Hugging Face Spaces** (Docker Space, free 16GB RAM; +$5/mo
   persistent storage mounted at `/data`). Push `backend/` with the Dockerfile;
   set secrets `GOOGLE_API_KEY`, `APP_ACCESS_TOKEN`, `CORS_ORIGINS`,
   `PERSIST_DIR=/data/chroma_db`, `HF_HOME=/data/hf_cache`.
2. **Frontend → Vercel** (free). Import `frontend/`, set `NEXT_PUBLIC_API_URL`
   to the Space URL and `NEXT_PUBLIC_API_TOKEN`.
3. Ping `/health` with UptimeRobot (free) — doubles as keep-awake and outage alert.
4. When cold starts hurt, move the same Docker image to Fly.io (2GB + volume ≈ $11/mo).

**Run exactly ONE uvicorn worker** — Chroma's SQLite is single-writer and the
engine is a single in-process object. Scale vertically.

## Gemini free-tier limits (worth knowing)
Uploading documents costs **zero** quota (embeddings are local). Only chat /
quiz / notes / paper generation call Gemini: roughly 10-15 requests/min and a
few hundred/day on the free tier. The app queues (max 2 concurrent), retries
per-minute limits automatically during streaming, and fails fast with a
friendly message otherwise. For a classroom, enable billing (~$0.002/request)
or give each academy its own key.

## Security posture
- Optional bearer-token auth + per-IP rate limiting (30 req/min)
- SSRF guard on link ingestion (private/internal addresses refused)
- 20MB upload cap; prompt-injection fencing on all retrieved material
- Raw errors stay in server logs; clients get generic messages
- **Multi-user ready via Google Sign-In** — with `GOOGLE_OAUTH_CLIENT_ID` set,
  ID tokens are verified server-side on every request and ALL data access
  (documents, retrieval, chats, reset) is fenced to the signed-in account.
  Without it, the app runs single-tenant (access-code or open local dev).
