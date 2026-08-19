# StudyMind

An AI study assistant: upload documents, YouTube lectures, or web links, then **chat with them**, and auto-generate **quizzes, flashcards, and summaries** — powered by Retrieval-Augmented Generation (RAG).

Full-stack build: **FastAPI** backend + **Next.js / React** frontend.

## Project structure
```
studymind/
├── backend/              # FastAPI API (Python) — the server
│   ├── api.py               # API endpoints: /chat, /upload, /quiz, /summary, ...
│   ├── rag_engine.py        # the RAG "brain": Groq LLM, ChromaDB, HF embeddings
│   ├── requirements.txt     # Python dependencies
│   ├── .env.example         # template — copy to .env and add your key
│   └── .env                 # your GROQ_API_KEY (never committed)
└── frontend/            # Next.js + Tailwind app — the UI (the "wow")
```

## Tech stack
- **Backend:** FastAPI · Uvicorn · LangChain · Groq (LLM) · ChromaDB (vector store) · HuggingFace embeddings
- **Frontend:** Next.js (App Router) · React · TypeScript · Tailwind CSS

## Run locally
**1. Backend**
```bash
cd backend
pip install -r requirements.txt
# copy .env.example to .env and add your Groq key
uvicorn api:app --reload --port 8000
```
Backend → http://localhost:8000  (interactive API docs at `/docs`)

**2. Frontend**
```bash
cd frontend
npm install
npm run dev
```
Frontend → http://localhost:3000

## Roadmap
- [x] FastAPI backend wrapping the RAG engine
- [ ] Next.js frontend — chat, upload, URL, quiz, flashcards, summary
- [ ] Streaming answers (typewriter)
- [ ] Any-file + any-web-link ingestion
- [ ] Deploy: backend (Render/Railway) + frontend (Vercel)
