"""
rag_engine.py — StudyMind RAG Engine

The "brain": loads documents, splits them into chunks, embeds them into a
Chroma vector store, and answers questions grounded in ONLY the material the
student uploaded.

Design notes (post-audit, why the code looks the way it does):
- One retrieval helper feeds BOTH query() and query_stream() so the streaming
  path (the one the UI uses) can never silently drift from the tested path.
- Retrieval can be SCOPED to a single document (source) so a question about one
  file never drags in chunks from an unrelated one — the #1 cause of messy answers.
- Retrieved chunks are pruned by a relevance threshold, then labeled with their
  source in the prompt, so the model knows which excerpt came from where.
- Web search is EXPLICIT-only: it fires only when the student clearly asks to go
  online, never accidentally on words like "research" or "recent".
"""

import os
import re
import tempfile
import time
from typing import Optional, List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

from youtube_transcript_api import YouTubeTranscriptApi

PERSIST_DIR = "./chroma_db"

# --- Retrieval tuning ---------------------------------------------------------
# Chunks are kept small so they fit MiniLM's 256-token window (a 1000-char chunk
# gets its tail silently truncated and becomes unsearchable).
CHUNK_SIZE = 550
CHUNK_OVERLAP = 90
# Cosine relevance below this is treated as noise, EXCEPT we always keep the top
# few so a question never comes back completely empty.
RELEVANCE_FLOOR = 0.2
ALWAYS_KEEP = 4

AVAILABLE_MODELS = {
    "gemini-3.6-flash": "⚡ Gemini 3.6 Flash (1M context)",
}


def extract_video_id(url: str) -> Optional[str]:
    """Extract the 11-char YouTube video ID from any common URL shape."""
    pattern = r'(?:https?://)?(?:www\.)?(?:youtube\.com/(?:[^/]+/.+/|(?:v|e(?:mbed)?)/|.*[?&]v=)|youtu\.be/)([^"&?/\s]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None


class RAGEngine:
    def __init__(self, api_key: str = "", model_name: str = "gemini-3.6-flash"):
        self.api_key = api_key
        self.model_name = model_name
        self.vector_store: Optional[Chroma] = None
        self.doc_count = 0
        self.chunk_count = 0
        self.source_counts: Dict[str, int] = {}

        # Telemetry
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._llm_cache: Dict[int, Any] = {}

        # Free local embeddings (sentence-transformers, runs on CPU).
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

        if os.path.exists(PERSIST_DIR):
            self._reload_store()

    # ---- LLM ----------------------------------------------------------------
    def _get_llm(self, max_tokens: int = 8000):
        # Cache one client per output-size so we don't rebuild it on every call.
        llm = self._llm_cache.get(max_tokens)
        if llm is None:
            llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key or os.getenv("GOOGLE_API_KEY", ""),
                temperature=0.2,
                max_output_tokens=max_tokens,
            )
            self._llm_cache[max_tokens] = llm
        return llm

    @staticmethod
    def _is_rate_limit(e) -> bool:
        s = str(e).lower()
        return "429" in s or "resource_exhausted" in s or "quota" in s or "rate limit" in s

    # Free tier caps requests-per-minute; wait it out instead of failing the request.
    RETRY_WAITS = (20, 40)

    def _invoke_with_retry(self, prompt_text: str, max_tokens: int = 8000):
        """Non-streaming call with automatic backoff on the free-tier per-minute limit."""
        import time as _t
        attempts = (0,) + self.RETRY_WAITS
        last = None
        for i, wait in enumerate(attempts):
            if wait:
                _t.sleep(wait)
            try:
                return self._get_llm(max_tokens).invoke([HumanMessage(content=prompt_text)])
            except Exception as e:
                last = e
                if self._is_rate_limit(e) and i < len(attempts) - 1:
                    continue
                raise
        raise last  # pragma: no cover

    @staticmethod
    def _content_text(content) -> str:
        """Gemini 3.x can return content as a list of blocks; flatten to text."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for c in content:
                if isinstance(c, dict):
                    parts.append(c.get("text", ""))
                elif isinstance(c, str):
                    parts.append(c)
            return "".join(parts)
        return str(content)

    # ---- Store bookkeeping --------------------------------------------------
    def _reload_store(self):
        """Reconnect to the persisted store and recount documents/chunks per source."""
        try:
            self.vector_store = Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=self.embeddings,
                collection_metadata={"hnsw:space": "cosine"},
            )
            counts: Dict[str, int] = {}
            results = self.vector_store.get()
            for meta in results.get("metadatas", []) or []:
                if not meta:
                    continue
                src = meta.get("source", "?")
                counts[src] = counts.get(src, 0) + 1
            self.source_counts = counts
            self.doc_count = len(counts)
            self.chunk_count = sum(counts.values())
        except Exception:
            self.vector_store = None
            self.source_counts = {}
            self.doc_count = 0
            self.chunk_count = 0

    def update_model(self, model_name: str):
        self.model_name = model_name

    def get_stats(self) -> Dict[str, int]:
        return {"documents": self.doc_count, "chunks": self.chunk_count}

    def list_documents(self) -> List[Dict[str, Any]]:
        """Every indexed document with its chunk count — powers the UI doc manager."""
        return [{"source": s, "chunks": n} for s, n in sorted(self.source_counts.items())]

    def delete_document(self, source: str) -> Dict[str, int]:
        """Remove one document's chunks from the store (used for re-upload + manual delete)."""
        if self.vector_store is not None:
            try:
                self.vector_store._collection.delete(where={"source": source})
            except Exception:
                pass
            self._reload_store()
        return self.get_stats()

    def _index(self, chunks: List[Document]) -> int:
        """Create the store on first use, or add to it. Chunks must already carry a source."""
        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(
                chunks,
                self.embeddings,
                persist_directory=PERSIST_DIR,
                collection_metadata={"hnsw:space": "cosine"},
            )
        else:
            self.vector_store.add_documents(chunks)
        self._reload_store()
        return len(chunks)

    # ---- Loaders ------------------------------------------------------------
    @staticmethod
    def _load_any(path: str, ext: str):
        """Load many file types into LangChain Documents (text extraction)."""
        if ext == ".pdf":
            return PyPDFLoader(path).load()
        if ext in (".txt", ".md"):
            return TextLoader(path, encoding="utf-8").load()
        if ext == ".csv":
            from langchain_community.document_loaders import CSVLoader
            return CSVLoader(path).load()
        if ext == ".docx":
            try:
                import docx2txt
            except ImportError:
                raise Exception("Reading .docx needs the 'docx2txt' package (pip install docx2txt).")
            text = docx2txt.process(path) or ""
            if not text.strip():
                raise Exception("No text found in that Word document.")
            return [Document(page_content=text, metadata={"page": "Document"})]
        if ext == ".pptx":
            try:
                from pptx import Presentation
            except ImportError:
                raise Exception("Reading .pptx needs the 'python-pptx' package (pip install python-pptx).")
            prs = Presentation(path)
            parts = []
            for i, slide in enumerate(prs.slides, 1):
                texts = [sh.text for sh in slide.shapes if sh.has_text_frame and sh.text.strip()]
                if texts:
                    parts.append(f"[Slide {i}]\n" + "\n".join(texts))
            text = "\n\n".join(parts)
            if not text.strip():
                raise Exception("No text found in that PowerPoint.")
            return [Document(page_content=text, metadata={"page": "Slides"})]
        raise Exception(f"Unsupported file type '{ext}'. Try PDF, Word, PowerPoint, TXT, MD, or CSV.")

    def process_file(self, uploaded_file) -> int:
        ext = "." + uploaded_file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        try:
            docs = self._load_any(tmp_path, ext)
            for d in docs:
                d.metadata["source"] = uploaded_file.name
            chunks = self.splitter.split_documents(docs)
            if not chunks:
                raise Exception("No readable text found in that file.")
            # Re-uploading the same file replaces its old chunks instead of duplicating them.
            self.delete_document(uploaded_file.name)
            return self._index(chunks)
        finally:
            os.unlink(tmp_path)

    def process_youtube_url(self, url: str) -> int:
        video_id = extract_video_id(url)
        if not video_id:
            raise ValueError("Could not extract a valid 11-character video ID from YouTube link.")
        try:
            transcript_list = YouTubeTranscriptApi().fetch(video_id)
            # Plain transcript text — timestamps are dropped so they don't pollute embeddings.
            full_text = " ".join(entry.text.strip() for entry in transcript_list if entry.text.strip())
            if not full_text.strip():
                raise Exception("The transcript was empty.")
            source = f"🎥 YouTube ({video_id})"
            doc = Document(page_content=full_text, metadata={"source": source, "page": "Video transcript"})
            chunks = self.splitter.split_documents([doc])
            self.delete_document(source)
            return self._index(chunks)
        except Exception as e:
            raise Exception(f"Failed to fetch YouTube transcript: {str(e)}")

    def process_url(self, url: str) -> int:
        try:
            from langchain_community.document_loaders import WebBaseLoader
        except ImportError:
            raise Exception("Reading web links needs 'beautifulsoup4' (pip install beautifulsoup4).")
        loader = WebBaseLoader(url)
        loader.requests_kwargs = {
            "headers": {"User-Agent": "Mozilla/5.0 (compatible; StudyMind/1.0)"},
            "timeout": 20,
        }
        try:
            docs = loader.load()
        except Exception as e:
            raise Exception(f"Couldn't open that page: {e}")
        docs = [d for d in docs if d.page_content and d.page_content.strip()]
        if not docs:
            raise Exception("No readable text found — the page may block bots or need a login.")
        for d in docs:
            d.metadata["source"] = url
            d.metadata.setdefault("page", "Web page")
        chunks = self.splitter.split_documents(docs)
        if not chunks:
            raise Exception("No readable text found on that page.")
        self.delete_document(url)
        return self._index(chunks)

    # ---- Web search (EXPLICIT only) ----------------------------------------
    def web_search(self, query: str, n: int = 5):
        """Free live web search via DuckDuckGo (no API key needed)."""
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
            out = []
            with DDGS() as d:
                for r in d.text(query, max_results=n):
                    out.append({
                        "title": r.get("title", ""),
                        "snippet": r.get("body", "") or r.get("snippet", ""),
                        "url": r.get("href", "") or r.get("url", ""),
                    })
            return out
        except Exception:
            return []

    # Explicit phrases ONLY — never fire on ordinary study words like "research"
    # (which contains "search") or "recent". The student has to actually ask.
    WEB_TRIGGERS = (
        "search the web", "search online", "search the internet", "web search",
        "look online", "look it up online", "look on the web", "on the internet",
        "browse the web", "google it", "google for", "latest news",
    )

    def _decide_web_search(self, question: str) -> Optional[str]:
        q = question.lower()
        if any(t in q for t in self.WEB_TRIGGERS):
            return question.strip()[:200]
        return None

    # ---- Shared retrieval ---------------------------------------------------
    ANAPHORA = ("that", "this", "these", "those", "it", "them", "more",
                "again", "further", "above", "previous", "same")

    def _history_block(self, history: Optional[List[Dict[str, str]]]) -> str:
        history = history or []
        recent = history[-6:]
        if not recent:
            return ""
        lines = []
        for m in recent:
            who = "Student" if m.get("role") == "user" else "You"
            lines.append(f"{who}: {m.get('content', '')}")
        return "Conversation so far:\n" + "\n".join(lines) + "\n\n"

    def _retrieval_query(self, question: str, history: Optional[List[Dict[str, str]]]) -> str:
        """Only fold the previous question in when the follow-up is clearly anaphoric
        ('explain that more'), so a genuine new short question isn't dragged off-topic."""
        words = question.lower().split()
        if len(words) <= 6 and any(w.strip("?.,") in self.ANAPHORA for w in words):
            history = history or []
            prev = next((m.get("content", "") for m in reversed(history) if m.get("role") == "user"), "")
            if prev:
                return f"{prev} {question}"
        return question

    def _effective_k(self, question: str, base_k: int, source: str) -> int:
        broad_terms = ["explain everything", "explain the whole", "explain the entire",
                       "summarize everything", "summarise everything", "the whole thing",
                       "overview of", "comprehensive", "cover all", "all the concepts",
                       "all topics", "everything about"]
        ql = question.lower()
        is_broad = any(t in ql for t in broad_terms)
        if is_broad:
            # Widen — but only within the scoped document, never across the whole mixed store.
            pool = self.source_counts.get(source) if source else self.chunk_count
            return min(pool or 0, 18) if source else min(self.chunk_count, 12)
        return base_k

    def _response_tier(self, question: str, pattern: str):
        """Match answer length + retrieval width to the task, so small questions stay fast
        and 'explain in detail' still gets the long treatment.
        Returns (max_output_tokens, retrieval_k, length_guidance)."""
        if pattern and pattern.strip():
            return 8000, 6, "Produce the FULL document the student asked for."
        ql = question.lower().strip()
        n_words = len(ql.split())
        # Explicit multi-word phrases → low false-positive, safe to match anywhere.
        long_signals = ("explain everything", "explain the whole", "explain the entire",
                        "overview", "comprehensive", "in detail", "detailed", "elaborate",
                        "step by step", "in depth", "thoroughly", "everything about",
                        "full derivation", "essay", "long answer", "walk me through")
        brevity_signals = ("briefly", "in short", "in one line", "one line", "short answer",
                           "tl;dr", "quickly", "in a sentence")
        # Question-word openers → only trust them at the START (so "when" mid-sentence
        # in a complex question doesn't wrongly mark it short).
        short_starts = ("what is", "what are", "what does", "define", "definition of",
                        "who is", "who are", "when ", "where ", "which ", "name the",
                        "list ", "is ", "are ", "does ", "do ", "can ")
        if any(s in ql for s in long_signals):
            return 8000, 8, ("Be thorough and well-structured — cover the main concepts with "
                             "short headings, bullet points, and concrete examples.")
        is_short = (
            any(s in ql for s in brevity_signals)
            or n_words <= 6
            or (any(ql.startswith(s) for s in short_starts) and n_words <= 12)
        )
        if is_short:
            return 1200, 4, ("Answer directly and concisely — a focused explanation without long "
                             "preamble or exhaustive detail. No headings for a short answer.")
        return 2500, 6, ("Match the length to the question — concise by default; expand only where "
                         "it genuinely aids understanding.")

    def _gather(self, retrieval_query: str, effective_k: int, source: str):
        """One retrieval call → (docs kept after threshold, mean-confidence 0-100).
        Both query() and query_stream() use this, so they can't diverge."""
        if not self.vector_store or effective_k <= 0:
            return [], 0
        kwargs = {"k": effective_k}
        if source:
            kwargs["filter"] = {"source": source}
        scored = self.vector_store.similarity_search_with_relevance_scores(retrieval_query, **kwargs)
        if not scored:
            return [], 0
        kept = [(doc, sc) for i, (doc, sc) in enumerate(scored) if i < ALWAYS_KEEP or sc >= RELEVANCE_FLOOR]
        docs = [d for d, _ in kept]
        used = [sc for _, sc in kept]
        confidence = int(max(0.0, min(1.0, sum(used) / len(used))) * 100) if used else 0
        return docs, confidence

    @staticmethod
    def _label_context(docs: List[Document]) -> str:
        """Prefix each chunk with its source so the model knows which file it came from."""
        blocks = []
        for d in docs:
            src = d.metadata.get("source", "?")
            page = d.metadata.get("page", "")
            tag = f"[From: {src}" + (f" · {page}]" if page else "]")
            blocks.append(f"{tag}\n{d.page_content}")
        return "\n\n".join(blocks)

    @staticmethod
    def _sources_payload(docs: List[Document]) -> List[Dict[str, Any]]:
        out, seen = [], set()
        for doc in docs:
            key = (doc.metadata.get("source"), doc.metadata.get("page"))
            if key in seen:
                continue
            seen.add(key)
            c = doc.page_content
            out.append({
                "content": c[:400] + ("..." if len(c) > 400 else ""),
                "source": doc.metadata.get("source", "?"),
                "page": doc.metadata.get("page", "N/A"),
            })
        return out

    def _build_prompt(self, question: str, history_block: str, context: str,
                      pattern: str, web_block: str, length_guidance: str = "") -> str:
        """Two clean, separate prompts: a tutor for chat, a generator for papers."""
        if pattern and pattern.strip():
            return (
                "You are StudyMind's document generator. Produce the FULL finished document the "
                "student asks for, ready to download.\n"
                "- Follow the student's request exactly (e.g. 'reproduce this, change the name to Ali' "
                "means output the reference almost verbatim with only that change).\n"
                "- Mirror the FORMAT SAMPLE's structure, sections, question types, marks, and style.\n"
                "- Draw the content from the reference excerpts below. Output clean markdown.\n\n"
                f"{history_block}"
                f"FORMAT SAMPLE to mirror:\n{pattern.strip()[:15000]}\n\n"
                f"Reference excerpts:\n{context}\n"
                f"{web_block}\n"
                f"Request: {question}\n\nDocument:"
            )
        web_rule = (
            "- Some excerpts are labelled [WEB]. Treat them as supplementary, prefer the student's "
            "own material, and mention the source URL when you use a web fact.\n" if web_block else ""
        )
        if not context.strip():
            # No documents indexed → general assistant mode (answer from own knowledge).
            return (
                "You are StudyMind, a friendly, knowledgeable tutor and study assistant.\n\n"
                "How to answer:\n"
                f"- Length: {length_guidance}\n"
                "- Explain clearly for a beginner, in plain language, with a concrete example where it helps.\n"
                "- Answer from your general knowledge. If you're unsure or it's outside what you know, say so honestly rather than inventing facts.\n"
                f"{web_rule}\n"
                f"{history_block}"
                f"{web_block}\n"
                f"Question: {question}\n\nAnswer:"
            )
        return (
            "You are StudyMind, an expert tutor helping a student learn from THEIR OWN uploaded material.\n\n"
            "How to answer:\n"
            "- Ground every claim in the excerpts below. If the answer isn't there, say so plainly "
            "(\"I couldn't find that in your material\") instead of inventing facts.\n"
            "- Each excerpt is tagged [From: <document>]. Excerpts may come from different documents — "
            "use only the ones relevant to the question and ignore the rest.\n"
            f"- Length: {length_guidance}\n"
            "- Explain clearly for a beginner, in plain language.\n"
            "- Use the conversation so far to resolve follow-ups like \"explain that more\".\n"
            f"{web_rule}\n"
            f"{history_block}"
            f"Excerpts from the student's material:\n{context}\n"
            f"{web_block}\n"
            f"Question: {question}\n\nAnswer:"
        )

    def _prepare(self, question, history, k, source, pattern):
        """Everything both query paths need before calling the LLM.
        Returns (prompt_text, sources, confidence, web_used, max_tokens)."""
        max_tokens, base_k, length_guidance = self._response_tier(question, pattern)
        history_block = self._history_block(history)
        retrieval_query = self._retrieval_query(question, history)
        effective_k = self._effective_k(question, base_k, source)
        docs, confidence = self._gather(retrieval_query, effective_k, source)
        context = self._label_context(docs)

        web_results, web_sources, web_block = [], [], ""
        wq = self._decide_web_search(question)
        if wq:
            web_results = self.web_search(wq, n=5)
            if web_results:
                web_text = "\n".join(f"- {r['title']}: {r['snippet']} ({r['url']})" for r in web_results)
                web_block = f"\n[WEB] Live search results for \"{wq}\":\n{web_text}\n"
                web_sources = [
                    {"content": (r.get("snippet") or "")[:300],
                     "source": "🌐 " + (r.get("title") or "web result"),
                     "page": r.get("url", "")}
                    for r in web_results
                ]

        prompt_text = self._build_prompt(question, history_block, context, pattern, web_block, length_guidance)
        sources = self._sources_payload(docs) + web_sources
        return prompt_text, sources, confidence, bool(web_results), max_tokens

    def _usage_payload(self, in_tok: int, out_tok: int) -> Dict[str, Any]:
        self.total_input_tokens += in_tok
        self.total_output_tokens += out_tok
        return {
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "context_window": 1_000_000,
            "session_input": self.total_input_tokens,
            "session_output": self.total_output_tokens,
        }

    # ---- Public query paths -------------------------------------------------
    def query(self, question: str, history=None, search_type: str = "Similarity Search",
              k: int = 5, pattern: str = "", source: str = "") -> Dict[str, Any]:
        # No documents is fine — falls through to general-assistant mode in _build_prompt.
        start = time.time()
        prompt_text, sources, confidence, web_used, max_tokens = self._prepare(question, history, k, source, pattern)

        response = self._invoke_with_retry(prompt_text, max_tokens)
        answer = self._content_text(getattr(response, "content", response))
        usage = getattr(response, "usage_metadata", None) or {}
        usage_payload = self._usage_payload(int(usage.get("input_tokens", 0) or 0),
                                            int(usage.get("output_tokens", 0) or 0))
        return {
            "answer": answer,
            "sources": sources,
            "web_used": web_used,
            "latency_ms": int((time.time() - start) * 1000),
            "confidence": confidence,
            "usage": usage_payload,
            "cost": "free",
        }

    def query_stream(self, question: str, history=None, k: int = 5, pattern: str = "", source: str = ""):
        """Streaming twin of query() — shares the SAME retrieval via _prepare().
        Works with no documents too (general-assistant mode)."""
        start = time.time()
        prompt_text, sources, confidence, web_used, max_tokens = self._prepare(question, history, k, source, pattern)

        import time as _t
        in_tok, out_tok = 0, 0
        attempts = (0,) + self.RETRY_WAITS
        streamed = False
        for i, wait in enumerate(attempts):
            if wait:
                # Tell the UI we're pausing for the free-tier limit (so it doesn't look frozen).
                yield {"notice": f"⏳ Free-tier limit reached — waiting {wait}s and retrying…"}
                _t.sleep(wait)
            try:
                for chunk in self._get_llm(max_tokens).stream([HumanMessage(content=prompt_text)]):
                    txt = self._content_text(getattr(chunk, "content", ""))
                    if txt:
                        streamed = True
                        yield {"token": txt}
                    um = getattr(chunk, "usage_metadata", None)
                    if um:
                        in_tok = int(um.get("input_tokens", 0) or in_tok)
                        out_tok = int(um.get("output_tokens", 0) or out_tok)
                break  # streamed to completion
            except Exception as e:
                # Retry only if we've emitted nothing yet — can't un-send tokens mid-stream.
                if self._is_rate_limit(e) and not streamed and i < len(attempts) - 1:
                    continue
                raise
        usage_payload = self._usage_payload(in_tok, out_tok)
        yield {
            "done": True,
            "sources": sources,
            "confidence": confidence,
            "web_used": web_used,
            "latency_ms": int((time.time() - start) * 1000),
            "usage": usage_payload,
            "cost": "free",
        }

    # ---- Quiz / summary -----------------------------------------------------
    def _sample_context(self, k: int = 10, source: str = "") -> str:
        if not self.vector_store:
            return ""
        kwargs = {"k": k}
        if source:
            kwargs["filter"] = {"source": source}
        docs = self.vector_store.similarity_search("key concepts main topics overview", **kwargs)
        return self._label_context(docs)

    def generate_quiz(self, num_questions: int = 5, source: str = "") -> List[dict]:
        if not self.vector_store:
            return []
        context = self._sample_context(k=8, source=source)
        prompt = f"""Generate exactly {num_questions} MCQ questions from this content.
Use ONLY the content below — do NOT use outside knowledge.

Content:
{context}

Format each question EXACTLY like this:

Q1. [Question text]
A) [Option]
B) [Option]
C) [Option]
D) [Option]
Answer: A
Explanation: [One sentence why]

Q2. ...and so on"""
        response = self._invoke_with_retry(prompt)
        raw = self._content_text(response.content)
        usage = getattr(response, "usage_metadata", None) or {}
        self._usage_payload(int(usage.get("input_tokens", 0) or 0), int(usage.get("output_tokens", 0) or 0))
        return self._parse_quiz(raw)

    def _parse_quiz(self, raw: str) -> List[dict]:
        questions = []
        for block in re.split(r'Q\d+\.', raw)[1:]:
            try:
                lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
                q, opts, ans, exp = lines[0], {}, "", ""
                for ln in lines[1:]:
                    if ln.startswith("A)"):  opts["A"] = ln[2:].strip()
                    elif ln.startswith("B)"): opts["B"] = ln[2:].strip()
                    elif ln.startswith("C)"): opts["C"] = ln[2:].strip()
                    elif ln.startswith("D)"): opts["D"] = ln[2:].strip()
                    elif ln.lower().startswith("answer:"):
                        ans = ln.split(":")[-1].strip().upper()[:1]
                    elif ln.lower().startswith("explanation:"):
                        exp = ln.split(":", 1)[-1].strip()
                if q and len(opts) == 4 and ans in "ABCD":
                    questions.append({"question": q, "options": opts, "answer": ans, "explanation": exp})
            except Exception:
                continue
        return questions

    def summarize(self, source: str = "") -> Dict[str, Any]:
        if not self.vector_store:
            return {}
        context = self._sample_context(k=10, source=source)
        prompt = f"""Analyze this content and produce a structured study summary.

Content:
{context}

Write in this exact format:

## 📌 Overview
[3-4 sentences about what this document covers]

## 🔑 Key Concepts
- [Concept]: [Brief explanation]
- [Concept]: [Brief explanation]
- [Concept]: [Brief explanation]
- [Concept]: [Brief explanation]
- [Concept]: [Brief explanation]

## 📖 Important Definitions
- [Term]: [Definition]
- [Term]: [Definition]

## 🚀 Key Takeaways
- [Takeaway]
- [Takeaway]
"""
        response = self._invoke_with_retry(prompt)
        raw = self._content_text(response.content)
        usage = getattr(response, "usage_metadata", None) or {}
        self._usage_payload(int(usage.get("input_tokens", 0) or 0), int(usage.get("output_tokens", 0) or 0))
        return {"summary": raw, "cost": "free"}

    # ---- Document generator (paper mode / explicit generator) ---------------
    def _fetch_url_text(self, url: str) -> str:
        from langchain_community.document_loaders import WebBaseLoader
        loader = WebBaseLoader(url)
        loader.requests_kwargs = {"headers": {"User-Agent": "Mozilla/5.0 (StudyMind)"}, "timeout": 20}
        docs = loader.load()
        return "\n\n".join(d.page_content for d in docs if d.page_content)

    def generate_exam(self, pattern_text: str = "", instructions: str = "",
                      reference_text: str = "", reference_url: str = "", source: str = "") -> Dict[str, Any]:
        parts = []
        if (reference_text or "").strip():
            parts.append(reference_text.strip()[:15000])
        if (reference_url or "").strip():
            try:
                parts.append(self._fetch_url_text(reference_url.strip())[:15000])
            except Exception:
                pass
        if not parts and self.vector_store:
            parts.append(self._sample_context(k=min(self.chunk_count or 30, 40), source=source))
        context = "\n\n".join(parts)
        if not context.strip():
            return {"paper": "⚠️ Add reference material — a document, a link, or your knowledge base — first."}
        sample = (pattern_text or "").strip()[:12000]
        want = instructions.strip() or "Create a new exam paper from the reference material, matching the sample's format if one is provided."
        sample_block = sample if sample else "(none provided)"
        prompt = f"""You are StudyMind's document generator. Do EXACTLY what the user's INSTRUCTIONS ask, using the REFERENCE MATERIAL as the source content and (if given) the SAMPLE as the format/style to copy.

USER INSTRUCTIONS — this is the priority; follow it precisely:
{want}

SAMPLE / FORMAT TO COPY (optional):
{sample_block}

REFERENCE MATERIAL (the source content):
{context}

Rules:
- Obey the USER INSTRUCTIONS above all else. If they ask to reproduce the reference with a small change (e.g. change a name, title, or date), output the reference almost verbatim with ONLY that change — do NOT turn it into an exam paper. If they ask for an exam paper, produce one that mirrors the sample's pattern (sections, question types, marks). Otherwise, just do exactly what they asked.
- If a SAMPLE format is provided, copy its structure, sections, and style.
- Output clean, well-formatted markdown, ready to download."""
        response = self._invoke_with_retry(prompt)
        paper = self._content_text(response.content)
        usage = getattr(response, "usage_metadata", None) or {}
        self._usage_payload(int(usage.get("input_tokens", 0) or 0), int(usage.get("output_tokens", 0) or 0))
        return {"paper": paper}

    def reset(self):
        # Clear the data through Chroma's API first. On Windows the SQLite/HNSW
        # files stay locked by this process, so deleting the folder outright fails
        # (WinError 32) — dropping the collection is the reliable way to empty it.
        if self.vector_store is not None:
            try:
                self.vector_store.delete_collection()
            except Exception:
                pass
        self.vector_store = None
        self.doc_count = 0
        self.chunk_count = 0
        self.source_counts = {}
        # Best-effort folder cleanup; ignore if the files are still locked.
        if os.path.exists(PERSIST_DIR):
            import shutil
            try:
                shutil.rmtree(PERSIST_DIR)
            except Exception:
                pass
