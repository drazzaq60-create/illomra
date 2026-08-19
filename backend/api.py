"""
api.py — Illomra backend (FastAPI).

This is the API layer. It reuses the engine (rag_engine.py) and exposes it
over HTTP so a Next.js/React frontend can call it.

Run it:
    uvicorn api:app --reload --port 8000
Then open http://localhost:8000/docs to try every endpoint in the browser.

Hardening (client delivery):
- Optional bearer-token auth (set APP_ACCESS_TOKEN) on everything except /health.
- Per-IP rate limit (30 req/min) + request logging middleware.
- 20 MB upload cap, SSRF guard on /link (in the engine), generic error details
  (raw exceptions go to the server log, never to the client).
"""

import json
import logging
import os
import secrets
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# Load the .env sitting right next to this file BEFORE importing the engine, so
# env-driven engine config (PERSIST_DIR) sees the values no matter which folder
# uvicorn is launched from.
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from fastapi import (APIRouter, Depends, FastAPI, File, Header, HTTPException,
                     Request, Response, UploadFile)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from rag_engine import RAGEngine, extract_video_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("studymind")

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
UPLOAD_TOO_BIG = "File is over 20 MB — split it or upload a smaller file."

# Uploaded .docx format samples are kept here so /export can clone the ORIGINAL
# file (real header/footer/fonts/margins) instead of generating a generic one.
# On a hosted box point this at the persistent volume (e.g. /data/pattern_files).
PATTERN_DIR = os.getenv(
    "PATTERN_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "pattern_files"),
)
os.makedirs(PATTERN_DIR, exist_ok=True)

# If set (non-empty), every endpoint except /health requires
# "Authorization: Bearer <token>". Unset = local dev, no auth.
APP_ACCESS_TOKEN = os.getenv("APP_ACCESS_TOKEN", "").strip()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the embedding model at boot so the FIRST question isn't slow.
    try:
        get_engine()
    except Exception:
        log.exception("Engine warm-up failed — will retry lazily on first request")
    yield


app = FastAPI(title="Illomra API", version="0.2.0", lifespan=lifespan)


# ---- Middleware --------------------------------------------------------------
# Order matters: CORS is added LAST so it wraps everything (even 429s from the
# rate limiter get CORS headers); the access log wraps the rate limiter so
# rate-limited requests still show up in the log.

RATE_LIMIT_MAX = 30       # requests ...
RATE_LIMIT_WINDOW = 60.0  # ... per this many seconds, per client IP
_rate_hits: "defaultdict[str, deque]" = defaultdict(deque)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    # /health stays open for load balancers; OPTIONS preflights are free.
    if request.url.path == "/health" or request.method == "OPTIONS":
        return await call_next(request)
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    hits = _rate_hits[ip]
    while hits and now - hits[0] > RATE_LIMIT_WINDOW:
        hits.popleft()
    if len(hits) >= RATE_LIMIT_MAX:
        return JSONResponse(status_code=429,
                            content={"detail": "Too many requests — slow down a little."})
    hits.append(now)
    return await call_next(request)


@app.middleware("http")
async def access_log(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = int((time.perf_counter() - start) * 1000)
    log.info("%s %s -> %s (%d ms)",
             request.method, request.url.path, response.status_code, duration_ms)
    return response


# CORS = permission for the frontend (running on a different origin) to call us.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Optional bearer-token auth ----------------------------------------------
def require_auth(authorization: str = Header(default="")) -> None:
    if not APP_ACCESS_TOKEN:
        return  # local dev: auth disabled
    expected = f"Bearer {APP_ACCESS_TOKEN}"
    if not authorization or not secrets.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Missing or invalid access token.")


# Every route on this router requires auth (when APP_ACCESS_TOKEN is set).
# /health is registered directly on the app and stays open.
router = APIRouter(dependencies=[Depends(require_auth)])


# ---- Engine (one shared instance, single-process) -----------------------------
_engine = None
_engine_lock = threading.Lock()


def get_engine() -> RAGEngine:
    global _engine
    key = os.getenv("GOOGLE_API_KEY", "")
    if not key:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_API_KEY is missing. Add a line 'GOOGLE_API_KEY=your_key' to your .env file.",
        )
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = RAGEngine(api_key=key)
    return _engine


# ---- Client-safe error mapping ------------------------------------------------
def _is_rate_limit_err(e) -> bool:
    s = str(e)
    return "429" in s or "RESOURCE_EXHAUSTED" in s or "quota" in s.lower() or "rate limit" in s.lower()


def _is_busy_err(e) -> bool:
    return str(e).lower().startswith("busy:")


def _friendly(e) -> str:
    if _is_rate_limit_err(e):
        return ("You've hit Gemini's free per-minute limit — wait ~30-60 seconds and try again. "
                "(Big questions use more of the limit.)")
    if _is_busy_err(e):
        return "The server is answering other questions — try again in a moment."
    return "The AI request failed — please try again in a moment."


def _ai_http_error(e) -> HTTPException:
    """Map an engine/LLM exception to a client-safe HTTPException.
    The raw exception goes to the server log, never to the client."""
    if isinstance(e, HTTPException):
        return e
    log.exception("AI request failed")
    if _is_rate_limit_err(e):
        return HTTPException(status_code=429, detail=_friendly(e))
    if _is_busy_err(e):
        return HTTPException(status_code=503, detail=_friendly(e))
    return HTTPException(status_code=502, detail=_friendly(e))


# ---- Shapes of incoming JSON (FastAPI validates these automatically) ----
class ChatIn(BaseModel):
    question: str
    history: list = []
    pattern: str = ""     # non-empty switches the engine into document-generator mode
    source: str = ""      # "" = search all documents; otherwise scope to this one
    sources: list = []    # exam-paper mode: chosen reference documents ([] = all)
    paper_opts: str = ""  # legacy free-text paper options (kept for compatibility)


class QuizIn(BaseModel):
    num_questions: int = 5
    source: str = ""
    topic: str = ""  # optional focus topic — used as the retrieval query when given
    avoid: list = []  # stems of recently asked questions, to prevent repeats


class SummaryIn(BaseModel):
    source: str = ""


class DeleteDocIn(BaseModel):
    source: str


class UrlIn(BaseModel):
    url: str


class ExportIn(BaseModel):
    text: str
    format: str = "pdf"
    pattern_id: str = ""  # id of a stored .docx sample — export clones that file
    layout: dict = {}     # visual spec from a photo sample: font / centered header / footer


# A tiny adapter so the engine's process_file (built for Streamlit uploads)
# also accepts FastAPI uploads. The engine only needs .name and .read().
class _UploadShim:
    def __init__(self, filename: str, data: bytes):
        self.name = filename
        self._data = data

    def read(self) -> bytes:
        return self._data


def _fill_docx(doc, text: str, layout: dict, center_rest: bool = False):
    """Write the markdown-lite paper text into a python-docx Document."""
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    center_n = int((layout or {}).get("center_top_lines", 0) or 0)
    line_no = 0
    for raw in text.split("\n"):
        st = raw.strip()
        if not st:
            doc.add_paragraph("")
            continue
        line_no += 1
        if st.startswith("### "):
            p = doc.add_heading(st[4:], level=3)
        elif st.startswith("## "):
            p = doc.add_heading(st[3:], level=2)
        elif st.startswith("# "):
            p = doc.add_heading(st[2:], level=1)
        elif st.startswith(("- ", "* ")):
            p = doc.add_paragraph(st[2:].replace("**", ""), style="List Bullet")
        else:
            p = doc.add_paragraph(st.replace("**", ""))
        # The sample's opening title/institute block was center-aligned — mirror it.
        if line_no <= center_n or center_rest:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            if line_no <= center_n:
                for run in p.runs:
                    run.bold = True


def _paper_to_bytes(text: str, fmt: str, layout: dict = None, pattern_path: str = None):
    """Convert a markdown paper into pdf / docx / pptx bytes.
    - pattern_path (a stored .docx sample): the export CLONES that file — its real
      header, footer, margins, page setup, and fonts are preserved exactly.
    - layout (from a photo sample): styles the generic export — serif font,
      centered title block, footer text."""
    layout = layout or {}
    if fmt == "pdf":
        # fpdf's core fonts are latin-1 only. Normalize common typography the
        # model produces (em-dashes, smart quotes) to ASCII equivalents first;
        # refuse loudly ONLY for genuinely unmappable text (e.g. Urdu) instead
        # of silently stripping it out of the exported paper.
        text = text.translate(str.maketrans({
            "—": "-", "–": "-", "‘": "'", "’": "'",
            "“": '"', "”": '"', "…": "...", " ": " ",
            "•": "-", "→": "->", "×": "x", "−": "-",
        }))
        try:
            text.encode("latin-1")
        except UnicodeEncodeError:
            raise ValueError("PDF export can't render Urdu or special characters yet — "
                             "download as Word instead (full support).")
        from fpdf import FPDF
        from fpdf.enums import XPos, YPos

        family = "Times" if layout.get("font") == "serif" else "Helvetica"
        footer_text = str(layout.get("footer", "") or "")

        class PaperPDF(FPDF):
            def footer(self):
                self.set_y(-13)
                self.set_font(family, "I", 9)
                left = footer_text[:80]
                self.cell(self.epw / 2, 6, left, align="L")
                self.cell(self.epw / 2, 6, f"Page {self.page_no()}", align="R")

        pdf = PaperPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=18)
        center_n = int(layout.get("center_top_lines", 0) or 0)
        line_no = 0

        def _line(t, size, style="", align="L"):
            pdf.set_font(family, style, size)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(pdf.epw, size * 0.55, t, align=align,
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        for raw in text.split("\n"):
            st = raw.strip()
            if not st:
                pdf.ln(3)
                continue
            line_no += 1
            centered = "C" if line_no <= center_n else "L"
            if st.startswith("### "):
                _line(st[4:], 12, "B", centered)
            elif st.startswith("## "):
                _line(st[3:], 14, "B", centered)
            elif st.startswith("# "):
                _line(st[2:], 16, "B", centered)
            elif st.startswith(("- ", "* ")):
                _line("  - " + st[2:].replace("**", ""), 11, "", centered)
            else:
                _line(st.replace("**", ""), 11, "B" if line_no <= center_n else "", centered)
        return bytes(pdf.output()), "application/pdf", "pdf"
    if fmt == "docx":
        import io
        from docx import Document as Docx
        if pattern_path and os.path.exists(pattern_path):
            # EXACT-format path: clone the teacher's own sample file. Its header,
            # footer, margins, styles, and fonts stay untouched — we only replace
            # the body content with the new paper.
            doc = Docx(pattern_path)
            for tbl in list(doc.tables):
                tbl._element.getparent().remove(tbl._element)
            for p in list(doc.paragraphs):
                p._element.getparent().remove(p._element)
            _fill_docx(doc, text, layout)
        else:
            doc = Docx()
            if layout.get("font") == "serif":
                doc.styles["Normal"].font.name = "Times New Roman"
            _fill_docx(doc, text, layout)
            footer_text = str(layout.get("footer", "") or "")
            if footer_text:
                try:
                    doc.sections[0].footer.paragraphs[0].text = footer_text[:200]
                except Exception:
                    log.exception("Could not write docx footer")
        buf = io.BytesIO(); doc.save(buf)
        return buf.getvalue(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "docx"
    if fmt == "pptx":
        import io
        from pptx import Presentation
        from pptx.util import Inches, Pt
        prs = Presentation()
        blank = prs.slide_layouts[6]
        lines = text.split("\n")
        groups, chunk = [], []
        for l in lines:
            chunk.append(l)
            if len(chunk) >= 14:
                groups.append(chunk); chunk = []
        if chunk:
            groups.append(chunk)
        for group in groups or [[""]]:
            slide = prs.slides.add_slide(blank)
            tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(6.5))
            tf = tb.text_frame
            tf.word_wrap = True
            for i, ln in enumerate(group):
                clean = ln.replace("#", "").replace("**", "").strip()
                p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                p.text = clean
                p.font.size = Pt(12)
        buf = io.BytesIO(); prs.save(buf)
        return buf.getvalue(), "application/vnd.openxmlformats-officedocument.presentationml.presentation", "pptx"
    raise ValueError("Unsupported format")


# ---- Routes -------------------------------------------------------------------
@app.get("/health")
def health():
    try:
        eng = get_engine()
    except HTTPException:
        return {"status": "degraded", "documents": 0, "chunks": 0, "store_ok": False}
    return {
        "status": "ok",
        "documents": eng.doc_count,
        "chunks": eng.chunk_count,
        "store_ok": not eng.store_error,
    }


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=UPLOAD_TOO_BIG)
    try:
        chunks = get_engine().process_file(_UploadShim(file.filename, data))
    except HTTPException:
        raise
    except Exception as e:
        # Photo transcription goes through Gemini — surface limit/busy errors honestly.
        if _is_rate_limit_err(e) or _is_busy_err(e):
            raise _ai_http_error(e)
        log.exception("Upload failed for %r", file.filename)
        raise HTTPException(status_code=400,
                            detail="Couldn't read that file — supported: PDF, Word, PPT, TXT, MD, CSV, or a photo (JPG/PNG).")
    return {"filename": file.filename, "chunks": chunks, "stats": get_engine().get_stats()}


@router.post("/link")
def link(body: UrlIn):
    """Smart link handler: YouTube link -> transcript, anything else -> web page."""
    url = body.url.strip()
    try:
        if extract_video_id(url):
            chunks = get_engine().process_youtube_url(url)
        else:
            chunks = get_engine().process_url(url)
    except HTTPException:
        raise
    except ValueError as e:
        # Our own guard messages (bad YouTube link, SSRF rejection) — safe to show.
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        log.exception("Link ingestion failed for %r", url)
        raise HTTPException(status_code=400,
                            detail="Couldn't read that link — the site may block bots or need a login.")
    return {"chunks": chunks, "stats": get_engine().get_stats()}


@router.post("/chat-stream")
def chat_stream(body: ChatIn):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Ask a question first.")

    def gen():
        try:
            for part in get_engine().query_stream(
                body.question,
                history=body.history,
                pattern=body.pattern,
                source=body.source,
                paper_opts=body.paper_opts,
                ref_sources=[s for s in body.sources if isinstance(s, str) and s] or None,
            ):
                yield json.dumps(part) + "\n"
        except Exception as e:
            log.exception("chat-stream failed")
            yield json.dumps({"done": True, "error": _friendly(e),
                              "sources": [], "confidence": 0, "usage": {}}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@router.post("/quiz")
def quiz(body: QuizIn):
    try:
        return {"questions": get_engine().generate_quiz(
            num_questions=body.num_questions, source=body.source, topic=body.topic,
            avoid=[a for a in body.avoid if isinstance(a, str)][:20])}
    except Exception as e:
        raise _ai_http_error(e)


@router.post("/summary")
def summary(body: SummaryIn = SummaryIn()):
    try:
        return get_engine().summarize(source=body.source)
    except Exception as e:
        raise _ai_http_error(e)


@router.get("/documents")
def documents():
    """List every indexed document with its chunk count (powers the doc manager)."""
    return {"documents": get_engine().list_documents()}


@router.post("/documents/delete")
def delete_document(body: DeleteDocIn):
    try:
        stats = get_engine().delete_document(body.source)
    except HTTPException:
        raise
    except Exception:
        log.exception("Delete failed for %r", body.source)
        raise HTTPException(status_code=400, detail="Could not delete that document — try again.")
    return {"status": "deleted", "stats": stats}


@router.post("/extract")
async def extract(file: UploadFile = File(...)):
    """Extract text from an uploaded sample/pattern paper (not indexed as content).
    - Photos: Gemini vision transcribes AND reads the visual layout (font,
      centered header block, footer) so exports can mirror the look.
    - .docx samples: the raw file is also stored, so /export can clone it for an
      EXACT-format paper (real header/footer/fonts preserved)."""
    import hashlib
    import tempfile
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=UPLOAD_TOO_BIG)
    ext = "." + (file.filename or "x.txt").split(".")[-1].lower()
    if ext in RAGEngine.IMAGE_EXTS:
        try:
            text, layout = get_engine().image_to_text(data, with_layout=True)
        except Exception as e:
            raise _ai_http_error(e)
        return {"text": text[:15000], "layout": layout}
    pattern_id = ""
    if ext == ".docx":
        pattern_id = hashlib.sha256(data).hexdigest()[:16]
        try:
            with open(os.path.join(PATTERN_DIR, f"{pattern_id}.docx"), "wb") as f:
                f.write(data)
        except Exception:
            log.exception("Could not store pattern file")
            pattern_id = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        docs = RAGEngine._load_any(path, ext)
        text = "\n\n".join(d.page_content for d in docs)
    except Exception:
        log.exception("Extract failed for %r", file.filename)
        raise HTTPException(status_code=400,
                            detail="Couldn't read that file — supported: PDF, Word, PPT, TXT, MD, CSV, or a photo.")
    finally:
        os.unlink(path)
    return {"text": text[:15000], "pattern_id": pattern_id}


@router.post("/export")
def export(body: ExportIn):
    import re as _re
    pattern_path = None
    if body.pattern_id and _re.fullmatch(r"[0-9a-f]{16}", body.pattern_id):
        candidate = os.path.join(PATTERN_DIR, f"{body.pattern_id}.docx")
        if os.path.exists(candidate):
            pattern_path = candidate
    try:
        data, mime, ext = _paper_to_bytes(body.text, body.format,
                                          layout=body.layout, pattern_path=pattern_path)
    except ValueError as e:
        # Our own messages (Urdu/PDF limitation, unsupported format) — safe to show.
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        log.exception("Export failed")
        raise HTTPException(status_code=400, detail="Export failed — try a different format.")
    return Response(content=data, media_type=mime,
                    headers={"Content-Disposition": f'attachment; filename="studymind-paper.{ext}"'})


@router.get("/stats")
def stats():
    return get_engine().get_stats()


@router.get("/usage")
def usage():
    """Per-model free-tier usage for the UI's limit bar (our own counts —
    Google exposes no remaining-quota API, so daily_limit values are estimates)."""
    return get_engine().usage_snapshot()


@router.post("/reset")
def reset():
    get_engine().reset()
    return {"status": "reset"}


app.include_router(router)
