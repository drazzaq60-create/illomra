"""
api.py — StudyMind backend (FastAPI).

This is the API layer. It reuses your existing brain (rag_engine.py) and
exposes it over HTTP so a Next.js/React frontend can call it.

Run it:
    uvicorn api:app --reload --port 8000
Then open http://localhost:8000/docs to try every endpoint in the browser.
"""

import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.responses import StreamingResponse
import json
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_engine import RAGEngine, AVAILABLE_MODELS, extract_video_id

# Load the .env sitting right next to this file — works no matter which
# folder you launch uvicorn from.
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the embedding model at boot so the FIRST question isn't slow.
    try:
        get_engine()
    except Exception:
        pass
    yield


app = FastAPI(title="StudyMind API", version="0.1.0", lifespan=lifespan)

# CORS = permission for the frontend (running on a different port) to call us.
# Next.js dev server runs on http://localhost:3000 by default.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# One shared engine for now (single user). Per-user isolation comes later.
_engine = None


def get_engine() -> RAGEngine:
    global _engine
    key = os.getenv("GOOGLE_API_KEY", "")
    if not key:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_API_KEY is missing. Add a line 'GOOGLE_API_KEY=your_key' to your .env file.",
        )
    if _engine is None:
        _engine = RAGEngine(api_key=key)
    return _engine


def _friendly(e) -> str:
    s = str(e)
    if "429" in s or "RESOURCE_EXHAUSTED" in s or "quota" in s.lower() or "rate limit" in s.lower():
        return "You've hit Gemini's free per-minute limit — wait ~30-60 seconds and try again. (Big questions use more of the limit.)"
    return f"AI request failed: {s}"


# ---- Shapes of incoming JSON (FastAPI validates these automatically) ----
class ChatIn(BaseModel):
    question: str
    history: list = []
    k: int = 5
    pattern: str = ""
    source: str = ""  # "" = search all documents; otherwise scope to this one


class QuizIn(BaseModel):
    num_questions: int = 5
    source: str = ""


class SummaryIn(BaseModel):
    source: str = ""


class DeleteDocIn(BaseModel):
    source: str


class UrlIn(BaseModel):
    url: str


class ExamIn(BaseModel):
    pattern_text: str = ""
    instructions: str = ""
    reference_text: str = ""
    reference_url: str = ""


class ExportIn(BaseModel):
    text: str
    format: str = "pdf"


# A tiny adapter so the engine's process_file (built for Streamlit uploads)
# also accepts FastAPI uploads. The engine only needs .name and .read().
class _UploadShim:
    def __init__(self, filename: str, data: bytes):
        self.name = filename
        self._data = data

    def read(self) -> bytes:
        return self._data


def _paper_to_bytes(text: str, fmt: str):
    """Convert a markdown paper into pdf / docx / pptx bytes."""
    if fmt == "pdf":
        from fpdf import FPDF
        from fpdf.enums import XPos, YPos
        safe = text.encode("latin-1", "ignore").decode("latin-1")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        def _line(t, size, style=""):
            pdf.set_font("Helvetica", style, size)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(pdf.epw, size * 0.55, t, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        for raw in safe.split("\n"):
            st = raw.strip()
            if st.startswith("### "):
                _line(st[4:], 12, "B")
            elif st.startswith("## "):
                _line(st[3:], 14, "B")
            elif st.startswith("# "):
                _line(st[2:], 16, "B")
            elif st.startswith(("- ", "* ")):
                _line("  - " + st[2:].replace("**", ""), 11)
            elif st:
                _line(st.replace("**", ""), 11)
            else:
                pdf.ln(3)
        return bytes(pdf.output()), "application/pdf", "pdf"
    if fmt == "docx":
        import io
        from docx import Document as Docx
        doc = Docx()
        for raw in text.split("\n"):
            st = raw.strip()
            if st.startswith("### "):
                doc.add_heading(st[4:], level=3)
            elif st.startswith("## "):
                doc.add_heading(st[3:], level=2)
            elif st.startswith("# "):
                doc.add_heading(st[2:], level=1)
            elif st.startswith(("- ", "* ")):
                doc.add_paragraph(st[2:].replace("**", ""), style="List Bullet")
            elif st:
                doc.add_paragraph(st.replace("**", ""))
            else:
                doc.add_paragraph("")
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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "key_loaded": bool(os.getenv("GOOGLE_API_KEY")),  # True if your .env was read
        "models": list(AVAILABLE_MODELS.keys()),
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    try:
        chunks = get_engine().process_file(_UploadShim(file.filename, data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read that file: {e}")
    return {"filename": file.filename, "chunks": chunks, "stats": get_engine().get_stats()}


@app.post("/youtube")
def youtube(body: UrlIn):
    try:
        chunks = get_engine().process_youtube_url(body.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"chunks": chunks, "stats": get_engine().get_stats()}


@app.post("/link")
def link(body: UrlIn):
    """Smart link handler: YouTube link -> transcript, anything else -> web page."""
    url = body.url.strip()
    try:
        if extract_video_id(url):
            chunks = get_engine().process_youtube_url(url)
        else:
            chunks = get_engine().process_url(url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"chunks": chunks, "stats": get_engine().get_stats()}


@app.post("/chat")
def chat(body: ChatIn):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Ask a question first.")
    try:
        return get_engine().query(body.question, history=body.history, k=body.k, pattern=body.pattern, source=body.source)
    except Exception as e:
        raise HTTPException(status_code=502, detail=_friendly(e))


@app.post("/chat-stream")
def chat_stream(body: ChatIn):
    def gen():
        try:
            for part in get_engine().query_stream(body.question, history=body.history, k=body.k, pattern=body.pattern, source=body.source):
                yield json.dumps(part) + "\n"
        except Exception as e:
            yield json.dumps({"done": True, "error": _friendly(e), "sources": [], "confidence": 0, "usage": {}}) + "\n"
    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/quiz")
def quiz(body: QuizIn):
    try:
        return {"questions": get_engine().generate_quiz(num_questions=body.num_questions, source=body.source)}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI request failed: {e}")


@app.post("/summary")
def summary(body: SummaryIn = SummaryIn()):
    try:
        return get_engine().summarize(source=body.source)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI request failed: {e}")


@app.get("/documents")
def documents():
    """List every indexed document with its chunk count (powers the doc manager)."""
    return {"documents": get_engine().list_documents()}


@app.post("/documents/delete")
def delete_document(body: DeleteDocIn):
    try:
        stats = get_engine().delete_document(body.source)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not delete that document: {e}")
    return {"status": "deleted", "stats": stats}


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    """Extract plain text from an uploaded sample/pattern paper (not indexed as content)."""
    import tempfile
    data = await file.read()
    ext = "." + (file.filename or "x.txt").split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        docs = RAGEngine._load_any(path, ext)
        text = "\n\n".join(d.page_content for d in docs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read that file: {e}")
    finally:
        os.unlink(path)
    return {"text": text[:15000]}


@app.post("/exam")
def exam(body: ExamIn):
    try:
        return get_engine().generate_exam(body.pattern_text, body.instructions, body.reference_text, body.reference_url)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Paper generation failed: {e}")


@app.post("/export")
def export(body: ExportIn):
    try:
        data, mime, ext = _paper_to_bytes(body.text, body.format)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Export failed: {e}")
    return Response(content=data, media_type=mime, headers={"Content-Disposition": f'attachment; filename="studymind-paper.{ext}"'})


@app.get("/stats")
def stats():
    return get_engine().get_stats()


@app.post("/reset")
def reset():
    get_engine().reset()
    return {"status": "reset"}
