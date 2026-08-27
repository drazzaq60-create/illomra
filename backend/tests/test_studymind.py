"""Five-test suite for the StudyMind FastAPI backend (from the code-review spec).

No network LLM calls: every path that would reach Gemini is monkeypatched.
The one deliberately slow test is test_ingest_roundtrip, which constructs a
real RAGEngine (local HuggingFace embeddings only — still no Gemini) against
a temporary Chroma directory.
"""
import json

from fastapi.testclient import TestClient

import api
import rag_engine
from rag_engine import RAGEngine


def _bare_engine() -> RAGEngine:
    """Engine instance WITHOUT __init__ — no embedding-model load.

    Safe for the pure instance methods (_parse_quiz, _response_tier,
    _decide_web_search) which never touch instance state.
    """
    return RAGEngine.__new__(RAGEngine)


# ---- 1. quiz parsing ---------------------------------------------------------

WELL_FORMED = """\
Q1. What is a stack?
A) A LIFO data structure
B) A FIFO data structure
C) A sorting algorithm
D) A hash table
Answer: A
Explanation: A stack is last-in, first-out.

Q2. What does CPU stand for?
A) Central Processing Unit
B) Computer Power Unit
C) Central Print Unit
D) Core Processing Utility
Answer: A
Explanation: CPU means Central Processing Unit.
"""

MALFORMED_MISSING_D = """\
Q1. What is a queue?
A) LIFO
B) FIFO
C) A tree
Answer: B
Explanation: This block has no option D, so it must be skipped.
"""


def test_parse_quiz():
    eng = _bare_engine()

    parsed = eng._parse_quiz(WELL_FORMED)
    assert len(parsed) == 2
    for q in parsed:
        assert set(q) == {"question", "options", "answer", "explanation"}
        assert set(q["options"]) == {"A", "B", "C", "D"}
        assert all(isinstance(v, str) and v for v in q["options"].values())
        assert q["answer"] in "ABCD"
        assert isinstance(q["explanation"], str)
    assert parsed[0]["question"] == "What is a stack?"
    assert parsed[0]["answer"] == "A"
    assert parsed[0]["options"]["A"] == "A LIFO data structure"
    assert parsed[1]["question"] == "What does CPU stand for?"

    # Malformed block (missing option D) is skipped — no crash, no partial dict.
    assert eng._parse_quiz(MALFORMED_MISSING_D) == []


# ---- 2. response tier --------------------------------------------------------

def test_response_tier():
    eng = _bare_engine()

    # Short factual question -> small budget.
    assert eng._response_tier("what is a stack?", "")[0] == 1200
    # Explicit "in detail" -> full budget.
    assert eng._response_tier("Explain self-attention in detail", "")[0] == 8000
    # Document-generator mode (any non-empty pattern) -> full budget.
    assert eng._response_tier("make me a midterm paper", "SAMPLE")[0] == 8000
    # Mid-sentence "when" must NOT mark a medium question as short.
    assert eng._response_tier(
        "How does backprop update the weights when the loss is high", "")[0] == 4000


# ---- 3. web-search trigger ---------------------------------------------------

def test_decide_web_search():
    eng = _bare_engine()

    # "research" contains "search" but is NOT an explicit trigger.
    assert eng._decide_web_search("explain my research methodology") is None
    # Explicit ask -> truthy query string.
    assert eng._decide_web_search("search the web for the latest transformers papers")


# ---- 4. /chat-stream NDJSON contract ----------------------------------------

class _FakeEngine:
    def __init__(self):
        self.calls = []

    def query_stream(self, question, history=None, pattern="", source="", paper_opts="", ref_sources=None, owner=None):
        self.calls.append({"question": question, "history": history,
                           "pattern": pattern, "source": source,
                           "paper_opts": paper_opts, "ref_sources": ref_sources,
                           "owner": owner})
        yield {"token": "Hel"}
        yield {"token": "lo"}
        yield {"done": True, "sources": [], "confidence": 0, "web_used": False,
               "usage": {}, "latency_ms": 1, "cost": "free"}


class _BoomEngine:
    def query_stream(self, *args, **kwargs):
        raise Exception("boom")


def test_chat_stream_contract(monkeypatch):
    fake = _FakeEngine()
    monkeypatch.setattr(api, "get_engine", lambda: fake)
    monkeypatch.setattr(api, "AUTH_MODE", "open")
    # No context manager -> lifespan (real-engine warm-up) never runs.
    client = TestClient(api.app)

    resp = client.post("/chat-stream", json={"question": "hi"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")

    lines = [ln for ln in resp.text.splitlines() if ln.strip()]
    frames = [json.loads(ln) for ln in lines]  # every non-empty line is valid JSON
    assert "".join(f["token"] for f in frames if "token" in f) == "Hello"
    assert frames[-1]["done"] is True
    assert "error" not in frames[-1]
    assert fake.calls == [{"question": "hi", "history": [], "pattern": "",
                           "source": "", "paper_opts": "", "ref_sources": None,
                           "owner": None}]

    # Error contract the frontend depends on: engine blows up -> still HTTP 200
    # with EXACTLY ONE NDJSON line carrying done==True and an "error" key.
    monkeypatch.setattr(api, "get_engine", lambda: _BoomEngine())
    resp = client.post("/chat-stream", json={"question": "hi again"})
    assert resp.status_code == 200
    lines = [ln for ln in resp.text.splitlines() if ln.strip()]
    assert len(lines) == 1
    frame = json.loads(lines[0])
    assert frame["done"] is True
    assert "error" in frame and frame["error"]


# ---- 5. real-engine ingest roundtrip ----------------------------------------

NOTES_TEXT = (
    "Stacks and Queues — study notes.\n\n"
    "A stack is a LIFO (last-in, first-out) data structure. Elements are pushed "
    "onto the top and popped from the top. Typical uses: undo history, the call "
    "stack for function invocations, and depth-first search.\n\n"
    "A queue is a FIFO (first-in, first-out) data structure. Elements are "
    "enqueued at the back and dequeued from the front. Typical uses: task "
    "scheduling, breadth-first search, and buffering producer-consumer work.\n\n"
    "A deque supports insertion and removal at both ends, generalising both the "
    "stack and the queue. Ring buffers implement queues in fixed memory.\n\n"
    "Complexity: push, pop, enqueue and dequeue are all O(1) amortised in a "
    "dynamic-array or linked-list implementation. Searching an unsorted stack "
    "or queue is O(n) because every element may need to be inspected.\n"
)


def test_ingest_roundtrip(tmp_path, monkeypatch):
    """The ONE real-engine test: constructs RAGEngine (loads the local
    HuggingFace embedding model, ~10-20s) against a temp Chroma dir.
    Local embeddings only — no Gemini call anywhere in this path."""
    monkeypatch.setattr(rag_engine, "PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(api, "_engine", None)  # force a fresh engine on the temp dir
    monkeypatch.setattr(api, "AUTH_MODE", "open")

    client = TestClient(api.app)

    # Upload a small .txt (multipart).
    r1 = client.post(
        "/upload",
        files={"file": ("notes.txt", NOTES_TEXT.encode("utf-8"), "text/plain")},
    )
    assert r1.status_code == 200, r1.text
    body1 = r1.json()
    assert body1["filename"] == "notes.txt"
    assert body1["chunks"] >= 1
    first_chunks = body1["stats"]["chunks"]
    assert first_chunks == body1["chunks"]

    # /documents lists it with the right chunk count.
    r2 = client.get("/documents")
    assert r2.status_code == 200
    docs = r2.json()["documents"]
    assert [d["source"] for d in docs] == ["notes.txt"]
    assert docs[0]["chunks"] == first_chunks

    # Idempotent re-upload: the SAME file must replace, not duplicate.
    r3 = client.post(
        "/upload",
        files={"file": ("notes.txt", NOTES_TEXT.encode("utf-8"), "text/plain")},
    )
    assert r3.status_code == 200, r3.text
    assert r3.json()["stats"]["chunks"] == first_chunks
    assert r3.json()["stats"]["documents"] == 1

    # Delete removes it completely.
    r4 = client.post("/documents/delete", json={"source": "notes.txt"})
    assert r4.status_code == 200, r4.text
    assert r4.json()["status"] == "deleted"
    assert r4.json()["stats"] == {"documents": 0, "chunks": 0}

    r5 = client.get("/documents")
    assert r5.status_code == 200
    assert r5.json()["documents"] == []
