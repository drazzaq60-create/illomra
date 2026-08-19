"""Tests for model-decided (agentic) web search in query_stream — no real LLM."""
import rag_engine
from rag_engine import ModelQuota, RAGEngine


class _Chunk:
    def __init__(self, t):
        self.content = t
        self.usage_metadata = {"input_tokens": 5, "output_tokens": 3}


def _engine(monkeypatch, tmp_path, scripts):
    """Fake engine whose successive .stream() calls play the given scripts."""
    monkeypatch.setattr(rag_engine, "_QUOTA", ModelQuota(["model-a"], path=str(tmp_path / "u.json")))
    e = RAGEngine.__new__(RAGEngine)
    e._llm_cache = {}
    e.vector_store = None
    e.chunk_count = 0
    e.source_counts = {}
    e.total_input_tokens = 0
    e.total_output_tokens = 0
    import threading
    e._state_lock = threading.Lock()

    calls = {"n": 0, "prompts": []}

    class FakeLLM:
        def stream(self, msgs):
            calls["prompts"].append(msgs[0].content)
            script = scripts[min(calls["n"], len(scripts) - 1)]
            calls["n"] += 1
            for t in script:
                yield _Chunk(t)

    e._get_llm = lambda model, max_tokens=8000: FakeLLM()
    return e, calls


def _run(e):
    frames = list(e.query_stream("what is the latest AI news?"))
    tokens = "".join(f.get("token", "") for f in frames)
    notices = [f["notice"] for f in frames if "notice" in f]
    done = [f for f in frames if f.get("done")][0]
    return tokens, notices, done


def test_model_requests_search_then_answers(monkeypatch, tmp_path):
    """Marker stream → search runs → second stream is the real answer."""
    e, calls = _engine(monkeypatch, tmp_path, scripts=[
        ["<<SEA", "RCH: latest AI news", ">>"],           # phase 0: model asks to search
        ["The latest news ", "is X."],                    # phase 1: answer with results
    ])
    searched = {}

    def fake_search(q, n=5):
        searched["q"] = q
        return [{"title": "News", "snippet": "X happened", "url": "https://example.com"}]

    e.web_search = fake_search
    tokens, notices, done = _run(e)

    assert searched["q"] == "latest AI news"
    assert tokens == "The latest news is X."          # marker never reached the UI
    assert any("Searching the web" in n for n in notices)
    assert done["web_used"] is True
    assert any("example.com" in str(s.get("page", "")) for s in done["sources"])
    # Second prompt must NOT offer the search marker again (no loops).
    assert "<<SEARCH:" not in calls["prompts"][1]


def test_normal_answer_streams_unchanged(monkeypatch, tmp_path):
    """No marker → buffered start flushes and the answer streams normally."""
    e, _ = _engine(monkeypatch, tmp_path, scripts=[
        ["Photosynthesis ", "is how plants ", "make food."],
    ])
    e.web_search = lambda q, n=5: (_ for _ in ()).throw(AssertionError("must not search"))
    tokens, notices, done = _run(e)
    assert tokens == "Photosynthesis is how plants make food."
    assert not any("Searching" in n for n in notices)
    assert done["web_used"] is False


def test_failed_search_still_answers(monkeypatch, tmp_path):
    """Search returns nothing → user is told, and the answer still arrives."""
    e, _ = _engine(monkeypatch, tmp_path, scripts=[
        ["<<SEARCH: something obscure>>"],
        ["I don't have current data, but here's what I know."],
    ])
    e.web_search = lambda q, n=5: []
    tokens, notices, done = _run(e)
    assert "here's what I know" in tokens
    assert any("found nothing" in n for n in notices)
    assert done["web_used"] is False
