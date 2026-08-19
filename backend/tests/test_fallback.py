"""Tests for the model fallback chain + quota tracker (no LLM calls)."""
import time

import rag_engine
from rag_engine import ModelQuota


def _fresh_quota(tmp_path, chain=("model-a", "model-b", "model-lite")):
    return ModelQuota(list(chain), path=str(tmp_path / "usage.json"))


def test_minute_limit_cooldown_then_recovers(tmp_path):
    q = _fresh_quota(tmp_path)
    assert q.usable_models() == ["model-a", "model-b", "model-lite"]

    # A per-minute 429 (no 'day' in the message) → short cooldown, others usable.
    q.mark_limited("model-a", "429 RESOURCE_EXHAUSTED: rate limit exceeded per minute")
    assert q.usable_models() == ["model-b", "model-lite"]
    # It comes back after the cooldown window.
    q.state["model-a"]["cooldown_until"] = time.time() - 1
    assert "model-a" in q.usable_models()


def test_daily_limit_out_until_reset(tmp_path):
    q = _fresh_quota(tmp_path)
    # A daily-quota 429 → exhausted until the reset, not a short cooldown.
    q.mark_limited("model-a", "429 quota metric generate_requests_per_model_per_day exceeded")
    assert "model-a" not in q.usable_models()
    slot = q.state["model-a"]
    assert slot["exhausted_until"] > time.time() + 3600  # hours away, not seconds
    # soonest_retry_s ignores day-exhausted models (they're not a short wait).
    assert q.soonest_retry_s() == 0.0  # other two models are free right now


def test_snapshot_shape_and_persistence(tmp_path):
    path = tmp_path / "usage.json"
    q = ModelQuota(["model-a", "model-lite"], path=str(path))
    q.record_request("model-a")
    q.record_request("model-a")

    snap = q.snapshot()
    assert snap["primary"] == "model-a"
    assert snap["totals"]["capacity"] == 250 + 1000  # flash + lite estimates
    assert snap["totals"]["used_today"] == 2
    assert snap["totals"]["resets_in_s"] > 0
    a = next(m for m in snap["models"] if m["model"] == "model-a")
    assert a["used_today"] == 2 and a["status"] == "ok"

    # A new tracker on the same file sees the counts (survives --reload restarts).
    q2 = ModelQuota(["model-a", "model-lite"], path=str(path))
    assert q2.snapshot()["totals"]["used_today"] == 2


def test_invoke_falls_through_chain(tmp_path, monkeypatch):
    """_invoke_llm hops to the next model on a 429 and records the success."""
    quota = _fresh_quota(tmp_path, chain=("model-a", "model-b"))
    monkeypatch.setattr(rag_engine, "_QUOTA", quota)

    calls = []

    class FakeResp:
        content = "answer"
        usage_metadata = {"input_tokens": 1, "output_tokens": 1}

    class FakeLLM:
        def __init__(self, model):
            self.model = model

        def invoke(self, msgs):
            calls.append(self.model)
            if self.model == "model-a":
                raise Exception("429 RESOURCE_EXHAUSTED per day quota")
            return FakeResp()

    engine = rag_engine.RAGEngine.__new__(rag_engine.RAGEngine)
    engine._llm_cache = {}
    engine._get_llm = lambda model, max_tokens=8000: FakeLLM(model)

    resp = engine._invoke_llm("hello", 1200)
    assert calls == ["model-a", "model-b"]
    assert resp.content == "answer"
    # model-a is now day-exhausted; model-b recorded one request.
    assert "model-a" not in quota.usable_models()
    assert quota.state["model-b"]["used"] == 1
