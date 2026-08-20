"""Google-auth mode: token verification wiring + REAL per-user isolation.
No network calls — the Google verifier is monkeypatched; Chroma is real (tmp)."""
import importlib
import io

import pytest
from fastapi.testclient import TestClient

import api
import rag_engine


@pytest.fixture()
def google_mode(monkeypatch, tmp_path):
    """Switch the app into google auth with a fake verifier: 'Bearer <name>'
    authenticates as owner 'g<name>'. Fresh engine + stores in tmp."""
    monkeypatch.setattr(api, "AUTH_MODE", "google")
    monkeypatch.setattr(api, "GOOGLE_OAUTH_CLIENT_ID", "test-client")
    monkeypatch.setattr(api, "_verify_google_token", lambda tok: f"g{tok}")
    monkeypatch.setattr(api, "CONVO_STORE", str(tmp_path / "convos.json"))
    monkeypatch.setattr(rag_engine, "PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(api, "_engine", None)
    monkeypatch.setattr(api, "RATE_LIMIT_MAX", 1000)
    client = TestClient(api.app)
    return client


def _upload(client, who, filename, text):
    return client.post(
        "/upload",
        headers={"Authorization": f"Bearer {who}"},
        files={"file": (filename, io.BytesIO(text.encode()), "text/plain")},
    )


def test_auth_required_and_verified(google_mode):
    client = google_mode
    # No header -> 401 (never leaks data).
    assert client.get("/documents").status_code == 401
    # Bad scheme -> 401.
    assert client.get("/documents", headers={"Authorization": "Basic x"}).status_code == 401
    # Verified token -> 200.
    r = client.get("/documents", headers={"Authorization": "Bearer alice"})
    assert r.status_code == 200
    # /health stays open (no auth) and reports the mode.
    h = client.get("/health").json()
    assert h["auth"] == "google"


def test_two_users_fully_isolated(google_mode):
    client = google_mode
    body = "Photosynthesis converts light energy into chemical energy. " * 20
    assert _upload(client, "alice", "alice_notes.txt", body).status_code == 200
    assert _upload(client, "bob", "bob_notes.txt", "Newton's laws of motion. " * 30).status_code == 200

    # Each sees ONLY their own documents and stats.
    a_docs = client.get("/documents", headers={"Authorization": "Bearer alice"}).json()["documents"]
    b_docs = client.get("/documents", headers={"Authorization": "Bearer bob"}).json()["documents"]
    assert [d["source"] for d in a_docs] == ["alice_notes.txt"]
    assert [d["source"] for d in b_docs] == ["bob_notes.txt"]
    a_stats = client.get("/stats", headers={"Authorization": "Bearer alice"}).json()
    assert a_stats["documents"] == 1

    # Retrieval is fenced: every chunk Bob's queries can reach belongs to Bob.
    eng = api.get_engine()
    docs, conf, cites = eng._gather("laws of motion", 6, "", None, False, owner="gbob")
    assert docs and all(d.metadata.get("owner") == "gbob" for d in docs)
    docs_a, _, _ = eng._gather("photosynthesis", 6, "", None, False, owner="galice")
    assert docs_a and all(d.metadata.get("owner") == "galice" for d in docs_a)

    # Bob cannot delete Alice's document (owner-fenced delete is a no-op for him).
    client.post("/documents/delete", headers={"Authorization": "Bearer bob"},
                json={"source": "alice_notes.txt"})
    a_docs2 = client.get("/documents", headers={"Authorization": "Bearer alice"}).json()["documents"]
    assert [d["source"] for d in a_docs2] == ["alice_notes.txt"]

    # Alice's reset clears ONLY Alice.
    client.post("/reset", headers={"Authorization": "Bearer alice"})
    assert client.get("/documents", headers={"Authorization": "Bearer alice"}).json()["documents"] == []
    assert [d["source"] for d in client.get("/documents", headers={"Authorization": "Bearer bob"}).json()["documents"]] == ["bob_notes.txt"]


def test_convos_per_account(google_mode):
    client = google_mode
    client.put("/convos", headers={"Authorization": "Bearer alice"},
               json={"convos": [{"id": "a1", "title": "Alice chat", "messages": []}], "current": "a1"})
    a = client.get("/convos", headers={"Authorization": "Bearer alice"}).json()
    b = client.get("/convos", headers={"Authorization": "Bearer bob"}).json()
    assert a["convos"] and a["convos"][0]["title"] == "Alice chat"
    assert b["convos"] == []
