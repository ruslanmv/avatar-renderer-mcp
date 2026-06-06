"""
tests/test_auth.py
──────────────────
Tests for the additive Hugging Face auth layer:

  • users_db    – SQLite users / sessions / job-ownership round-trips
  • security    – session tokens + signed OAuth state
  • /me         – anonymous when AUTH is disabled (default)
  • full flow   – OAuth callback → session → protected /render-upload → /status
                  ownership, with the HF network calls mocked.

Auth is opt-in (settings.AUTH_ENABLED); the default-off path keeps the rest of
the suite and existing behaviour unchanged.
"""

from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from app import api as api_module
from app import auth_hf, security, users_db

ASSETS_DIR = Path(__file__).parent / "assets"
TEST_FACE = ASSETS_DIR / "alice.png"
TEST_WAV = ASSETS_DIR / "hello.wav"


# --------------------------------------------------------------------------- #
# users_db
# --------------------------------------------------------------------------- #
def test_users_db_roundtrip(tmp_path):
    db = str(tmp_path / "t.sqlite3")
    users_db.init_db(db)

    user = users_db.upsert_user(
        db, hf_user_id="hf-1", hf_username="alice", hf_email="a@b.c", hf_access_token="tok"
    )
    assert user["id"] >= 1
    # Upsert again updates in place (no duplicate row).
    user2 = users_db.upsert_user(
        db, hf_user_id="hf-1", hf_username="alice2", hf_email="a@b.c", hf_access_token="tok2"
    )
    assert user2["id"] == user["id"]
    assert user2["hf_username"] == "alice2"

    token = security.new_session_token()
    users_db.create_session(db, user_id=user["id"], session_token=token, ttl_hours=24)
    found = users_db.get_user_by_session(db, token)
    assert found is not None and found["id"] == user["id"]

    # Job ownership
    users_db.record_job(db, user_id=user["id"], job_id="job-1")
    assert users_db.get_job_owner(db, "job-1") == user["id"]
    users_db.update_job_status(db, job_id="job-1", status="finished", completed=True)
    jobs = users_db.list_user_jobs(db, user["id"])
    assert jobs and jobs[0]["job_id"] == "job-1" and jobs[0]["status"] == "finished"

    # Logout invalidates the session
    users_db.delete_session(db, token)
    assert users_db.get_user_by_session(db, token) is None


def test_expired_session_is_rejected(tmp_path):
    db = str(tmp_path / "t.sqlite3")
    users_db.init_db(db)
    user = users_db.upsert_user(
        db, hf_user_id="hf-2", hf_username="bob", hf_email=None, hf_access_token=None
    )
    token = security.new_session_token()
    users_db.create_session(db, user_id=user["id"], session_token=token, ttl_hours=-1)  # already expired
    assert users_db.get_user_by_session(db, token) is None


# --------------------------------------------------------------------------- #
# security helpers
# --------------------------------------------------------------------------- #
def test_security_helpers():
    assert security.new_session_token() != security.new_session_token()
    assert security.extract_bearer_token("Bearer abc123") == "abc123"
    assert security.extract_bearer_token("bearer abc123") == "abc123"
    assert security.extract_bearer_token("Basic x") is None
    assert security.extract_bearer_token(None) is None

    state = security.make_oauth_state("secret")
    assert security.verify_oauth_state("secret", state) is True
    assert security.verify_oauth_state("wrong", state) is False
    assert security.verify_oauth_state("secret", "tampered.sig") is False


def test_normalize_profile():
    p = auth_hf.normalize_profile({"sub": "42", "preferred_username": "carol", "email": "c@x.y"})
    assert p == {"hf_user_id": "42", "hf_username": "carol", "hf_email": "c@x.y", "picture": None}


# --------------------------------------------------------------------------- #
# /me with auth disabled (default behaviour)
# --------------------------------------------------------------------------- #
def test_me_anonymous_when_auth_disabled():
    client = TestClient(api_module.app)
    res = client.get("/me")
    assert res.status_code == 200
    body = res.json()
    assert body["authenticated"] is False
    assert body["authEnabled"] is False


# --------------------------------------------------------------------------- #
# Full OAuth → protected render → ownership flow (network mocked)
# --------------------------------------------------------------------------- #
@pytest.fixture
def auth_client(tmp_path, monkeypatch):
    db = str(tmp_path / "auth.sqlite3")
    users_db.init_db(db)

    s = api_module.settings
    monkeypatch.setattr(s, "AUTH_ENABLED", True)
    monkeypatch.setattr(s, "DATABASE_URL", db)
    monkeypatch.setattr(s, "HF_CLIENT_ID", "cid")
    monkeypatch.setattr(s, "HF_CLIENT_SECRET", "secret")
    monkeypatch.setattr(s, "HF_REDIRECT_URI", "http://testserver/auth/huggingface/callback")
    monkeypatch.setattr(s, "FRONTEND_URL", "http://frontend.example")

    # Mock the HF network calls.
    monkeypatch.setattr(
        api_module.auth_hf, "exchange_code_for_token", lambda **_: {"access_token": "hf-token"}
    )
    monkeypatch.setattr(
        api_module.auth_hf,
        "fetch_user_info",
        lambda **_: {"sub": "user-123", "preferred_username": "alice", "email": "alice@hf.co"},
    )

    # Stub the heavy pipeline so /render-upload completes instantly.
    def _fake_pipeline(*, face_image, audio, out_path, **_kwargs):
        Path(out_path).write_bytes(b"\x00" * 1024)
        return str(out_path)

    monkeypatch.setattr(api_module, "render_pipeline", _fake_pipeline)
    return TestClient(api_module.app), db


def _do_login(client) -> str:
    """Run the OAuth login+callback and return the issued session token."""
    login = client.get("/auth/huggingface/login", follow_redirects=False)
    assert login.status_code == 302
    authorize = login.headers["location"]
    state = parse_qs(urlparse(authorize).query)["state"][0]

    callback = client.get(
        "/auth/huggingface/callback",
        params={"code": "abc", "state": state},
        cookies={"hf_oauth_state": state},  # Secure cookie won't auto-send over http
        follow_redirects=False,
    )
    assert callback.status_code == 302
    redirect = callback.headers["location"]
    assert redirect.startswith("http://frontend.example")
    return parse_qs(urlparse(redirect).query)["session_token"][0]


def test_full_auth_render_flow(auth_client):
    client, db = auth_client
    token = _do_login(client)
    auth = {"Authorization": f"Bearer {token}"}

    # /me reflects the signed-in user.
    me = client.get("/me", headers=auth).json()
    assert me["authenticated"] is True
    assert me["username"] == "alice"
    assert "hf_access_token" not in me  # never leaked to the client

    # Unauthenticated render is rejected.
    with TEST_FACE.open("rb") as fimg, TEST_WAV.open("rb") as faud:
        anon = client.post(
            "/render-upload",
            files={"avatar": ("a.png", fimg, "image/png"), "audio": ("a.wav", faud, "audio/wav")},
        )
    assert anon.status_code == 401

    # Authenticated render succeeds and the MP4 is retrievable by the owner.
    with TEST_FACE.open("rb") as fimg, TEST_WAV.open("rb") as faud:
        r = client.post(
            "/render-upload",
            headers=auth,
            files={"avatar": ("a.png", fimg, "image/png"), "audio": ("a.wav", faud, "audio/wav")},
        )
    assert r.status_code == 200
    job_id = r.json()["jobId"]

    status = client.get(f"/status/{job_id}", headers=auth)
    assert status.status_code == 200
    assert status.headers["content-type"] == "video/mp4"

    # A different user cannot read someone else's job.
    other = users_db.upsert_user(
        db, hf_user_id="user-999", hf_username="mallory", hf_email=None, hf_access_token=None
    )
    other_token = security.new_session_token()
    users_db.create_session(db, user_id=other["id"], session_token=other_token, ttl_hours=24)
    forbidden = client.get(f"/status/{job_id}", headers={"Authorization": f"Bearer {other_token}"})
    assert forbidden.status_code == 403

    # Logout invalidates the session.
    assert client.post("/logout", headers=auth).status_code == 200
    assert client.get("/me", headers=auth).status_code == 401
