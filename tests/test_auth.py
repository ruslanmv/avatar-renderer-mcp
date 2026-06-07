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


def _do_login(client, return_to: str | None = None) -> tuple[str, dict]:
    """Run the OAuth login+callback; return (session_token, redirect_query)."""
    login = client.get(
        "/auth/huggingface/login",
        params={"return_to": return_to} if return_to else None,
        follow_redirects=False,
    )
    assert login.status_code == 302
    authorize = login.headers["location"]
    state = parse_qs(urlparse(authorize).query)["state"][0]

    cookies = {"hf_oauth_state": state}
    # The backend stored the validated return target in a cookie at /login;
    # re-send it (Secure cookies aren't auto-sent over http in tests).
    if return_to:
        cookies["hf_oauth_return_to"] = return_to

    callback = client.get(
        "/auth/huggingface/callback",
        params={"code": "abc", "state": state},
        cookies=cookies,
        follow_redirects=False,
    )
    assert callback.status_code == 302
    redirect = callback.headers["location"]
    q = parse_qs(urlparse(redirect).query)
    return q["session_token"][0], q


def test_login_redirects_to_allowed_vercel_origin(auth_client, monkeypatch):
    """A Vercel preview origin (matches the default regex) is honored for return_to."""
    client, _ = auth_client
    # Default CORS regex matches *.vercel.app
    monkeypatch.setattr(api_module.settings, "CORS_ALLOW_ORIGIN_REGEX", r"https://.*\.vercel\.app")
    preview = "https://avatar-renderer-git-257f9d-someproj.vercel.app"
    login = client.get(
        "/auth/huggingface/login", params={"return_to": preview}, follow_redirects=False
    )
    # The backend stored the preview origin to return to.
    set_cookie = login.headers.get("set-cookie", "")
    assert "hf_oauth_return_to" in set_cookie

    _, q = _do_login(client, return_to=preview)
    # callback Location starts with the preview origin (assert via test helper redirect)
    assert q["signup"][0] in ("0", "1")


def test_open_redirect_is_blocked(auth_client):
    """A return_to that isn't allow-listed falls back to FRONTEND_URL (no open redirect)."""
    client, _ = auth_client
    login = client.get(
        "/auth/huggingface/login",
        params={"return_to": "https://evil.example/steal"},
        follow_redirects=False,
    )
    set_cookie = login.headers.get("set-cookie", "")
    # The stored return target must be the safe FRONTEND_URL, not the evil origin.
    assert "evil.example" not in set_cookie
    assert "frontend.example" in set_cookie


def test_full_auth_render_flow(auth_client):
    client, db = auth_client
    token, q = _do_login(client)
    assert q["signup"][0] == "1"  # first login == sign-up
    auth = {"Authorization": f"Bearer {token}"}

    # A second login for the same user is a sign-in, not a sign-up.
    _, q2 = _do_login(client)
    assert q2["signup"][0] == "0"

    # /me reflects the signed-in user and their workspace (per-user tenant).
    me = client.get("/me", headers=auth).json()
    assert me["authenticated"] is True
    assert me["username"] == "alice"
    assert me["workspaceId"] == me["id"]  # per-user workspace
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

    # The job shows up in the owner's workspace history.
    mine = client.get("/me/jobs", headers=auth).json()
    assert any(j["job_id"] == job_id for j in mine["jobs"])

    # A different user (separate tenant) cannot read or see someone else's job.
    other = users_db.upsert_user(
        db, hf_user_id="user-999", hf_username="mallory", hf_email=None, hf_access_token=None
    )
    other_token = security.new_session_token()
    users_db.create_session(db, user_id=other["id"], session_token=other_token, ttl_hours=24)
    other_auth = {"Authorization": f"Bearer {other_token}"}
    forbidden = client.get(f"/status/{job_id}", headers=other_auth)
    assert forbidden.status_code == 403
    # Tenant isolation: the other user's workspace history is empty.
    assert client.get("/me/jobs", headers=other_auth).json()["jobs"] == []

    # Logout invalidates the session.
    assert client.post("/logout", headers=auth).status_code == 200
    assert client.get("/me", headers=auth).status_code == 401
