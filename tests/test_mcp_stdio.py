"""
tests/test_render_api.py
────────────────────────
CPU‑only smoke‑test that boots the FastAPI app in‑process, monkey‑patches the
heavy pipeline with a fast stub, then exercises:

  •  GET /avatars
  •  POST /render
  •  GET /status/<jobId>

The whole round‑trip must finish in < 5 s on GitHub’s 2‑core runner so the
main CI workflow stays snappy.
"""

import json
import os
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

# --------------------------------------------------------------------------- #
# 1 · Spin the API in ‑process                                                #
# --------------------------------------------------------------------------- #
# NOTE: importing *api* before monkey‑patch would pull the real pipeline, so
# we patch first via `pytest.fixture(autouse=True)`.

TMP_OUT = Path(__file__).parent / "tmp_out.mp4"


@pytest.fixture(autouse=True)
def _patch_pipeline(monkeypatch):
    """
    Replace the GPU‑heavy `render_pipeline` with a stub that just writes an
    empty MP4 so the API flow can complete instantly.
    """

    def _fake_pipeline(face_image, audio, reference_video, out_path):
        # write a trivial MP4 header so FFmpeg players don’t choke
        TMP_OUT.write_bytes(b"\x00" * 1024)
        Path(out_path).write_bytes(TMP_OUT.read_bytes())

    # Patch inside *mcp_server* & *worker* import chain
    monkeypatch.setattr("app.pipeline.render_pipeline", _fake_pipeline, raising=False)
    yield
    if TMP_OUT.exists():
        TMP_OUT.unlink()


# Import after patching
from app import api as api_module  # noqa: E402  pylint: disable=wrong-import-position

client = TestClient(api_module.app)


# --------------------------------------------------------------------------- #
# 2 · Fixtures – test assets                                                  #
# --------------------------------------------------------------------------- #
ASSETS_DIR = Path(__file__).parent / "assets"
TEST_FACE = ASSETS_DIR / "alice.png"
TEST_WAV = ASSETS_DIR / "hello.wav"


# --------------------------------------------------------------------------- #
# 3 · Actual tests                                                             #
# --------------------------------------------------------------------------- #
def test_list_avatars(tmp_path, monkeypatch):
    # monkey‑patch AVATAR_DIR to point at our test asset
    monkeypatch.setattr(api_module, "AVATAR_DIR", tmp_path)
    (tmp_path / "alice.png").write_bytes(b"\x89PNG\r\n\x1a\n")  # stub PNG

    res = client.get("/avatars")
    assert res.status_code == 200
    assert res.json() == ["alice"]


def test_render_flow(monkeypatch):
    # 1) Patch AVATAR_DIR so the service “finds” our face asset
    monkeypatch.setattr(api_module, "AVATAR_DIR", TEST_FACE.parent)

    # 2) POST render request
    payload = {"avatarId": "alice", "voiceUrl": TEST_WAV.as_uri()}
    res = client.post("/render", json=payload)
    assert res.status_code == 200

    body = res.json()
    job_id = body["jobId"]
    status_url = body["statusUrl"]
    # sanity
    UUID(job_id)  # throws if invalid
    assert status_url.endswith(job_id)

    # 3) Immediately poll status – our stub writes output synchronously
    res2 = client.get(f"/status/{job_id}")
    # Must be 200 and an MP4
    assert res2.status_code == 200
    assert res2.headers["content-type"] == "video/mp4"
    # Response body should contain the same dummy bytes we wrote
    assert len(res2.content) >= 1024
