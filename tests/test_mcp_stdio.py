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

    def _fake_pipeline(*, face_image, audio, out_path, reference_video=None,
                      viseme_json=None, quality_mode="auto"):
        # write a trivial MP4 header so FFmpeg players don't choke
        TMP_OUT.write_bytes(b"\x00" * 1024)
        Path(out_path).write_bytes(TMP_OUT.read_bytes())
        return str(out_path)  # Return the output path like the real function

    # Patch the render_pipeline as imported in api.py
    monkeypatch.setattr("app.api.render_pipeline", _fake_pipeline, raising=False)
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
    # No need to patch AVATAR_DIR - the endpoint returns model status
    res = client.get("/avatars")
    assert res.status_code == 200

    # Verify response structure
    data = res.json()
    assert "status" in data
    assert "models" in data
    assert "system" in data
    assert "rendering_modes" in data
    assert data["status"] == "ready"


def test_render_flow(monkeypatch):
    # POST render request with correct payload format
    payload = {
        "avatarPath": str(TEST_FACE),
        "audioPath": str(TEST_WAV),
        "qualityMode": "auto"
    }
    res = client.post("/render", json=payload)
    assert res.status_code == 200

    body = res.json()
    job_id = body["jobId"]
    status_url = body["statusUrl"]
    # sanity
    UUID(job_id)  # throws if invalid
    assert status_url.endswith(job_id)

    # Immediately poll status – our stub writes output synchronously
    res2 = client.get(f"/status/{job_id}")
    # Must be 200 and an MP4
    assert res2.status_code == 200
    assert res2.headers["content-type"] == "video/mp4"
    # Response body should contain the same dummy bytes we wrote
    assert len(res2.content) >= 1024
