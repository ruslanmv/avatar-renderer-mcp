"""
tests/test_mcp_stdio.py
───────────────────────
Unit tests for the MCP STDIO server (``app.mcp_server``).

The heavy ``render_pipeline`` is monkey-patched with a fast stub, and the
server's request handler + worker loop are exercised in-process:

  •  unknown tool        → {"error": "unknown_tool"}
  •  missing params      → {"error": "missing_audio_path"}
  •  valid render_avatar → {"jobId": ..., "output": ..., "qualityMode": ...}

Importing ``app.mcp_server`` must NOT block on stdin or spawn threads — the
serving loop lives behind ``main()`` — so these tests can drive it directly.
"""

import threading
import time
from pathlib import Path

import pytest

import app.mcp_server as mcp

ASSETS_DIR = Path(__file__).parent / "assets"
TEST_FACE = ASSETS_DIR / "alice.png"
TEST_WAV = ASSETS_DIR / "hello.wav"


@pytest.fixture(autouse=True)
def _patch_pipeline(monkeypatch, tmp_path):
    """Replace render_pipeline with a stub that writes a trivial MP4."""

    def _fake_pipeline(*, face_image, audio, out_path, **_kwargs):
        Path(out_path).write_bytes(b"\x00" * 1024)
        return str(out_path)

    monkeypatch.setattr(mcp, "render_pipeline", _fake_pipeline, raising=True)


@pytest.fixture
def captured_replies(monkeypatch):
    """Capture everything the server would write to stdout."""
    replies: list[dict] = []
    monkeypatch.setattr(mcp, "_reply", replies.append, raising=True)
    return replies


def test_unknown_tool(captured_replies):
    mcp._handle_request({"tool": "nope", "params": {}})
    assert captured_replies == [{"error": "unknown_tool"}]


def test_missing_params(captured_replies):
    mcp._handle_request({"tool": "render_avatar", "params": {"avatar_path": "x.png"}})
    assert captured_replies == [{"error": "missing_audio_path"}]


def test_render_flow(captured_replies, tmp_path):
    """Full enqueue → worker render → reply round-trip."""
    # Run one pass of the worker loop in a background thread.
    mcp._stop_flag.clear()
    worker = threading.Thread(target=mcp._worker_loop, daemon=True)
    worker.start()
    try:
        out_path = tmp_path / "out.mp4"
        mcp._handle_request(
            {
                "tool": "render_avatar",
                "params": {
                    "avatar_path": str(TEST_FACE),
                    "audio_path": str(TEST_WAV),
                    "quality_mode": "real_time",
                    "out_path": str(out_path),
                },
            }
        )

        # Wait for the worker to produce a reply.
        deadline = time.time() + 5
        while not captured_replies and time.time() < deadline:
            time.sleep(0.02)
    finally:
        mcp._stop_flag.set()
        worker.join(timeout=2)

    assert len(captured_replies) == 1
    reply = captured_replies[0]
    assert "error" not in reply
    assert reply["output"] == str(tmp_path / "out.mp4")
    assert reply["qualityMode"] == "real_time"
    assert Path(reply["output"]).exists()
