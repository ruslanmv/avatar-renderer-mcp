"""
tests/test_production.py — production-contract tests.

Covers the strict quality-tier behaviour (no degraded fallback for premium),
the modes config, and graceful degradation of the quality/portrait helpers.
These run without a GPU or model weights.
"""

import pytest


# --------------------------------------------------------------------------- #
# Quality-tier config
# --------------------------------------------------------------------------- #
def test_modes_config():
    from app.modes import get_render_config, is_strict, STRICT_QUALITY_MODES

    assert is_strict("premium") and is_strict("high_quality") and is_strict("cinematic")
    assert not is_strict("standard") and not is_strict("preview")
    assert STRICT_QUALITY_MODES == {"high_quality", "premium", "cinematic"}

    # Strict tiers forbid fallback; soft tiers allow it.
    assert get_render_config("premium").allow_fallback is False
    assert get_render_config("high_quality").allow_fallback is False
    assert get_render_config("preview").allow_fallback is True
    assert get_render_config("standard").allow_fallback is True

    # Unknown / auto resolve sensibly.
    assert get_render_config("nonsense").name == "standard"
    assert get_render_config("auto").name == "standard"

    # Premium has an artifact gate + requires a face.
    prem = get_render_config("premium")
    assert prem.max_artifact_score is not None and prem.require_face is True
    assert "mouth_artifact_cleanup" in prem.required_enhancements


# --------------------------------------------------------------------------- #
# Strict no-fallback contract
# --------------------------------------------------------------------------- #
def _boom(*a, **k):
    raise RuntimeError("simulated render failure")


def test_premium_refuses_fallback(monkeypatch, tmp_path):
    from app import render as R

    monkeypatch.setattr(R, "render_method", _boom)
    with pytest.raises(RuntimeError):
        R.render(
            face_image="x.png", audio="y.wav",
            out_path=str(tmp_path / "o.mp4"), quality_mode="premium",
        )


def test_high_quality_refuses_fallback(monkeypatch, tmp_path):
    from app import render as R

    monkeypatch.setattr(R, "render_method", _boom)
    with pytest.raises(RuntimeError):
        R.render(
            face_image="x.png", audio="y.wav",
            out_path=str(tmp_path / "o.mp4"), quality_mode="high_quality",
        )


def test_preview_allows_fallback(monkeypatch, tmp_path):
    from app import render as R
    import app.simple_render as S

    out = str(tmp_path / "o.mp4")
    monkeypatch.setattr(R, "render_method", _boom)

    def _fake_simple(*, face_image, audio, out_path, **k):
        with open(out_path, "wb") as f:
            f.write(b"\x00" * 16)
        return out_path

    monkeypatch.setattr(S, "simple_render", _fake_simple)
    # Avoid the (heavy/ffmpeg) quality step in this unit test.
    import app.quality as Q
    monkeypatch.setattr(Q, "compute_quality_report", lambda *a, **k: {"passed": True})

    res = R.render(face_image="x.png", audio="y.wav", out_path=out, quality_mode="preview")
    assert res == out


# --------------------------------------------------------------------------- #
# Quality + portrait helpers degrade gracefully (no GPU / maybe no cv2)
# --------------------------------------------------------------------------- #
def test_inspect_portrait_returns_shape():
    from app.quality import inspect_portrait

    r = inspect_portrait("/does/not/exist.png")
    assert set(["ok", "warnings", "message", "faces"]).issubset(r.keys())


def test_quality_report_shape():
    from app.quality import compute_quality_report

    rep = compute_quality_report("/does/not/exist.mp4", config=None, provenance={"x": 1})
    assert "passed" in rep and "metrics" in rep and rep["provenance"] == {"x": 1}


def test_mouth_cleanup_registered():
    from app.enhancements import registry

    enh = registry.get("mouth_artifact_cleanup")
    assert enh is not None and enh.stage == "post_process" and enh.priority == 5
