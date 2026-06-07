"""
tests/test_engines.py — multi-engine registry, license guard, and selection.

These run without a GPU or model weights (premium engines report unavailable).
"""

import pytest


def test_engine_registry_catalog():
    from app.engines import registry

    info = {e["name"]: e for e in registry.info_all()}
    # Core engines are registered with license metadata.
    for name in ("simple", "wav2lip_fast", "fullface", "musetalk", "latentsync", "diff2lip", "wav2lip"):
        assert name in info
        assert "available" in info[name] and "commercial_ok" in info[name]
    # simple is always available (ffmpeg-only).
    assert registry.get("simple").is_available() is True
    # premium engines need external repos/weights → unavailable in test env.
    assert registry.get("latentsync").is_available() is False


def test_license_guard():
    from app.compliance import assert_engine_allowed, is_commercial_safe

    assert is_commercial_safe("latentsync") and is_commercial_safe("musetalk")
    assert not is_commercial_safe("wav2lip") and not is_commercial_safe("codeformer")
    # Non-commercial engine blocked in commercial mode; allowed otherwise.
    with pytest.raises(RuntimeError):
        assert_engine_allowed("wav2lip", commercial=True)
    assert_engine_allowed("wav2lip", commercial=False)  # no raise
    assert_engine_allowed("latentsync", commercial=True)  # no raise


def test_select_engine_explicit_validates_availability():
    from app.render import select_engine

    # Explicit unavailable premium engine → error (no silent downgrade).
    with pytest.raises(RuntimeError):
        select_engine("standard", "latentsync")
    # Explicit available in-process engine works.
    assert select_engine("standard", "simple") == "simple"


def test_select_engine_auto_falls_to_available_inproc():
    from app.render import select_engine

    # standard/auto: premium unavailable in test env → falls to an in-process engine.
    chosen = select_engine("standard", "auto")
    assert chosen in ("wav2lip_fast", "simple")


def test_strict_premium_raises_when_no_engine_available():
    from app.render import select_engine

    # premium requires commercial-safe premium engines; none available in test env
    # → strict tier must raise rather than downgrade.
    with pytest.raises(RuntimeError):
        select_engine("premium", "auto")


def test_premium_blocks_noncommercial_explicit_engine():
    from app.render import select_engine

    # Even if asked explicitly, premium (commercial) refuses research-only Wav2Lip.
    with pytest.raises(RuntimeError):
        select_engine("premium", "wav2lip_fast")
