"""
compliance.py — engine license registry + commercial-use guard.

Some engines are NOT licensed for commercial use (verified June 2026):
  - Wav2Lip: research/personal only (LRS2 training-data license)
  - CodeFormer: NTU S-Lab 1.0 (non-commercial)
  - LivePortrait: MIT code, but InsightFace buffalo_l weights restrict commercial

Commercial-safe: SadTalker (Apache-2.0), MuseTalk (MIT), LatentSync (OpenRAIL++),
GFPGAN (Apache-2.0), MediaPipe (Apache-2.0), Diff2Lip (verify upstream).
"""

from __future__ import annotations

MODEL_LICENSES = {
    "simple": {"commercial_ok": True, "license": "internal"},
    "sadtalker": {"commercial_ok": True, "license": "Apache-2.0"},
    "musetalk": {"commercial_ok": True, "license": "MIT"},
    "latentsync": {"commercial_ok": True, "license": "OpenRAIL++ (attribution)"},
    "diff2lip": {"commercial_ok": "verify", "license": "verify upstream"},
    "gfpgan": {"commercial_ok": True, "license": "Apache-2.0"},
    "mediapipe": {"commercial_ok": True, "license": "Apache-2.0"},
    # Not commercial-safe:
    "wav2lip": {"commercial_ok": False, "license": "research/personal only (LRS2)"},
    "wav2lip_fast": {"commercial_ok": False, "license": "research/personal only (LRS2 weights)"},
    "fullface": {"commercial_ok": False, "license": "uses Wav2Lip weights (LRS2)"},
    "codeformer": {"commercial_ok": "verify", "license": "NTU S-Lab 1.0 (non-commercial)"},
    "liveportrait": {"commercial_ok": "verify", "license": "MIT code; InsightFace buffalo_l restricts"},
    "hallo3": {"commercial_ok": "verify", "license": "verify upstream"},
}


def license_info(engine: str) -> dict:
    return MODEL_LICENSES.get(engine, {"commercial_ok": "verify", "license": "unknown"})


def is_commercial_safe(engine: str) -> bool:
    return MODEL_LICENSES.get(engine, {}).get("commercial_ok") is True


def assert_engine_allowed(engine: str, *, commercial: bool) -> None:
    """Raise if a non-commercial-safe engine is used in commercial mode."""
    info = MODEL_LICENSES.get(engine)
    if info is None:
        raise RuntimeError(f"No license metadata for engine '{engine}'.")
    if commercial and info["commercial_ok"] is not True:
        raise RuntimeError(
            f"Engine '{engine}' is not approved for commercial use "
            f"(license: {info['license']}). Choose a commercial-safe engine."
        )
