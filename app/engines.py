"""
engines.py — lip-sync engine registry (capabilities + availability + run).

A thin, uniform layer over the existing renderers so the orchestrator can select
among engines by quality tier, availability, and commercial license. Engines that
need external repos + weights (the premium ones) report is_available() = False on
the lightweight ZeroGPU preview and become available on the full GPU build.

Engine roles (see docs/MULTI_ENGINE_FEASIBILITY.md):
  in-process (ZeroGPU preview): simple, wav2lip_fast, fullface
  full pipeline (GPU build):    diff2lip, musetalk, latentsync, wav2lip(pipeline)
  not-yet-wired:                sadtalker, liveportrait, hallo3
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List

from .compliance import license_info

log = logging.getLogger("avatar-renderer.engines")

_MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "models"))
_EXT = Path(os.environ.get("EXT_DEPS_DIR", "external_deps"))


@dataclass
class Engine:
    name: str
    kind: str                       # "inproc" | "pipeline"
    needs_video_input: bool         # lip-sync engines edit a video (need motion stage)
    requires_gpu: bool
    description: str
    _run: Callable[..., str]
    _available: Callable[[], bool]
    note: str = ""

    def is_available(self) -> bool:
        try:
            return bool(self._available())
        except Exception:
            return False

    @property
    def commercial_ok(self):
        return license_info(self.name).get("commercial_ok")

    def info(self) -> dict:
        lic = license_info(self.name)
        return {
            "name": self.name,
            "kind": self.kind,
            "available": self.is_available(),
            "requires_gpu": self.requires_gpu,
            "needs_video_input": self.needs_video_input,
            "commercial_ok": lic.get("commercial_ok"),
            "license": lic.get("license"),
            "description": self.description,
            "note": self.note,
        }


# ── availability probes ──────────────────────────────────────────────────────
def _have(*paths: Path) -> bool:
    return all(p.exists() for p in paths)


def _torch_ok() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _enh_available(name: str) -> bool:
    try:
        from .enhancements import registry as enh
        e = enh.get(name)
        return bool(e and e.is_available())
    except Exception:
        return False


def _fomm_ok() -> bool:
    return (_EXT / "first-order-model").exists()


# ── run adapters (delegate to existing renderers) ────────────────────────────
def _run_method(method: str):
    def _fn(*, face_image: str, audio: str, out_path: str) -> str:
        from .render import render_method
        return render_method(method, face_image=face_image, audio=audio, out_path=out_path)
    return _fn


def _run_pipeline(engine: str):
    def _fn(*, face_image: str, audio: str, out_path: str) -> str:
        from .render import render_method
        # render_method maps musetalk/latentsync/diff2lip/wav2lip_pipeline → render_pipeline
        m = {"wav2lip": "wav2lip_pipeline"}.get(engine, engine)
        return render_method(m, face_image=face_image, audio=audio, out_path=out_path)
    return _fn


class EngineRegistry:
    def __init__(self) -> None:
        self._engines: Dict[str, Engine] = {}

    def register(self, e: Engine) -> None:
        self._engines[e.name] = e

    def get(self, name: str) -> Engine:
        if name not in self._engines:
            raise KeyError(f"Unknown engine: {name}")
        return self._engines[name]

    def has(self, name: str) -> bool:
        return name in self._engines

    def list_available(self) -> List[str]:
        return [n for n, e in self._engines.items() if e.is_available()]

    def info_all(self) -> List[dict]:
        return [e.info() for e in self._engines.values()]


registry = EngineRegistry()

# In-process engines (run on ZeroGPU preview) -------------------------------- #
registry.register(Engine(
    "simple", "inproc", False, False, "Static portrait + audio (no lip-sync).",
    _run_method("simple"), lambda: shutil.which("ffmpeg") is not None,
))
registry.register(Engine(
    "wav2lip_fast", "inproc", False, True,
    "In-process Wav2Lip (full-face, dev-v0.1.25) + GFPGAN. ZeroGPU preview default.",
    _run_method("wav2lip_gfpgan"), _torch_ok,
))
registry.register(Engine(
    "wav2lip_raw", "inproc", False, True,
    "In-process Wav2Lip (full-face, dev-v0.1.25), no GFPGAN restore (raw model).",
    _run_method("wav2lip"), _torch_ok,
))
registry.register(Engine(
    "wav2lip_band", "inproc", False, True,
    "In-process Wav2Lip mouth-band blend on a GFPGAN'd static base (anti-flicker).",
    _run_method("wav2lip_band"), _torch_ok,
))
registry.register(Engine(
    "fullface", "inproc", False, True,
    "Wav2Lip + GFPGAN + whole-head motion (static background) + blink.",
    _run_method("fullface"), _torch_ok,
))

# Full-pipeline engines (need external repos + weights; full GPU build) ------ #
registry.register(Engine(
    "diff2lip", "pipeline", True, True,
    "FOMM motion + Diff2Lip diffusion lip-sync + GFPGAN.",
    _run_pipeline("diff2lip"),
    lambda: _fomm_ok() and _have(_EXT / "Diff2Lip")
    and (_have(_MODEL_ROOT / "diff2lip" / "Diff2Lip.pth")
         or _have(_MODEL_ROOT / "diff2lip" / "e7.24.1.3_model260000_paper.pt")),
))
registry.register(Engine(
    "musetalk", "pipeline", True, True,
    "FOMM motion + MuseTalk latent-space lip-sync (real-time).",
    _run_pipeline("musetalk"), lambda: _fomm_ok() and _enh_available("musetalk_lipsync"),
))
registry.register(Engine(
    "latentsync", "pipeline", True, True,
    "FOMM motion + LatentSync diffusion lip-sync (premium).",
    _run_pipeline("latentsync"), lambda: _fomm_ok() and _enh_available("latentsync_lipsync"),
))
registry.register(Engine(
    "wav2lip", "pipeline", True, True,
    "FOMM motion + Wav2Lip (research-only license).",
    _run_pipeline("wav2lip"),
    lambda: _fomm_ok() and _have(_EXT / "Wav2Lip", _MODEL_ROOT / "wav2lip" / "wav2lip_gan.pth"),
))

# Registered but not yet wired (appear in /engines as unavailable + reason) -- #
for _n, _desc, _note in [
    ("sadtalker", "Audio-driven single-image talking head (Apache).", "not yet wired as a backend"),
    ("liveportrait", "Expression/motion driver.", "non-commercial (InsightFace); not wired"),
    ("hallo3", "Cinematic Diffusion Transformer.", "not wired"),
]:
    registry.register(Engine(
        _n, "pipeline", True, True, _desc,
        _run_pipeline(_n), lambda: False, note=_note,
    ))
