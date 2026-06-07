"""
render.py – Renderer dispatcher.

Single entry point used by the API and the Celery worker. Chooses between:

  • the full ML pipeline (app.pipeline.render_pipeline) — FOMM/Diff2Lip/Wav2Lip,
    needs torch + model checkpoints (+ ideally a GPU); and
  • the dependency-light demo renderer (app.simple_render.simple_render) — ffmpeg
    only, always works on a free CPU Space.

Selection via the RENDER_BACKEND env var:
    "auto"   (default) – try the full pipeline, fall back to simple on any failure
    "full"            – full pipeline only (raise if unavailable)
    "simple"          – demo renderer only (fast, no ML deps)

Nothing heavy is imported at module load, so importing this module (and thus
app.api) never requires torch — that's what lets the lightweight Space boot.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

log = logging.getLogger("avatar-renderer.render")


def _backend() -> str:
    return (os.getenv("RENDER_BACKEND") or "auto").strip().lower()


def _try_lipsync(face_image, audio, out_path, enhancements=None, use_gfpgan=None, head_motion=False) -> str:
    from .lipsync import wav2lip_render  # heavy/lazy

    return wav2lip_render(
        face_image=face_image, audio=audio, out_path=out_path,
        enhancements=enhancements, use_gfpgan=use_gfpgan, head_motion=head_motion,
    )


# Named methods for the quality bake-off (selectable per request).
#   simple        – no lip-sync (static portrait + Ken Burns)        [baseline]
#   wav2lip       – mouth lip-sync, NO GFPGAN                         [blurry mouth]
#   wav2lip_gfpgan– mouth lip-sync + GFPGAN (sharp), static bg/face   [best mouth]
#   fullface      – wav2lip_gfpgan + whole-head motion (bg static) + blink
_METHODS = {"simple", "wav2lip", "wav2lip_gfpgan", "fullface"}


def render_method(method: str, *, face_image: str, audio: str, out_path: str) -> str:
    """Render with an explicitly chosen method (used for the comparison/bake-off)."""
    method = (method or "").strip().lower()
    from .simple_render import simple_render

    if method == "simple":
        return simple_render(face_image=face_image, audio=audio, out_path=out_path)
    if method == "wav2lip":
        return _try_lipsync(face_image, audio, out_path, use_gfpgan=False)
    if method == "wav2lip_gfpgan":
        return _try_lipsync(face_image, audio, out_path, use_gfpgan=True)
    if method == "fullface":
        return _try_lipsync(
            face_image, audio, out_path,
            use_gfpgan=True, head_motion=True, enhancements=["eye_gaze_blink"],
        )
    raise ValueError(f"Unknown method: {method}")


def _try_pipeline(**kwargs) -> str:
    from .pipeline import render_pipeline  # heavy import (torch) — lazy

    return render_pipeline(**kwargs)


def render(
    *,
    face_image: str,
    audio: str,
    out_path: str,
    reference_video: Optional[str] = None,
    viseme_json: Optional[str] = None,
    quality_mode: str = "auto",
    enhancements: Optional[List[str]] = None,
    transcript: Optional[str] = None,
    method: Optional[str] = None,
) -> str:
    """Render an avatar video, returning the output path.

    Order by RENDER_BACKEND:
      simple → ffmpeg only.
      lipsync → Wav2Lip (talking face) then ffmpeg fallback.
      full → FOMM/Diff2Lip pipeline then ffmpeg fallback.
      auto (default) → Wav2Lip → full pipeline → ffmpeg, whichever works first.
    """
    from .simple_render import simple_render

    # Explicit method (bake-off / UI selection) takes precedence.
    if method and method.strip().lower() in _METHODS and method.strip().lower() != "auto":
        try:
            return render_method(method, face_image=face_image, audio=audio, out_path=out_path)
        except Exception as exc:
            log.warning("Method '%s' failed (%s); falling back to ffmpeg renderer.", method, exc)
            return simple_render(face_image=face_image, audio=audio, out_path=out_path)

    backend = _backend()

    if backend == "simple":
        return simple_render(face_image=face_image, audio=audio, out_path=out_path)

    # Build the attempt order.
    attempts = []
    if backend in ("auto", "lipsync"):
        attempts.append(("wav2lip", lambda: _try_lipsync(face_image, audio, out_path, enhancements)))
    if backend in ("auto", "full"):
        attempts.append(
            (
                "pipeline",
                lambda: _try_pipeline(
                    face_image=face_image,
                    audio=audio,
                    out_path=out_path,
                    reference_video=reference_video,
                    viseme_json=viseme_json,
                    quality_mode=quality_mode,
                    enhancements=enhancements,
                    transcript=transcript,
                ),
            )
        )

    for name, fn in attempts:
        try:
            log.info("Rendering with '%s' backend...", name)
            return fn()
        except Exception as exc:
            log.warning("'%s' renderer failed: %s", name, exc)

    log.warning("All model renderers unavailable; using the ffmpeg demo renderer.")
    return simple_render(face_image=face_image, audio=audio, out_path=out_path)
