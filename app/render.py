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
) -> str:
    """Render an avatar video, returning the output path."""
    backend = _backend()

    if backend != "simple":
        try:
            from .pipeline import render_pipeline  # heavy import (torch) — lazy

            return render_pipeline(
                face_image=face_image,
                audio=audio,
                out_path=out_path,
                reference_video=reference_video,
                viseme_json=viseme_json,
                quality_mode=quality_mode,
                enhancements=enhancements,
                transcript=transcript,
            )
        except Exception as exc:  # missing torch/models, or a runtime failure
            if backend == "full":
                raise
            log.warning(
                "Full pipeline unavailable (%s); falling back to the demo renderer.",
                exc,
            )

    from .simple_render import simple_render

    return simple_render(face_image=face_image, audio=audio, out_path=out_path)
