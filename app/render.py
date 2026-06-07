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
from pathlib import Path
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


def render_method(method: str, *, face_image: str, audio: str, out_path: str, enhancements=None) -> str:
    """Render with an explicitly chosen method (used for the comparison/bake-off)."""
    method = (method or "").strip().lower()
    from .simple_render import simple_render

    enh = list(dict.fromkeys(enhancements or []))  # de-dup, preserve order
    if method == "simple":
        return simple_render(face_image=face_image, audio=audio, out_path=out_path)
    if method == "wav2lip":
        return _try_lipsync(face_image, audio, out_path, use_gfpgan=False, enhancements=enh or None)
    if method == "wav2lip_gfpgan":
        return _try_lipsync(face_image, audio, out_path, use_gfpgan=True, enhancements=enh or None)
    if method == "fullface":
        enh = list(dict.fromkeys(enh + ["eye_gaze_blink"]))
        return _try_lipsync(
            face_image, audio, out_path,
            use_gfpgan=True, head_motion=True, enhancements=enh,
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
    """Render an avatar video (tier-driven, with a strict production contract).

    The quality_mode resolves to a RenderConfig (app.modes). An explicit `method`
    overrides the tier's method (used by the bake-off / UI method selector).

    Production rule: strict tiers (high_quality/premium/cinematic) NEVER deliver a
    degraded ffmpeg fallback — they raise instead, and a quality_report.json is
    written next to the output (premium delivery is gated on it).
    """
    from .modes import get_render_config, is_strict
    from .simple_render import simple_render

    config = get_render_config(quality_mode)
    strict = is_strict(quality_mode)

    # Choose method: explicit method param wins, else the tier's method.
    chosen = (method or "").strip().lower()
    if not chosen or chosen == "auto" or chosen not in _METHODS:
        chosen = config.method

    # Merge enhancements: tier-required + caller-requested (de-duped).
    eff_enh = list(dict.fromkeys(list(config.required_enhancements) + list(enhancements or [])))

    provenance = {
        "quality_mode": (quality_mode or "standard"),
        "method": chosen,
        "strict": strict,
        "fallback_used": False,
        "enhancements": eff_enh,
    }

    out: Optional[str] = None
    try:
        log.info("Rendering: mode=%s method=%s strict=%s", quality_mode, chosen, strict)
        out = render_method(chosen, face_image=face_image, audio=audio, out_path=out_path, enhancements=eff_enh)
    except Exception as exc:
        if strict or not config.allow_fallback:
            # Never deliver a degraded result for a strict tier.
            raise RuntimeError(
                f"'{quality_mode}' render failed and fallback is disabled: {exc}"
            ) from exc
        log.warning("Method '%s' failed (%s); using ffmpeg fallback.", chosen, exc)
        provenance["fallback_used"] = True
        provenance["method"] = "simple"
        out = simple_render(face_image=face_image, audio=audio, out_path=out_path)

    # Quality report + premium gate (best-effort; never breaks non-strict).
    try:
        from .quality import compute_quality_report, write_quality_report

        report = compute_quality_report(out, config=config, provenance=provenance)
        write_quality_report(str(Path(out).parent), report)
        log.info("Quality report: passed=%s metrics=%s", report.get("passed"), report.get("metrics"))
        if strict and not report.get("passed", True):
            raise RuntimeError(f"'{quality_mode}' quality gate failed: {report.get('failures')}")
    except RuntimeError:
        raise
    except Exception as exc:
        log.warning("Quality report skipped: %s", exc)

    return out
