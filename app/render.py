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


# Named methods. The "pipeline" methods restore the dev-v0.1.25 architecture
# (FOMM → MuseTalk/LatentSync/Diff2Lip → Wav2Lip fallback → GFPGAN); they need
# the external repos + model weights present (full GPU build). The "in-process"
# methods (wav2lip*/fullface/simple) run on the lightweight ZeroGPU Space.
_PIPELINE_METHODS = {"pipeline", "musetalk", "latentsync", "diff2lip", "wav2lip_pipeline"}
_INPROC_METHODS = {"simple", "wav2lip", "wav2lip_gfpgan", "fullface"}
_METHODS = _PIPELINE_METHODS | _INPROC_METHODS


def _run_pipeline_method(method: str, *, face_image: str, audio: str, out_path: str) -> str:
    """Route to the original high-quality render_pipeline with the chosen lip-sync."""
    from .pipeline import render_pipeline  # heavy (torch) — lazy

    kw = dict(face_image=face_image, audio=audio, out_path=out_path, quality_mode="high_quality")
    if method == "musetalk":
        kw["enhancements"] = ["musetalk_lipsync"]
    elif method == "latentsync":
        kw["enhancements"] = ["latentsync_lipsync"]
    elif method == "wav2lip_pipeline":
        kw["force_wav2lip"] = True
    elif method in ("pipeline", "auto"):
        # Try the best available lip-sync model in priority order; render_pipeline
        # falls through MuseTalk → LatentSync → Diff2Lip → Wav2Lip internally.
        kw["enhancements"] = ["musetalk_lipsync", "latentsync_lipsync"]
    # "diff2lip": no lip_sync enhancement → render_pipeline's built-in Diff2Lip path
    return render_pipeline(**kw)


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
    if method in _PIPELINE_METHODS:
        return _run_pipeline_method(method, face_image=face_image, audio=audio, out_path=out_path)
    raise ValueError(f"Unknown method: {method}")


# ── Flexible engine catalog (compose a workflow) ─────────────────────────────
# Each engine maps to how it plugs into render_pipeline. Engines need their repo
# + weights present (full GPU build); on the lightweight Space they fall back.
LIPSYNC_ENGINES = {
    "auto": {"enh": ["musetalk_lipsync", "latentsync_lipsync"]},  # best available
    "musetalk": {"enh": ["musetalk_lipsync"]},
    "latentsync": {"enh": ["latentsync_lipsync"]},
    "diff2lip": {},                       # render_pipeline's built-in Diff2Lip
    "wav2lip": {"force_wav2lip": True},   # subprocess Wav2Lip (dev-v0.1.25 fallback)
    "wav2lip_fast": {"inproc": True},     # in-process Wav2Lip+GFPGAN (ZeroGPU)
}
MOTION_ENGINES = {
    "fomm": [],                           # render_pipeline default motion
    "liveportrait": ["liveportrait_driver"],
    "hallo3": ["hallo3_cinematic"],
    "gaussian": ["gaussian_splatting"],
    # "sadtalker": not yet wired as an enhancement (repo present; TODO)
}
EXTRA_ENGINES = {
    "gfpgan": [],                         # auto-applied if checkpoint present
    "cosyvoice": ["cosyvoice_tts"],
    "emotion": ["emotion_expressions"],
    "eye_gaze": ["eye_gaze_blink"],
    "gesture": ["gesture_animation"],
    "viseme": ["viseme_guided"],
    "mouth_cleanup": ["mouth_artifact_cleanup"],
}


def render_workflow(
    *, face_image: str, audio: str, out_path: str,
    lip_sync: str = "auto", motion_driver: str = "fomm",
    extras: Optional[List[str]] = None, transcript: Optional[str] = None,
) -> str:
    """Compose a render workflow from chosen engines and run it.

    lip_sync: auto|musetalk|latentsync|diff2lip|wav2lip|wav2lip_fast
    motion_driver: fomm|liveportrait|hallo3|gaussian
    extras: any of EXTRA_ENGINES keys (gfpgan/cosyvoice/emotion/eye_gaze/gesture/...)
    """
    extras = extras or []
    spec = LIPSYNC_ENGINES.get(lip_sync, LIPSYNC_ENGINES["auto"])

    # In-process fast path (works on the ZeroGPU Space, no external repos).
    if spec.get("inproc"):
        enh = [EXTRA_ENGINES[e][0] for e in extras if EXTRA_ENGINES.get(e)]
        return _try_lipsync(face_image, audio, out_path, use_gfpgan=True, enhancements=enh or None)

    # Full pipeline path (dev-v0.1.25): motion driver + lip-sync + extras.
    enh = list(MOTION_ENGINES.get(motion_driver, []))
    enh += spec.get("enh", [])
    for e in extras:
        enh += EXTRA_ENGINES.get(e, [])
    from .pipeline import render_pipeline  # heavy (torch) — lazy

    return render_pipeline(
        face_image=face_image, audio=audio, out_path=out_path,
        quality_mode="high_quality", force_wav2lip=spec.get("force_wav2lip", False),
        enhancements=list(dict.fromkeys(enh)) or None, transcript=transcript,
    )


# ── Engine-centric orchestrator (the multi-engine "best solution") ───────────
_LEGACY_ENGINE_ALIASES = {"wav2lip_gfpgan": "wav2lip_fast", "wav2lip_pipeline": "wav2lip"}


def select_engine(quality_mode: str, requested: str = "auto", commercial: bool = False) -> str:
    """Choose the lip-sync engine for a tier: explicit (validated) or best-available.

    Honors availability + commercial license. Strict tiers raise if nothing
    suitable is available (never silently downgrade).
    """
    from .compliance import assert_engine_allowed, is_commercial_safe
    from .engines import registry as eng
    from .modes import commercial_required, get_preferred_engines, is_strict

    commercial = commercial or commercial_required(quality_mode)
    strict = is_strict(quality_mode)
    req = _LEGACY_ENGINE_ALIASES.get((requested or "auto").strip().lower(),
                                     (requested or "auto").strip().lower())

    if req and req != "auto":
        if not eng.has(req):
            raise ValueError(f"Unknown engine: {req}")
        if not eng.get(req).is_available():
            raise RuntimeError(f"Engine '{req}' is not available on this deployment.")
        assert_engine_allowed(req, commercial=commercial)
        return req

    for name in get_preferred_engines(quality_mode):
        if not eng.has(name) or not eng.get(name).is_available():
            continue
        if commercial and not is_commercial_safe(name):
            continue
        return name

    if strict:
        raise RuntimeError(
            f"No available {'commercial-safe ' if commercial else ''}engine for "
            f"tier '{quality_mode}'. Install the premium engines (GPU build)."
        )
    for fb in ("wav2lip_fast", "simple"):
        if eng.has(fb) and eng.get(fb).is_available():
            return fb
    return "simple"


def orchestrate(
    *, face_image: str, audio: str, out_path: str,
    quality_mode: str = "standard", engine: str = "auto", commercial: bool = False,
) -> str:
    """Multi-engine render: select engine → run → (soft fallback) → quality gate/report."""
    from .compliance import license_info
    from .engines import registry as eng
    from .modes import get_render_config, is_strict
    from .simple_render import simple_render

    config = get_render_config(quality_mode)
    strict = is_strict(quality_mode)
    chosen = select_engine(quality_mode, engine, commercial)
    prov = {"quality_mode": quality_mode, "engine": chosen, "strict": strict,
            "commercial": commercial, "fallback_used": False,
            "license": license_info(chosen).get("license")}

    out = None
    try:
        log.info("Orchestrate: mode=%s engine=%s strict=%s", quality_mode, chosen, strict)
        out = eng.get(chosen)._run(face_image=face_image, audio=audio, out_path=out_path)
    except Exception as exc:
        if strict:
            raise RuntimeError(f"'{quality_mode}' render failed on engine '{chosen}': {exc}") from exc
        log.warning("Engine '%s' failed (%s); trying fallbacks.", chosen, exc)
        prov["fallback_used"] = True
        for fb in ("wav2lip_fast", "simple"):
            if fb == chosen or not (eng.has(fb) and eng.get(fb).is_available()):
                continue
            try:
                out = eng.get(fb)._run(face_image=face_image, audio=audio, out_path=out_path)
                prov["engine"] = fb
                break
            except Exception as fexc:
                log.warning("Fallback '%s' failed: %s", fb, fexc)
        if out is None:
            out = simple_render(face_image=face_image, audio=audio, out_path=out_path)
            prov["engine"] = "simple"

    try:
        from .quality import compute_quality_report, write_quality_report
        report = compute_quality_report(out, config=config, provenance=prov)
        write_quality_report(str(Path(out).parent), report)
        log.info("Quality report: passed=%s metrics=%s", report.get("passed"), report.get("metrics"))
        if strict and not report.get("passed", True):
            raise RuntimeError(f"'{quality_mode}' quality gate failed: {report.get('failures')}")
    except RuntimeError:
        raise
    except Exception as exc:
        log.warning("Quality report skipped: %s", exc)
    return out


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
        # Soft fallback chain: best models → in-process Wav2Lip+GFPGAN → ffmpeg.
        log.warning("Method '%s' failed (%s); trying fallbacks.", chosen, exc)
        provenance["fallback_used"] = True
        for fb in ("wav2lip_gfpgan", "simple"):
            if fb == chosen:
                continue
            try:
                out = render_method(fb, face_image=face_image, audio=audio, out_path=out_path, enhancements=eff_enh)
                provenance["method"] = fb
                break
            except Exception as fexc:
                log.warning("Fallback '%s' failed: %s", fb, fexc)
        if out is None:
            out = simple_render(face_image=face_image, audio=audio, out_path=out_path)
            provenance["method"] = "simple"

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
