"""
modes.py — Quality-tier definitions (single source of truth).

Tiers map a user-facing quality mode to a concrete render contract: which
method/backend to use, whether a degraded fallback is allowed, encode params,
required enhancements, and the artifact-score gate for premium delivery.

Production rule:
    preview/real_time/standard → fallback allowed
    high_quality/premium/cinematic → STRICT: never deliver a degraded fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# Quality modes that must NOT silently fall back to a lower-quality renderer.
STRICT_QUALITY_MODES = {"high_quality", "premium", "cinematic"}


@dataclass(frozen=True)
class RenderConfig:
    name: str
    method: str                 # render method (see app.render._METHODS)
    allow_fallback: bool        # may we deliver simple/ffmpeg if the method fails?
    fps: int
    crf: int
    resolution: int
    required_enhancements: List[str] = field(default_factory=list)
    optional_enhancements: List[str] = field(default_factory=list)
    # Premium gate: max allowed mouth-artifact score (0..1). None = no gate.
    max_artifact_score: Optional[float] = None
    require_face: bool = False   # fail if no face detected in the source


RENDER_CONFIGS = {
    "preview": RenderConfig(
        name="preview", method="simple", allow_fallback=True,
        fps=25, crf=23, resolution=720,
    ),
    "real_time": RenderConfig(
        name="real_time", method="wav2lip_fast", allow_fallback=True,
        fps=25, crf=22, resolution=768,
        required_enhancements=["mouth_artifact_cleanup"],
    ),
    "standard": RenderConfig(
        name="standard", method="pipeline", allow_fallback=True,
        fps=25, crf=20, resolution=768,
        required_enhancements=["mouth_artifact_cleanup"],
        optional_enhancements=["eye_gaze_blink"],
    ),
    # NOTE: mouth_artifact_score is recorded in the report for observability, but
    # it is NOT used as a hard gate — color heuristics can't reliably tell a real
    # double-lip remnant from a normal smiling mouth (a landmark/ML detector is
    # the proper future gate). Strict tiers still enforce: no degraded fallback,
    # and a face must be present.
    "high_quality": RenderConfig(
        name="high_quality", method="pipeline", allow_fallback=False,
        fps=25, crf=18, resolution=768,
        required_enhancements=["mouth_artifact_cleanup"],
        optional_enhancements=["eye_gaze_blink"],
        max_artifact_score=None, require_face=True,
    ),
    "premium": RenderConfig(
        name="premium", method="pipeline", allow_fallback=False,
        fps=30, crf=16, resolution=1080,
        required_enhancements=["mouth_artifact_cleanup", "eye_gaze_blink"],
        optional_enhancements=["emotion_expressions"],
        max_artifact_score=None, require_face=True,
    ),
    "cinematic": RenderConfig(
        name="cinematic", method="pipeline", allow_fallback=False,
        fps=30, crf=16, resolution=1080,
        required_enhancements=["mouth_artifact_cleanup", "eye_gaze_blink"],
        max_artifact_score=None, require_face=True,
    ),
}

# Back-compat: "auto" behaves like standard (fallback allowed).
RENDER_CONFIGS["auto"] = RENDER_CONFIGS["standard"]


def get_render_config(quality_mode: Optional[str]) -> RenderConfig:
    """Resolve a quality mode to its RenderConfig (defaults to standard)."""
    return RENDER_CONFIGS.get((quality_mode or "standard").strip().lower(), RENDER_CONFIGS["standard"])


def is_strict(quality_mode: Optional[str]) -> bool:
    return (quality_mode or "").strip().lower() in STRICT_QUALITY_MODES
