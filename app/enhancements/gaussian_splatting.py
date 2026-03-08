"""
Enhancement #10: 3D Gaussian Splatting (InsTaG) Support
=========================================================

Integrates InsTaG (CVPR 2025) for real-time 3D talking head generation
from just 5 seconds of video. This is a transformative capability that
enables view-consistent 3D avatars.

Pipeline stage: motion_driver (alternative to FOMM/LivePortrait)
Priority: 3 (highest priority when explicitly enabled)

Models required:
  - InsTaG checkpoints in models/instag/
  - OR external_deps/InsTaG/ cloned repo
  - Pre-trained 3D Gaussian model for the specific avatar

This enhancement requires a SHORT calibration video (5-10 seconds) of the
person to build their 3D Gaussian representation. Once built, the 3D model
can be driven by audio in real-time.

Workflow:
  1. [One-time] Build 3D Gaussian model from calibration video
  2. [Per-request] Drive the 3D model with audio input
  3. [Per-request] Render frames from desired viewpoint
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from . import Enhancement, EnhancementContext, registry

logger = logging.getLogger(__name__)

_MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "models"))
_INSTAG_MODEL_DIR = _MODEL_ROOT / "instag"
_GAUSSIAN_CACHE_DIR = _MODEL_ROOT / "gaussian_cache"
_EXT_DEPS_DIR = Path(os.environ.get("EXT_DEPS_DIR", "external_deps"))
_INSTAG_REPO = _EXT_DEPS_DIR / "InsTaG"


def _instag_available() -> bool:
    """Check if InsTaG repo and models are present."""
    if not _INSTAG_REPO.exists():
        return False
    has_script = (
        (_INSTAG_REPO / "inference.py").exists() or
        (_INSTAG_REPO / "render.py").exists() or
        (_INSTAG_REPO / "scripts" / "inference.py").exists()
    )
    if not has_script:
        return False
    if _INSTAG_MODEL_DIR.exists():
        return bool(list(_INSTAG_MODEL_DIR.glob("*.pth")) or
                     list(_INSTAG_MODEL_DIR.glob("*.ply")))
    return False


def _get_cached_gaussian(avatar_id: str) -> Optional[Path]:
    """Check if a pre-built 3D Gaussian model exists for this avatar."""
    cache_dir = _GAUSSIAN_CACHE_DIR / avatar_id
    if cache_dir.exists():
        ply_files = list(cache_dir.glob("*.ply"))
        if ply_files:
            return ply_files[0]
    return None


def _build_gaussian_model(
    calibration_video: str,
    avatar_id: str,
    output_dir: Path,
) -> Optional[Path]:
    """Build a 3D Gaussian model from a calibration video.

    This is a one-time operation that takes 5-15 minutes.
    The resulting .ply file is cached for future use.
    """
    train_script = _INSTAG_REPO / "train.py"
    if not train_script.exists():
        train_script = _INSTAG_REPO / "scripts" / "train.py"
    if not train_script.exists():
        logger.error("InsTaG train script not found")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{str(_INSTAG_REPO.resolve())}{os.pathsep}"
        f"{env.get('PYTHONPATH', '')}"
    )

    cmd = [
        sys.executable, str(train_script),
        "--video_path", calibration_video,
        "--output_dir", str(output_dir),
        "--model_dir", str(_INSTAG_MODEL_DIR),
    ]

    logger.info("Building 3D Gaussian model for avatar '%s' (this may take 5-15 minutes)...", avatar_id)

    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1200)
        if result.returncode == 0:
            ply_files = list(output_dir.glob("*.ply"))
            if ply_files:
                logger.info("3D Gaussian model built: %s", ply_files[0])
                return ply_files[0]
    except Exception as e:
        logger.error("Failed to build 3D Gaussian model: %s", e)

    return None


class GaussianSplattingEnhancement(Enhancement):
    """InsTaG: 3D Gaussian Splatting for real-time 3D talking heads."""

    @property
    def name(self) -> str:
        return "gaussian_splatting"

    @property
    def stage(self) -> str:
        return "motion_driver"

    @property
    def priority(self) -> int:
        return 3  # Highest priority when enabled

    @property
    def description(self) -> str:
        return (
            "InsTaG (CVPR 2025) 3D Gaussian Splatting for real-time 3D talking heads. "
            "Creates a 3D avatar from 5 seconds of calibration video, then drives it "
            "with audio in real-time. Transformative quality but requires initial setup."
        )

    def is_available(self) -> bool:
        return _instag_available()

    def apply(self, ctx: EnhancementContext) -> EnhancementContext:
        # Only activate for explicit "3d" quality mode
        if ctx.quality_mode != "3d":
            logger.info("3D Gaussian Splatting skipped: quality_mode=%s (requires '3d')", ctx.quality_mode)
            return ctx

        if not ctx.face_image:
            logger.warning("3D Gaussian Splatting: no face image")
            return ctx

        # Use face_image filename as avatar_id for caching
        avatar_id = Path(ctx.face_image).stem

        # Check for cached Gaussian model
        gaussian_path = _get_cached_gaussian(avatar_id)

        if gaussian_path is None and ctx.reference_video:
            # Build from calibration video
            cache_dir = _GAUSSIAN_CACHE_DIR / avatar_id
            gaussian_path = _build_gaussian_model(ctx.reference_video, avatar_id, cache_dir)

        if gaussian_path is None:
            logger.warning(
                "3D Gaussian Splatting: no cached model and no calibration video provided. "
                "Provide reference_video with 5+ seconds of the person speaking."
            )
            return ctx

        # Render frames from 3D model
        output_dir = ctx.tmp_dir / "gaussian_frames"
        output_dir.mkdir(parents=True, exist_ok=True)

        render_script = _INSTAG_REPO / "render.py"
        if not render_script.exists():
            render_script = _INSTAG_REPO / "inference.py"

        env = os.environ.copy()
        env["PYTHONPATH"] = (
            f"{str(_INSTAG_REPO.resolve())}{os.pathsep}"
            f"{env.get('PYTHONPATH', '')}"
        )

        cmd = [
            sys.executable, str(render_script),
            "--gaussian_path", str(gaussian_path),
            "--audio_path", ctx.audio_path,
            "--output_dir", str(output_dir),
            "--fps", str(ctx.fps),
        ]

        logger.info("Rendering 3D Gaussian avatar...")

        try:
            result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)

            if result.returncode == 0:
                frames = sorted(output_dir.glob("*.png"))
                if frames:
                    ctx.frames_dir = output_dir
                    ctx.video_path = None
                    logger.info("3D Gaussian rendering: %d frames generated", len(frames))
                else:
                    logger.warning("3D Gaussian rendering: no frames generated")
            else:
                logger.warning("3D Gaussian rendering failed: %s", result.stderr[-500:])
        except subprocess.TimeoutExpired:
            logger.warning("3D Gaussian rendering timed out")
        except Exception as e:
            logger.warning("3D Gaussian rendering error: %s", e)

        return ctx


# Auto-register
registry.register(GaussianSplattingEnhancement())
