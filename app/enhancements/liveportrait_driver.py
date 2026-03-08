"""
Enhancement #4: LivePortrait Expression-Controllable Motion Driver
===================================================================

Uses LivePortrait to generate expression-controllable motion from a single
portrait image. Can incorporate emotion data from Enhancement #1 to drive
contextually appropriate facial expressions.

Pipeline stage: motion_driver (alternative/supplement to FOMM)
Priority: 10

Models required:
  - LivePortrait checkpoints in models/liveportrait/ or external_deps/LivePortrait/

LivePortrait advantages over FOMM:
  - Precise control over individual expression parameters
  - Better identity preservation
  - Runs on 6-8GB VRAM
  - Supports gaze direction control
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

from . import Enhancement, EnhancementContext, registry

logger = logging.getLogger(__name__)

_MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "models"))
_LIVEPORTRAIT_MODEL_DIR = _MODEL_ROOT / "liveportrait"
_EXT_DEPS_DIR = Path(os.environ.get("EXT_DEPS_DIR", "external_deps"))
_LIVEPORTRAIT_REPO = _EXT_DEPS_DIR / "LivePortrait"


def _liveportrait_available() -> bool:
    """Check if LivePortrait repo and models are present."""
    if not _LIVEPORTRAIT_REPO.exists():
        return False
    # Check for inference script
    inference = _LIVEPORTRAIT_REPO / "inference.py"
    if not inference.exists():
        # Try alternative locations
        inference = _LIVEPORTRAIT_REPO / "src" / "inference.py"
    if not inference.exists():
        return False
    # Check for model weights
    if _LIVEPORTRAIT_MODEL_DIR.exists():
        return bool(list(_LIVEPORTRAIT_MODEL_DIR.glob("*.pth")) or
                     list(_LIVEPORTRAIT_MODEL_DIR.glob("*.safetensors")))
    # Check inside repo
    repo_ckpts = _LIVEPORTRAIT_REPO / "pretrained_weights"
    return repo_ckpts.exists()


def _build_expression_config(
    emotion: Optional[str],
    expression_params: Optional[Dict[str, float]],
) -> Dict[str, float]:
    """Convert emotion/expression data to LivePortrait control parameters.

    LivePortrait uses rotation (pitch, yaw, roll) and expression coefficients.
    We map our emotion expression params to LivePortrait's format.
    """
    config = {
        "pitch": 0.0,
        "yaw": 0.0,
        "roll": 0.0,
        "exp_coeff_scale": 1.0,
    }

    if expression_params is None:
        return config

    # Map emotion expressions to subtle head movements
    smile = expression_params.get("smile", 0.0)
    if smile > 0.3:
        config["pitch"] = -2.0  # Slight head tilt up when happy
    elif smile < -0.2:
        config["pitch"] = 3.0   # Head down when sad

    eyebrow_raise = expression_params.get("eyebrow_raise", 0.0)
    if eyebrow_raise > 0.5:
        config["pitch"] = -3.0  # Head back slightly when surprised

    # Scale expression intensity
    config["exp_coeff_scale"] = 1.0 + abs(smile) * 0.3

    return config


class LivePortraitDriverEnhancement(Enhancement):
    """LivePortrait: expression-controllable portrait animation from a single image."""

    @property
    def name(self) -> str:
        return "liveportrait_driver"

    @property
    def stage(self) -> str:
        return "motion_driver"

    @property
    def priority(self) -> int:
        return 10

    @property
    def description(self) -> str:
        return (
            "LivePortrait expression-controllable motion driver. Generates high-quality "
            "portrait animation with precise control over expressions, gaze, and head pose. "
            "Can incorporate emotion data for contextual expressions."
        )

    def is_available(self) -> bool:
        return _liveportrait_available()

    def apply(self, ctx: EnhancementContext) -> EnhancementContext:
        if not ctx.face_image or not Path(ctx.face_image).exists():
            logger.warning("LivePortrait: no face image available")
            return ctx

        if not ctx.audio_path or not Path(ctx.audio_path).exists():
            logger.warning("LivePortrait: no audio available")
            return ctx

        # Build expression config from emotion data (if Enhancement #1 ran)
        exp_config = _build_expression_config(
            ctx.detected_emotion,
            ctx.expression_params,
        )

        # Determine model directory
        model_dir = str(_LIVEPORTRAIT_MODEL_DIR) if _LIVEPORTRAIT_MODEL_DIR.exists() else str(
            _LIVEPORTRAIT_REPO / "pretrained_weights"
        )

        output_dir = ctx.tmp_dir / "liveportrait_frames"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find inference script
        inference_script = _LIVEPORTRAIT_REPO / "inference.py"
        if not inference_script.exists():
            inference_script = _LIVEPORTRAIT_REPO / "src" / "inference.py"

        env = os.environ.copy()
        env["PYTHONPATH"] = (
            f"{str(_LIVEPORTRAIT_REPO.resolve())}{os.pathsep}"
            f"{env.get('PYTHONPATH', '')}"
        )

        cmd = [
            sys.executable, str(inference_script),
            "--source_image", ctx.face_image,
            "--driving_audio", ctx.audio_path,
            "--output_dir", str(output_dir),
            "--model_dir", model_dir,
            "--pitch", str(exp_config["pitch"]),
            "--yaw", str(exp_config["yaw"]),
            "--roll", str(exp_config["roll"]),
        ]

        # Add reference video if available
        if ctx.reference_video and Path(ctx.reference_video).exists():
            cmd.extend(["--driving_video", ctx.reference_video])

        logger.info("Running LivePortrait with emotion=%s", ctx.detected_emotion)

        try:
            result = subprocess.run(
                cmd, env=env,
                capture_output=True, text=True,
                timeout=300,
            )

            if result.returncode == 0:
                # Check if frames were generated
                frames = sorted(output_dir.glob("*.png"))
                if frames:
                    ctx.frames_dir = output_dir
                    logger.info("LivePortrait generated %d frames", len(frames))
                else:
                    # Check for video output
                    videos = sorted(output_dir.glob("*.mp4"))
                    if videos:
                        ctx.video_path = videos[0]
                        logger.info("LivePortrait generated video: %s", videos[0])
                    else:
                        logger.warning("LivePortrait succeeded but no output found")
            else:
                logger.warning(
                    "LivePortrait failed (rc=%d): %s",
                    result.returncode, result.stderr[-500:],
                )
        except subprocess.TimeoutExpired:
            logger.warning("LivePortrait timed out after 300s")
        except Exception as e:
            logger.warning("LivePortrait error: %s", e)

        return ctx


# Auto-register
registry.register(LivePortraitDriverEnhancement())
