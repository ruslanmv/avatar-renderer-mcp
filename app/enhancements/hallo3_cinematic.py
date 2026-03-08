"""
Enhancement #6: Hallo3 Cinematic Quality Mode
===============================================

Hallo3 (CVPR 2025) uses Diffusion Transformer (DiT) architecture for
highly dynamic, cinematic-quality portrait animation from a single image + audio.

Pipeline stage: motion_driver (replaces FOMM entirely for cinematic mode)
Priority: 5 (highest priority motion driver — if enabled, it takes over)

Models required:
  - Hallo3 checkpoints in models/hallo3/
  - OR external_deps/hallo3/ cloned repo

This is the highest quality option but also the slowest (10-30s per frame).
Only activates when quality_mode is explicitly set to "cinematic".
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from . import Enhancement, EnhancementContext, registry

logger = logging.getLogger(__name__)

_MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "models"))
_HALLO3_MODEL_DIR = _MODEL_ROOT / "hallo3"
_EXT_DEPS_DIR = Path(os.environ.get("EXT_DEPS_DIR", "external_deps"))
_HALLO3_REPO = _EXT_DEPS_DIR / "hallo3"


def _hallo3_available() -> bool:
    """Check if Hallo3 repo and models are present."""
    if not _HALLO3_REPO.exists():
        return False
    # Check for inference script
    has_script = (
        (_HALLO3_REPO / "inference.py").exists() or
        (_HALLO3_REPO / "scripts" / "inference.py").exists()
    )
    if not has_script:
        return False
    # Check for model weights
    if _HALLO3_MODEL_DIR.exists():
        model_files = (
            list(_HALLO3_MODEL_DIR.glob("*.pth")) +
            list(_HALLO3_MODEL_DIR.glob("*.safetensors"))
        )
        return bool(model_files)
    # Check inside repo
    repo_ckpts = _HALLO3_REPO / "pretrained_models"
    return repo_ckpts.exists()


class Hallo3CinematicEnhancement(Enhancement):
    """Hallo3: Diffusion Transformer for cinematic-quality portrait animation."""

    @property
    def name(self) -> str:
        return "hallo3_cinematic"

    @property
    def stage(self) -> str:
        return "motion_driver"

    @property
    def priority(self) -> int:
        return 5  # Highest priority — if enabled, replaces other motion drivers

    @property
    def description(self) -> str:
        return (
            "Hallo3 (CVPR 2025) Diffusion Transformer for cinematic-quality portrait "
            "animation. Produces highly dynamic, expressive results but is computationally "
            "expensive. Best for offline content generation."
        )

    def is_available(self) -> bool:
        return _hallo3_available()

    def apply(self, ctx: EnhancementContext) -> EnhancementContext:
        # Only activate for explicit "cinematic" quality mode
        if ctx.quality_mode != "cinematic":
            logger.info("Hallo3 skipped: quality_mode=%s (requires 'cinematic')", ctx.quality_mode)
            return ctx

        if not ctx.face_image or not Path(ctx.face_image).exists():
            logger.warning("Hallo3: no face image available")
            return ctx

        output_dir = ctx.tmp_dir / "hallo3_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find inference script
        inference_script = _HALLO3_REPO / "inference.py"
        if not inference_script.exists():
            inference_script = _HALLO3_REPO / "scripts" / "inference.py"

        model_dir = str(_HALLO3_MODEL_DIR) if _HALLO3_MODEL_DIR.exists() else str(
            _HALLO3_REPO / "pretrained_models"
        )

        env = os.environ.copy()
        env["PYTHONPATH"] = (
            f"{str(_HALLO3_REPO.resolve())}{os.pathsep}"
            f"{env.get('PYTHONPATH', '')}"
        )

        output_video = str(output_dir / "hallo3_result.mp4")

        cmd = [
            sys.executable, str(inference_script),
            "--source_image", ctx.face_image,
            "--driving_audio", ctx.audio_path,
            "--output", output_video,
            "--model_dir", model_dir,
        ]

        # Add reference video for pose guidance if available
        if ctx.reference_video and Path(ctx.reference_video).exists():
            cmd.extend(["--pose_video", ctx.reference_video])

        logger.info("Running Hallo3 cinematic pipeline (this may take several minutes)...")

        try:
            result = subprocess.run(
                cmd, env=env,
                capture_output=True, text=True,
                timeout=1800,  # 30 minutes max for cinematic quality
            )

            if result.returncode == 0 and Path(output_video).exists():
                ctx.video_path = Path(output_video)
                ctx.frames_dir = None
                logger.info("Hallo3 cinematic rendering complete")
            else:
                logger.warning(
                    "Hallo3 failed (rc=%d): %s",
                    result.returncode, result.stderr[-500:],
                )
        except subprocess.TimeoutExpired:
            logger.warning("Hallo3 timed out after 30 minutes")
        except Exception as e:
            logger.warning("Hallo3 error: %s", e)

        return ctx


# Auto-register
registry.register(Hallo3CinematicEnhancement())
