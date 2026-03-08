"""
Enhancement #5: LatentSync Latent-Space Lip-Sync
==================================================

End-to-end latent-space lip synchronization from ByteDance.
Operates entirely in latent space without intermediate motion representations,
producing sharper lip movements than Diff2Lip/Wav2Lip.

Pipeline stage: lip_sync
Priority: 20 (runs after MuseTalk, so only one lip-sync applies)

Models required:
  - LatentSync checkpoints in models/latentsync/
  - OR external_deps/LatentSync/ cloned repo

Outperforms Wav2Lip, VideoReTalking, DINet, and MuseTalk on HDTF and
VoxCeleb2 benchmarks.
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
_LATENTSYNC_MODEL_DIR = _MODEL_ROOT / "latentsync"
_EXT_DEPS_DIR = Path(os.environ.get("EXT_DEPS_DIR", "external_deps"))
_LATENTSYNC_REPO = _EXT_DEPS_DIR / "LatentSync"


def _latentsync_available() -> bool:
    """Check if LatentSync repo and models are present."""
    if not _LATENTSYNC_REPO.exists():
        return False
    # Check for inference script
    has_script = (
        (_LATENTSYNC_REPO / "inference.py").exists() or
        (_LATENTSYNC_REPO / "scripts" / "inference.py").exists() or
        (_LATENTSYNC_REPO / "run_inference.py").exists()
    )
    if not has_script:
        return False
    # Check for model weights
    if _LATENTSYNC_MODEL_DIR.exists():
        model_files = (
            list(_LATENTSYNC_MODEL_DIR.glob("*.pth")) +
            list(_LATENTSYNC_MODEL_DIR.glob("*.safetensors")) +
            list(_LATENTSYNC_MODEL_DIR.glob("*.ckpt"))
        )
        if model_files:
            return True
    # Check inside repo
    repo_ckpts = _LATENTSYNC_REPO / "checkpoints"
    return repo_ckpts.exists() and any(repo_ckpts.iterdir())


def _find_inference_script() -> Path:
    """Find the LatentSync inference script."""
    candidates = [
        _LATENTSYNC_REPO / "inference.py",
        _LATENTSYNC_REPO / "run_inference.py",
        _LATENTSYNC_REPO / "scripts" / "inference.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    return _LATENTSYNC_REPO / "inference.py"


class LatentSyncLipSyncEnhancement(Enhancement):
    """LatentSync: end-to-end latent-space lip synchronization."""

    @property
    def name(self) -> str:
        return "latentsync_lipsync"

    @property
    def stage(self) -> str:
        return "lip_sync"

    @property
    def priority(self) -> int:
        return 20

    @property
    def description(self) -> str:
        return (
            "LatentSync latent-space lip synchronization from ByteDance. "
            "Operates end-to-end in latent space for sharper lip movements "
            "than Diff2Lip/Wav2Lip. Best for high_quality mode."
        )

    def is_available(self) -> bool:
        return _latentsync_available()

    def apply(self, ctx: EnhancementContext) -> EnhancementContext:
        # Only apply in high_quality mode (MuseTalk handles real_time)
        if ctx.quality_mode == "real_time":
            logger.info("LatentSync skipped: quality_mode=real_time")
            return ctx

        # Skip if another lip-sync already applied
        if "musetalk_lipsync" in ctx.applied_enhancements:
            logger.info("LatentSync skipped: MuseTalk already applied")
            return ctx

        if ctx.frames_dir is None and ctx.video_path is None:
            logger.warning("LatentSync: no frames or video to process")
            return ctx

        # Build input video from frames if needed
        input_video = None
        if ctx.video_path and ctx.video_path.exists():
            input_video = str(ctx.video_path)
        elif ctx.frames_dir and ctx.frames_dir.exists():
            tmp_vid = ctx.tmp_dir / "latentsync_input.mp4"
            try:
                subprocess.check_call([
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-r", str(ctx.fps),
                    "-i", str(ctx.frames_dir / "%04d.png"),
                    "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(tmp_vid),
                ])
                input_video = str(tmp_vid)
            except Exception as e:
                logger.warning("Failed to create input video for LatentSync: %s", e)
                return ctx

        if input_video is None:
            return ctx

        output_video = str(ctx.tmp_dir / "latentsync_output.mp4")

        # Determine model path
        model_dir = str(_LATENTSYNC_MODEL_DIR) if _LATENTSYNC_MODEL_DIR.exists() else str(
            _LATENTSYNC_REPO / "checkpoints"
        )

        script = _find_inference_script()
        env = os.environ.copy()
        env["PYTHONPATH"] = (
            f"{str(_LATENTSYNC_REPO.resolve())}{os.pathsep}"
            f"{env.get('PYTHONPATH', '')}"
        )

        cmd = [
            sys.executable, str(script),
            "--video_path", input_video,
            "--audio_path", ctx.audio_path,
            "--output_path", output_video,
            "--checkpoint_dir", model_dir,
        ]

        logger.info("Running LatentSync...")

        try:
            result = subprocess.run(
                cmd, env=env,
                capture_output=True, text=True,
                timeout=600,
            )

            if result.returncode == 0 and Path(output_video).exists():
                ctx.video_path = Path(output_video)
                ctx.frames_dir = None
                logger.info("LatentSync lip-sync applied successfully")
            else:
                logger.warning(
                    "LatentSync failed (rc=%d): %s",
                    result.returncode, result.stderr[-500:],
                )
        except subprocess.TimeoutExpired:
            logger.warning("LatentSync timed out after 600s")
        except Exception as e:
            logger.warning("LatentSync error: %s", e)

        return ctx


# Auto-register
registry.register(LatentSyncLipSyncEnhancement())
