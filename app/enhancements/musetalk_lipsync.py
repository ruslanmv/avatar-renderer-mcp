"""
Enhancement #2: MuseTalk Lip-Sync
==================================

Replaces Wav2Lip with MuseTalk v1.5 for real-time lip-sync at 30+ FPS
with significantly better visual quality.

Pipeline stage: lip_sync (alternative to Wav2Lip/Diff2Lip)
Priority: 10 (runs before LatentSync which is priority 20)

Models required:
  - MuseTalk checkpoints in models/musetalk/
  - OR external_deps/MuseTalk/ cloned repo

When enabled for real_time mode, this replaces Wav2Lip as the lip-sync engine.
The original Wav2Lip path is untouched and remains the fallback.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

from . import Enhancement, EnhancementContext, registry

logger = logging.getLogger(__name__)

_MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "models"))
_MUSETALK_MODEL_DIR = _MODEL_ROOT / "musetalk"
_EXT_DEPS_DIR = Path(os.environ.get("EXT_DEPS_DIR", "external_deps"))
_MUSETALK_REPO = _EXT_DEPS_DIR / "MuseTalk"


def _musetalk_models_present() -> bool:
    """Check if MuseTalk model files exist."""
    # MuseTalk needs: musetalk.json + pytorch_model.bin (or safetensors)
    if _MUSETALK_MODEL_DIR.exists():
        model_files = list(_MUSETALK_MODEL_DIR.glob("*.bin")) + list(_MUSETALK_MODEL_DIR.glob("*.safetensors"))
        if model_files:
            return True
    # Also check inside the repo's models/ dir
    if _MUSETALK_REPO.exists():
        repo_models = _MUSETALK_REPO / "models"
        if repo_models.exists():
            return True
    return False


def _musetalk_script() -> Path:
    """Find the MuseTalk inference script."""
    # Check for our wrapper first
    wrapper = _MUSETALK_REPO / "musetalk_wrapper.py"
    if wrapper.exists():
        return wrapper
    # Standard inference script
    inference = _MUSETALK_REPO / "inference.py"
    if inference.exists():
        return inference
    # realtime inference
    realtime = _MUSETALK_REPO / "realtime_inference.py"
    if realtime.exists():
        return realtime
    return _MUSETALK_REPO / "inference.py"


def _run_musetalk(
    video_path: str,
    audio_path: str,
    output_path: str,
    model_dir: str,
) -> bool:
    """Run MuseTalk inference as subprocess."""
    script = _musetalk_script()
    if not script.exists():
        logger.error("MuseTalk script not found at %s", script)
        return False

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{str(_MUSETALK_REPO.resolve())}{os.pathsep}"
        f"{env.get('PYTHONPATH', '')}"
    )

    cmd = [
        sys.executable, str(script),
        "--video_path", video_path,
        "--audio_path", audio_path,
        "--output_path", output_path,
        "--model_dir", model_dir,
        "--fps", "25",
        "--batch_size", "8",
    ]

    logger.info("Running MuseTalk: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd, env=env,
            capture_output=True, text=True,
            timeout=300,
        )
        if result.returncode == 0 and Path(output_path).exists():
            logger.info("MuseTalk succeeded")
            return True
        else:
            logger.warning("MuseTalk failed (rc=%d): %s", result.returncode, result.stderr[-500:])
            return False
    except subprocess.TimeoutExpired:
        logger.warning("MuseTalk timed out")
        return False
    except Exception as e:
        logger.warning("MuseTalk error: %s", e)
        return False


class MuseTalkLipSyncEnhancement(Enhancement):
    """MuseTalk v1.5 lip-sync: sharper, higher quality than Wav2Lip at same speed."""

    @property
    def name(self) -> str:
        return "musetalk_lipsync"

    @property
    def stage(self) -> str:
        return "lip_sync"

    @property
    def priority(self) -> int:
        return 10

    @property
    def description(self) -> str:
        return (
            "MuseTalk v1.5 real-time lip synchronization. Runs at 30+ FPS with "
            "significantly better lip quality than Wav2Lip. Near drop-in replacement "
            "for the real_time quality mode."
        )

    def is_available(self) -> bool:
        return _MUSETALK_REPO.exists() and _musetalk_models_present()

    def apply(self, ctx: EnhancementContext) -> EnhancementContext:
        # Only apply in real_time mode (for high_quality, LatentSync or Diff2Lip is preferred)
        if ctx.quality_mode not in ("real_time", "auto"):
            logger.info("MuseTalk skipped: quality_mode=%s (only applies to real_time/auto)", ctx.quality_mode)
            return ctx

        if ctx.frames_dir is None and ctx.video_path is None:
            logger.warning("MuseTalk: no frames or video to process")
            return ctx

        # Build input video from frames if needed
        input_video = None
        if ctx.video_path and ctx.video_path.exists():
            input_video = str(ctx.video_path)
        elif ctx.frames_dir and ctx.frames_dir.exists():
            # Convert frames to temporary video
            tmp_vid = ctx.tmp_dir / "musetalk_input.mp4"
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
                logger.warning("Failed to create input video for MuseTalk: %s", e)
                return ctx

        if input_video is None:
            return ctx

        # Determine model directory
        model_dir = str(_MUSETALK_MODEL_DIR) if _MUSETALK_MODEL_DIR.exists() else str(_MUSETALK_REPO / "models")

        output_video = str(ctx.tmp_dir / "musetalk_output.mp4")

        success = _run_musetalk(
            video_path=input_video,
            audio_path=ctx.audio_path,
            output_path=output_video,
            model_dir=model_dir,
        )

        if success and Path(output_video).exists():
            ctx.video_path = Path(output_video)
            # Clear frames_dir since we now have a video
            ctx.frames_dir = None
            logger.info("MuseTalk lip-sync applied successfully")
        else:
            logger.warning("MuseTalk failed; original lip-sync result preserved")

        return ctx


# Auto-register
registry.register(MuseTalkLipSyncEnhancement())
