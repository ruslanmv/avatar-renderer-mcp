"""
Enhancement #9: GestureLSM Co-Speech Gesture Animation
========================================================

Generates upper-body/hand gestures synchronized with speech using
GestureLSM (ICCV 2025) or similar co-speech gesture models.

Pipeline stage: post_process
Priority: 30 (runs after eye gaze and viseme, as it affects larger body area)

Models required:
  - GestureLSM checkpoints in models/gesturelsm/
  - OR external_deps/GestureLSM/ cloned repo

This enhancement generates gesture keypoints from audio + text,
then applies upper-body deformation to the avatar frames.
Best suited for half-body or full-body avatar images.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import Enhancement, EnhancementContext, registry

logger = logging.getLogger(__name__)

_MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "models"))
_GESTURE_MODEL_DIR = _MODEL_ROOT / "gesturelsm"
_EXT_DEPS_DIR = Path(os.environ.get("EXT_DEPS_DIR", "external_deps"))
_GESTURE_REPO = _EXT_DEPS_DIR / "GestureLSM"


def _gesturelsm_available() -> bool:
    """Check if GestureLSM repo and models are present."""
    if not _GESTURE_REPO.exists():
        return False
    has_script = (
        (_GESTURE_REPO / "inference.py").exists() or
        (_GESTURE_REPO / "scripts" / "inference.py").exists()
    )
    if not has_script:
        return False
    if _GESTURE_MODEL_DIR.exists():
        return bool(list(_GESTURE_MODEL_DIR.glob("*.pth")) or
                     list(_GESTURE_MODEL_DIR.glob("*.pt")))
    repo_ckpts = _GESTURE_REPO / "checkpoints"
    return repo_ckpts.exists()


def _generate_gesture_keypoints(
    audio_path: str,
    transcript: Optional[str],
    duration_sec: float,
    fps: int = 25,
    model_dir: str = "",
) -> Optional[List[Dict]]:
    """Generate gesture keypoints from audio/text.

    If GestureLSM is not available, generates procedural idle gestures
    (subtle shoulder sway, hand position changes).
    """
    # Try GestureLSM subprocess
    if _gesturelsm_available():
        try:
            script = _GESTURE_REPO / "inference.py"
            if not script.exists():
                script = _GESTURE_REPO / "scripts" / "inference.py"

            import tempfile
            import json

            output_json = Path(tempfile.mktemp(suffix=".json"))

            env = os.environ.copy()
            env["PYTHONPATH"] = (
                f"{str(_GESTURE_REPO.resolve())}{os.pathsep}"
                f"{env.get('PYTHONPATH', '')}"
            )

            cmd = [
                sys.executable, str(script),
                "--audio_path", audio_path,
                "--output_json", str(output_json),
                "--fps", str(fps),
            ]
            if transcript:
                cmd.extend(["--text", transcript])
            if model_dir:
                cmd.extend(["--model_dir", model_dir])

            result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)

            if result.returncode == 0 and output_json.exists():
                with open(output_json) as f:
                    return json.load(f)
        except Exception as e:
            logger.warning("GestureLSM inference failed: %s", e)

    # Fallback: generate procedural idle animation
    return _generate_procedural_gestures(duration_sec, fps)


def _generate_procedural_gestures(
    duration_sec: float,
    fps: int = 25,
) -> List[Dict]:
    """Generate subtle procedural idle body movement.

    Creates gentle shoulder sway and breathing-like torso movement.
    This keeps the avatar looking alive even without the full gesture model.
    """
    import math
    import random

    total_frames = int(duration_sec * fps)
    keypoints = []

    # Breathing cycle: ~12-20 breaths/min
    breath_freq = random.uniform(0.2, 0.33)  # Hz
    # Shoulder sway: very slow, subtle
    sway_freq = random.uniform(0.05, 0.12)  # Hz

    for frame_idx in range(total_frames):
        t = frame_idx / fps

        # Breathing: subtle vertical oscillation of shoulders
        breath = math.sin(2 * math.pi * breath_freq * t) * 1.5  # ±1.5 pixels

        # Shoulder sway: very gentle lateral movement
        sway = math.sin(2 * math.pi * sway_freq * t) * 2.0  # ±2 pixels

        keypoints.append({
            "frame": frame_idx,
            "shoulder_dy": breath,
            "torso_dx": sway,
            "torso_dy": breath * 0.5,
        })

    return keypoints


def _apply_body_motion(
    frame: "np.ndarray",
    keypoint: Dict,
    blend_factor: float = 0.5,
) -> "np.ndarray":
    """Apply subtle body motion to a frame using affine transformation.

    Only affects the lower 2/3 of the frame (body area), leaving the face untouched.
    """
    import cv2
    import numpy as np

    h, w = frame.shape[:2]

    dx = keypoint.get("torso_dx", 0.0) * blend_factor
    dy = keypoint.get("torso_dy", 0.0) * blend_factor + keypoint.get("shoulder_dy", 0.0) * blend_factor

    if abs(dx) < 0.2 and abs(dy) < 0.2:
        return frame

    result = frame.copy()

    # Only affect body region (below face)
    body_y_start = int(h * 0.50)
    body_region = result[body_y_start:].copy()

    # Apply subtle translation
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(
        body_region, M,
        (body_region.shape[1], body_region.shape[0]),
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Blend at the boundary for smooth transition
    blend_height = int(h * 0.05)
    result[body_y_start + blend_height:] = shifted[blend_height:]

    # Smooth blend zone
    for i in range(blend_height):
        alpha = i / blend_height
        result[body_y_start + i] = (
            shifted[i].astype(float) * alpha +
            result[body_y_start + i].astype(float) * (1 - alpha)
        ).clip(0, 255).astype(np.uint8)

    return result


class GestureAnimationEnhancement(Enhancement):
    """GestureLSM: co-speech gesture and upper-body animation."""

    @property
    def name(self) -> str:
        return "gesture_animation"

    @property
    def stage(self) -> str:
        return "post_process"

    @property
    def priority(self) -> int:
        return 30

    @property
    def description(self) -> str:
        return (
            "Co-speech gesture and upper-body animation using GestureLSM (ICCV 2025). "
            "Generates speech-synchronized hand gestures and body movements. "
            "Falls back to procedural idle animation (breathing, subtle sway) "
            "if the full model is not available."
        )

    def is_available(self) -> bool:
        # Always available via procedural fallback
        return True

    def apply(self, ctx: EnhancementContext) -> EnhancementContext:
        import cv2
        import glob as glob_module

        if ctx.frames_dir is None or not ctx.frames_dir.exists():
            logger.warning("Gesture animation: no frames to process")
            return ctx

        frame_files = sorted(glob_module.glob(str(ctx.frames_dir / "*.png")))
        if not frame_files:
            return ctx

        total_frames = len(frame_files)
        duration_sec = total_frames / ctx.fps

        # Generate gesture keypoints
        model_dir = str(_GESTURE_MODEL_DIR) if _GESTURE_MODEL_DIR.exists() else ""
        keypoints = _generate_gesture_keypoints(
            ctx.audio_path, ctx.transcript, duration_sec, ctx.fps, model_dir,
        )

        if not keypoints:
            logger.info("No gesture keypoints generated; skipping")
            return ctx

        ctx.gesture_keypoints = keypoints

        out_dir = ctx.tmp_dir / "gesture_frames"
        out_dir.mkdir(parents=True, exist_ok=True)

        processed = 0
        for idx, frame_path in enumerate(frame_files):
            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            if idx < len(keypoints):
                frame = _apply_body_motion(frame, keypoints[idx])
                processed += 1

            out_path = out_dir / Path(frame_path).name
            cv2.imwrite(str(out_path), frame)

        logger.info("Gesture animation: applied to %d/%d frames", processed, total_frames)
        ctx.frames_dir = out_dir
        ctx.video_path = None

        return ctx


# Auto-register
registry.register(GestureAnimationEnhancement())
