"""
Enhancement #3: Eye Gaze + Blink Modeling
==========================================

Adds natural blink patterns (15-20 blinks/minute) and subtle gaze saccades
to rendered avatar frames. This is a pure post-processing step that operates
on individual frames using OpenCV.

Pipeline stage: post_process
Priority: 10 (runs early in post-processing)

No external models required — uses procedural generation with randomization
for natural-looking eye movements. Optionally uses dlib/mediapipe for precise
eye landmark detection.

This is one of the lowest-effort, highest-impact improvements: static eyes
are the #1 tell of a synthetic avatar.
"""

from __future__ import annotations

import logging
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import Enhancement, EnhancementContext, registry

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Blink Schedule Generation
# --------------------------------------------------------------------------- #

def generate_blink_schedule(
    duration_sec: float,
    fps: int = 25,
    blinks_per_minute: float = 17.0,
    min_interval_sec: float = 1.5,
    blink_duration_frames: int = 5,
) -> List[Dict]:
    """Generate a natural blink schedule.

    Returns a list of blink events:
        [{"frame": int, "duration_frames": int, "intensity": float}, ...]

    Blink timing follows a Poisson process with minimum inter-blink interval,
    matching human blink physiology (~15-20 blinks/min with slight irregularity).
    """
    total_frames = int(duration_sec * fps)
    avg_interval_sec = 60.0 / blinks_per_minute
    blinks = []
    t = random.uniform(0.5, 2.0)  # First blink after 0.5-2 seconds

    while t < duration_sec - 0.5:
        frame = int(t * fps)
        # Vary blink duration slightly (3-7 frames at 25fps = 120-280ms)
        dur = max(3, blink_duration_frames + random.randint(-2, 2))
        # Intensity varies: most blinks are full (1.0), some are partial
        intensity = random.choice([1.0, 1.0, 1.0, 0.7, 0.5])

        blinks.append({
            "frame": frame,
            "duration_frames": dur,
            "intensity": intensity,
        })

        # Next blink: Poisson-like with minimum interval
        interval = max(min_interval_sec, random.expovariate(1.0 / avg_interval_sec))
        t += interval

    return blinks


# --------------------------------------------------------------------------- #
# Gaze Saccade Schedule Generation
# --------------------------------------------------------------------------- #

def generate_gaze_schedule(
    duration_sec: float,
    fps: int = 25,
    saccades_per_second: float = 2.5,
) -> List[Dict]:
    """Generate a subtle gaze movement schedule.

    Returns a list of gaze events:
        [{"frame": int, "dx": float, "dy": float, "hold_frames": int}, ...]

    dx/dy are pixel offsets (typically ±1-3 pixels for subtle movement).
    Mimics human micro-saccades: small, quick eye movements during fixation.
    """
    total_frames = int(duration_sec * fps)
    schedule = []
    frame = 0

    while frame < total_frames:
        # Micro-saccade: very small eye movement
        dx = random.gauss(0, 1.2)  # ~1-2 pixel horizontal
        dy = random.gauss(0, 0.8)  # ~0.5-1 pixel vertical (less range)

        # Clamp to reasonable range
        dx = max(-3.0, min(3.0, dx))
        dy = max(-2.0, min(2.0, dy))

        # Hold position for a variable duration
        hold_frames = max(3, int(random.expovariate(saccades_per_second) * fps))

        schedule.append({
            "frame": frame,
            "dx": dx,
            "dy": dy,
            "hold_frames": hold_frames,
        })

        frame += hold_frames

    return schedule


# --------------------------------------------------------------------------- #
# Frame Processing
# --------------------------------------------------------------------------- #

def _find_eye_regions(frame: "np.ndarray") -> Optional[List[Tuple[int, int, int, int]]]:
    """Detect eye regions using OpenCV cascade classifier.

    Returns list of (x, y, w, h) bounding boxes for eyes, or None if detection fails.
    """
    try:
        import cv2
        # Try OpenCV's built-in eye cascade
        cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
        if not Path(cascade_path).exists():
            return None

        eye_cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect face first for better eye detection
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            # Try eye detection on full frame
            eyes = eye_cascade.detectMultiScale(gray, 1.1, 5)
            return list(eyes[:2]) if len(eyes) >= 2 else None

        # Search for eyes within the top half of the face
        x, y, w, h = faces[0]
        roi_gray = gray[y:y + h // 2, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)

        if len(eyes) >= 2:
            # Adjust coordinates to full frame
            result = []
            for ex, ey, ew, eh in eyes[:2]:
                result.append((ex + x, ey + y, ew, eh))
            return result

        return None
    except Exception:
        return None


def apply_blink_to_frame(
    frame: "np.ndarray",
    blink_progress: float,
    eye_regions: Optional[List[Tuple[int, int, int, int]]] = None,
) -> "np.ndarray":
    """Apply a blink effect to a single frame.

    blink_progress: 0.0 = eyes open, 1.0 = eyes fully closed
    Uses a smooth sine curve for natural eyelid motion.

    If eye_regions are detected, applies targeted darkening/narrowing.
    Otherwise, applies a subtle global effect on the upper face region.
    """
    import cv2
    import numpy as np

    if blink_progress <= 0.0:
        return frame

    # Smooth the blink with a sine curve (fast close, slow open)
    smooth_progress = math.sin(blink_progress * math.pi / 2)

    result = frame.copy()
    h, w = result.shape[:2]

    if eye_regions and len(eye_regions) >= 2:
        for (ex, ey, ew, eh) in eye_regions:
            # Darken the eye region progressively
            eyelid_height = int(eh * smooth_progress * 0.6)
            if eyelid_height > 0:
                # Create eyelid overlay (skin-colored darkening)
                overlay = result.copy()
                # Darken from top of eye downward
                cv2.rectangle(
                    overlay,
                    (ex, ey),
                    (ex + ew, ey + eyelid_height),
                    (0, 0, 0),
                    -1,
                )
                alpha = smooth_progress * 0.4  # Subtle blend
                result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)
    else:
        # Fallback: subtle darkening of upper face region (approximate eye area)
        eye_band_top = int(h * 0.30)
        eye_band_bottom = int(h * 0.45)
        overlay = result.copy()

        # Reduce brightness in eye band
        darken_factor = 1.0 - (smooth_progress * 0.15)
        overlay[eye_band_top:eye_band_bottom] = (
            overlay[eye_band_top:eye_band_bottom].astype(float) * darken_factor
        ).clip(0, 255).astype(np.uint8)

        result = overlay

    return result


def apply_gaze_shift(
    frame: "np.ndarray",
    dx: float,
    dy: float,
    eye_regions: Optional[List[Tuple[int, int, int, int]]] = None,
) -> "np.ndarray":
    """Apply a subtle gaze shift to a frame.

    Shifts the iris/pupil area by (dx, dy) pixels.
    If no eye regions detected, applies a very subtle affine shift
    to the eye band region.
    """
    import cv2
    import numpy as np

    if abs(dx) < 0.3 and abs(dy) < 0.3:
        return frame

    result = frame.copy()
    h, w = result.shape[:2]

    if eye_regions and len(eye_regions) >= 2:
        for (ex, ey, ew, eh) in eye_regions:
            # Extract eye ROI
            pad = 2
            y1 = max(0, ey - pad)
            y2 = min(h, ey + eh + pad)
            x1 = max(0, ex - pad)
            x2 = min(w, ex + ew + pad)

            eye_roi = result[y1:y2, x1:x2].copy()

            # Translate the eye region slightly
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(
                eye_roi, M,
                (eye_roi.shape[1], eye_roi.shape[0]),
                borderMode=cv2.BORDER_REPLICATE,
            )

            # Blend shifted back (very subtle)
            alpha = 0.5
            result[y1:y2, x1:x2] = cv2.addWeighted(shifted, alpha, eye_roi, 1 - alpha, 0)
    else:
        # Minimal fallback: tiny shift in eye band
        eye_y1 = int(h * 0.30)
        eye_y2 = int(h * 0.45)
        band = result[eye_y1:eye_y2].copy()
        M = np.float32([[1, 0, dx * 0.3], [0, 1, dy * 0.3]])
        shifted = cv2.warpAffine(
            band, M,
            (band.shape[1], band.shape[0]),
            borderMode=cv2.BORDER_REPLICATE,
        )
        result[eye_y1:eye_y2] = cv2.addWeighted(shifted, 0.3, band, 0.7, 0)

    return result


class EyeGazeBlinkEnhancement(Enhancement):
    """Adds natural blink patterns and subtle gaze saccades to avatar frames."""

    @property
    def name(self) -> str:
        return "eye_gaze_blink"

    @property
    def stage(self) -> str:
        return "post_process"

    @property
    def priority(self) -> int:
        return 10

    @property
    def description(self) -> str:
        return (
            "Adds natural blink patterns (15-20/min) and subtle gaze saccades "
            "to rendered frames. Zero external models needed — pure post-processing "
            "with massive realism improvement."
        )

    def is_available(self) -> bool:
        try:
            import cv2
            return True
        except ImportError:
            return False

    def apply(self, ctx: EnhancementContext) -> EnhancementContext:
        import cv2
        import glob as glob_module

        if ctx.frames_dir is None or not ctx.frames_dir.exists():
            # If we have a video but no frames, extract them first
            if ctx.video_path and ctx.video_path.exists():
                frames_dir = ctx.tmp_dir / "gaze_extracted_frames"
                frames_dir.mkdir(parents=True, exist_ok=True)
                try:
                    import subprocess
                    subprocess.check_call([
                        "ffmpeg", "-y", "-loglevel", "error",
                        "-i", str(ctx.video_path),
                        str(frames_dir / "%04d.png"),
                    ])
                    ctx.frames_dir = frames_dir
                except Exception as e:
                    logger.warning("Could not extract frames for gaze processing: %s", e)
                    return ctx
            else:
                logger.warning("No frames to process for eye gaze/blink")
                return ctx

        frame_files = sorted(glob_module.glob(str(ctx.frames_dir / "*.png")))
        if not frame_files:
            return ctx

        total_frames = len(frame_files)
        duration_sec = total_frames / ctx.fps

        # Generate schedules
        blink_schedule = generate_blink_schedule(duration_sec, ctx.fps)
        gaze_schedule = generate_gaze_schedule(duration_sec, ctx.fps)

        # Store schedules on context for other enhancements
        ctx.blink_schedule = [b["frame"] / ctx.fps for b in blink_schedule]
        ctx.gaze_schedule = gaze_schedule

        # Build per-frame lookup for blinks
        blink_map = {}  # frame_idx -> blink_progress (0-1)
        for blink in blink_schedule:
            start = blink["frame"]
            dur = blink["duration_frames"]
            intensity = blink["intensity"]
            for i in range(dur):
                frame_idx = start + i
                # Progress: 0 → 1 (closing) → 0 (opening)
                progress = 1.0 - abs(2.0 * i / dur - 1.0)
                blink_map[frame_idx] = progress * intensity

        # Build per-frame gaze lookup
        gaze_map = {}  # frame_idx -> (dx, dy)
        for gaze in gaze_schedule:
            start = gaze["frame"]
            for i in range(gaze["hold_frames"]):
                gaze_map[start + i] = (gaze["dx"], gaze["dy"])

        # Detect eye regions from the first frame (cache for all frames)
        first_frame = cv2.imread(frame_files[0])
        eye_regions = _find_eye_regions(first_frame) if first_frame is not None else None

        # Output directory
        out_dir = ctx.tmp_dir / "gaze_blink_frames"
        out_dir.mkdir(parents=True, exist_ok=True)

        processed = 0
        for idx, frame_path in enumerate(frame_files):
            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            modified = False

            # Apply blink if this frame is in a blink window
            if idx in blink_map:
                frame = apply_blink_to_frame(frame, blink_map[idx], eye_regions)
                modified = True

            # Apply gaze shift
            if idx in gaze_map:
                dx, dy = gaze_map[idx]
                frame = apply_gaze_shift(frame, dx, dy, eye_regions)
                modified = True

            # Write frame (always write to maintain frame numbering)
            out_path = out_dir / Path(frame_path).name
            cv2.imwrite(str(out_path), frame)
            if modified:
                processed += 1

        logger.info(
            "Eye gaze/blink applied: %d/%d frames modified (%d blinks, %d saccade events)",
            processed, total_frames, len(blink_schedule), len(gaze_schedule),
        )

        ctx.frames_dir = out_dir
        # Clear video_path since we've modified the frames
        ctx.video_path = None

        return ctx


# Auto-register
registry.register(EyeGazeBlinkEnhancement())
