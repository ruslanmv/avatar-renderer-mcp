"""
mouth_artifact_cleanup.py — remove "mouth-within-a-mouth" / double-lip artifacts.

Lip-sync models sometimes open the mouth but leave a faint remnant of the
original closed mouth (lips/teeth) inside the open cavity. This post-process
enhancement detects, per frame, the inner-mouth cavity and darkens lip/skin-
colored blobs that appear *inside* the dark cavity, while preserving real teeth,
the outer lips, and the rest of the face.

Stage: post_process, priority 5 — runs BEFORE eye/gaze (priority 10) and is
applied before face restoration when wired in the lip-sync path, so the restorer
doesn't sharpen a bad mouth texture.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path

from . import Enhancement, EnhancementContext, registry

log = logging.getLogger("avatar-renderer.mouth_cleanup")


def cleanup_frame(frame, face_box=None):
    """Remove double-lip remnants inside the open mouth of a single BGR frame.

    Args:
        frame: HxWx3 BGR image (numpy).
        face_box: optional (x1, y1, x2, y2). If None, the face is detected.
    Returns:
        The cleaned frame (numpy).
    """
    import cv2
    import numpy as np

    h, w = frame.shape[:2]

    # Locate the face (so the mouth ROI is accurate across different framings).
    if face_box is None:
        try:
            casc = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = casc.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 5, minSize=(60, 60))
            if len(faces):
                x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
                face_box = (x, y, x + fw, y + fh)
        except Exception:
            face_box = None
    if face_box is None:
        fx1, fy1, fx2, fy2 = int(w * 0.30), int(h * 0.10), int(w * 0.70), int(h * 0.95)
    else:
        fx1, fy1, fx2, fy2 = face_box
    fw, fh = fx2 - fx1, fy2 - fy1

    # Inner-mouth ROI: lower-centre of the face.
    mx1 = fx1 + int(fw * 0.28)
    mx2 = fx1 + int(fw * 0.72)
    my1 = fy1 + int(fh * 0.60)
    my2 = fy1 + int(fh * 0.88)
    mx1, my1 = max(0, mx1), max(0, my1)
    mx2, my2 = min(w, mx2), min(h, my2)
    if mx2 - mx1 < 8 or my2 - my1 < 6:
        return frame

    roi = frame[my1:my2, mx1:mx2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]

    # The open cavity is the darker interior; only act when a real cavity exists.
    dark = (v < max(60, int(np.percentile(v, 35)))).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cavity_ratio = float((dark > 0).mean())
    if cavity_ratio < 0.06:  # mouth essentially closed → nothing to clean
        return frame

    # Candidate remnants: lip/skin-coloured (reddish) pixels that sit INSIDE the
    # dark cavity (i.e., a second mouth), but NOT bright teeth (high V).
    reddish = cv2.inRange(hsv, np.array([0, 35, 60]), np.array([22, 190, 200]))
    reddish |= cv2.inRange(hsv, np.array([160, 35, 60]), np.array([180, 190, 200]))
    inside = cv2.dilate(dark, np.ones((9, 9), np.uint8), iterations=1)
    artifact = cv2.bitwise_and(reddish, inside)
    artifact = cv2.morphologyEx(artifact, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    if int((artifact > 0).sum()) < 30:
        return frame

    # Darken the remnant toward the cavity colour and feather it in.
    feather = (cv2.GaussianBlur(artifact, (0, 0), sigmaX=3) / 255.0)[..., None]
    darkened = (roi.astype(np.float32) * 0.45)
    blended = (roi.astype(np.float32) * (1 - feather) + darkened * feather).astype(np.uint8)

    out = frame.copy()
    out[my1:my2, mx1:mx2] = blended
    return out


class MouthArtifactCleanupEnhancement(Enhancement):
    @property
    def name(self) -> str:
        return "mouth_artifact_cleanup"

    @property
    def stage(self) -> str:
        return "post_process"

    @property
    def priority(self) -> int:
        return 5  # before eye_gaze_blink (10) and gesture

    @property
    def description(self) -> str:
        return "Removes double-lip / mouth-within-mouth artifacts inside open mouths."

    def is_available(self) -> bool:
        try:
            import cv2  # noqa: F401
            import numpy  # noqa: F401

            return True
        except ImportError:
            return False

    def apply(self, ctx: EnhancementContext) -> EnhancementContext:
        import cv2

        if ctx.frames_dir is None or not ctx.frames_dir.exists():
            return ctx
        frame_files = sorted(glob.glob(str(ctx.frames_dir / "*.png")))
        if not frame_files:
            return ctx

        out_dir = ctx.tmp_dir / "mouth_cleanup_frames"
        out_dir.mkdir(parents=True, exist_ok=True)
        for fp in frame_files:
            frame = cv2.imread(fp)
            if frame is None:
                continue
            cv2.imwrite(str(out_dir / Path(fp).name), cleanup_frame(frame))

        ctx.frames_dir = out_dir
        ctx.video_path = None
        return ctx


registry.register(MouthArtifactCleanupEnhancement())
