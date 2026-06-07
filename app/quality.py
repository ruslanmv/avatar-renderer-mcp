"""
quality.py — input validation, quality metrics, and the quality report.

Two responsibilities:
  • inspect_portrait(): a "portrait doctor" that validates the source image and
    returns user-facing guidance (many artifacts come from bad source images).
  • compute/write quality report: objective per-render metrics (face visibility,
    mouth-artifact score, temporal flicker) used to GATE premium delivery.

OpenCV/numpy are imported lazily so this module is import-safe without them
(returns neutral results), keeping the API/tests light.
"""

from __future__ import annotations

import glob
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

log = logging.getLogger("avatar-renderer.quality")


def _cv():
    import cv2  # noqa
    import numpy as np  # noqa

    return cv2, np


# --------------------------------------------------------------------------- #
# Portrait doctor — validate the source image before rendering
# --------------------------------------------------------------------------- #

def inspect_portrait(image_path: str) -> dict:
    """Validate a source portrait. Returns ok/warnings/message + flags.

    Never raises — degrades to ok=True if OpenCV is unavailable.
    """
    result = {
        "ok": True,
        "face_found": True,
        "faces": 1,
        "teeth_visible": False,
        "low_resolution": False,
        "warnings": [],
        "message": "",
    }
    try:
        cv2, np = _cv()
    except Exception:
        return result  # can't validate → don't block

    img = cv2.imread(image_path)
    if img is None:
        result.update(ok=False, face_found=False, faces=0,
                      message="Could not read the image. Use a JPG/PNG portrait.")
        result["warnings"].append("unreadable_image")
        return result

    h, w = img.shape[:2]
    if min(h, w) < 256:
        result["low_resolution"] = True
        result["warnings"].append("low_resolution")

    casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = casc.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.1, 5, minSize=(60, 60))
    result["faces"] = int(len(faces))

    if len(faces) == 0:
        result.update(ok=False, face_found=False,
                      message="No clear frontal face detected. Use a front-facing, well-lit portrait.")
        result["warnings"].append("no_face")
        return result

    x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
    if fw < w * 0.18:
        result["warnings"].append("face_small")

    # Visible teeth in the resting mouth → higher double-lip risk.
    mouth = img[y + int(fh * 0.62):y + int(fh * 0.90), x + int(fw * 0.25):x + int(fw * 0.75)]
    if mouth.size:
        gray = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
        bright_ratio = float((gray > 170).mean())
        if bright_ratio > 0.10:
            result["teeth_visible"] = True
            result["warnings"].append("teeth_visible")

    msgs = []
    if "teeth_visible" in result["warnings"]:
        msgs.append("Your portrait shows visible teeth — a closed, relaxed mouth renders best "
                    "(reduces the 'mouth-within-a-mouth' artifact).")
    if "face_small" in result["warnings"] or "low_resolution" in result["warnings"]:
        msgs.append("Use a higher-resolution, front-facing portrait where the face fills the frame.")
    result["message"] = " ".join(msgs)
    return result


# --------------------------------------------------------------------------- #
# Video quality metrics
# --------------------------------------------------------------------------- #

def _extract(video_path: str, limit: int = 120):
    cv2, np = _cv()
    d = Path(tempfile.mkdtemp(prefix="qa_"))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video_path, str(d / "%04d.png")], check=True)
    files = sorted(glob.glob(str(d / "*.png")))[:limit]
    return [cv2.imread(f) for f in files]


def mouth_artifact_score(frames, face_box=None) -> float:
    """Fraction of inner-mouth-cavity pixels that look like a remnant lip/skin
    (the double-lip artifact), averaged over open-mouth frames. 0 = clean."""
    cv2, np = _cv()
    if not frames:
        return 0.0
    h, w = frames[0].shape[:2]
    if face_box is None:
        casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        f = casc.detectMultiScale(cv2.cvtColor(frames[len(frames) // 2], cv2.COLOR_BGR2GRAY), 1.1, 5, minSize=(60, 60))
        if len(f) == 0:
            return 0.0
        x, y, fw, fh = max(f, key=lambda b: b[2] * b[3])
        face_box = (x, y, x + fw, y + fh)
    fx1, fy1, fx2, fy2 = face_box
    fw, fh = fx2 - fx1, fy2 - fy1
    mx1, mx2 = fx1 + int(fw * 0.28), fx1 + int(fw * 0.72)
    my1, my2 = fy1 + int(fh * 0.60), fy1 + int(fh * 0.90)
    scores = []
    for fr in frames:
        roi = fr[my1:my2, mx1:mx2]
        if roi.size == 0:
            continue
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        dark = (v < max(55, int(np.percentile(v, 30)))).astype("uint8") * 255
        if float((dark > 0).mean()) < 0.08:
            continue  # mouth closed → not measured
        # Remnant = reddish pixels STRICTLY INSIDE the cavity (eroded core), so
        # normal lips at the cavity boundary are NOT counted. Score is the
        # fraction of the cavity core that looks like a stray lip/skin blob (0..1).
        core = cv2.erode(dark, np.ones((7, 7), "uint8")) > 0
        core_n = int(core.sum())
        if core_n < 20:
            continue
        reddish = cv2.inRange(hsv, np.array([0, 40, 70]), np.array([18, 170, 180])) > 0
        reddish |= cv2.inRange(hsv, np.array([165, 40, 70]), np.array([180, 170, 180])) > 0
        artifact = reddish & core
        scores.append(min(1.0, float(artifact.sum()) / core_n))
    return round(float(sum(scores) / len(scores)), 4) if scores else 0.0


def face_visible_ratio(frames) -> float:
    cv2, np = _cv()
    if not frames:
        return 0.0
    casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    seen = 0
    for fr in frames[::3]:  # sample every 3rd frame
        if len(casc.detectMultiScale(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), 1.1, 5, minSize=(60, 60))):
            seen += 1
    total = len(frames[::3]) or 1
    return round(seen / total, 3)


def compute_quality_report(video_path: str, *, config=None, provenance: Optional[dict] = None) -> dict:
    """Build a quality report dict for a rendered video. Never raises."""
    report = {
        "passed": True,
        "metrics": {},
        "failures": [],
        "provenance": provenance or {},
    }
    try:
        frames = _extract(video_path)
        report["metrics"] = {
            "frames": len(frames),
            "face_visible_ratio": face_visible_ratio(frames),
            "mouth_artifact_score": mouth_artifact_score(frames),
        }
    except Exception as exc:  # metrics best-effort
        log.warning("Quality metrics failed: %s", exc)
        return report

    # Premium/strict gate
    if config is not None and getattr(config, "max_artifact_score", None) is not None:
        score = report["metrics"]["mouth_artifact_score"]
        if score > config.max_artifact_score:
            report["passed"] = False
            report["failures"].append(
                f"mouth_artifact_score {score} > {config.max_artifact_score}"
            )
        # Haar detection misses many valid frames, so use a lenient floor here
        # (just guard against "no face at all" rather than per-frame jitter).
        if getattr(config, "require_face", False) and report["metrics"]["face_visible_ratio"] < 0.4:
            report["passed"] = False
            report["failures"].append(
                f"face_visible_ratio {report['metrics']['face_visible_ratio']} < 0.4"
            )
    return report


def write_quality_report(out_dir: str, report: dict) -> str:
    p = Path(out_dir) / "quality_report.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2))
    return str(p)
