"""
pipeline.py  – FOMM + Diff2Lip (+ SadTalker / Wav2Lip fallback)

Input:
    face_image       – path to PNG/JPG portrait (one face, frontish)
    audio            – path to 16‑kHz mono WAV
    reference_video  – optional short driver MP4 (for full-body motion)
    out_path         – target MP4 file

The function must raise on any hard failure (so Celery -> WebSocket can emit
‘failed’).  It returns the absolute path of the finished MP4.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import torch

# --------------------------------------------------------------------------- #
# Model roots (mounted as read‑only volume in Kubernetes)
# --------------------------------------------------------------------------- #
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "/models"))

FOMM_CKPT   = MODEL_ROOT / "fomm" / "vox-cpk.pth.tar"
D2L_CKPT    = MODEL_ROOT / "diff2lip" / "Diff2Lip.pth"
SAD_CKPT    = MODEL_ROOT / "sadtalker" / "sadtalker.pth"
W2L_CKPT    = MODEL_ROOT / "wav2lip" / "wav2lip_gan.pth"
GFPGAN_CKPT = MODEL_ROOT / "gfpgan" / "GFPGANv1.3.pth"

# --------------------------------------------------------------------------- #
# Helpers (FOMM, Diff2Lip, SadTalker, Wav2Lip)                                #
# These can live in separate files; kept inline for readability.              #
# --------------------------------------------------------------------------- #


def run_fomm(face_img: str, audio_wav: str, ref_video: Optional[str], tmp_dir: Path) -> Path:
    """
    Generates a sequence of RGB frames (PNG) with head pose & expression.

    Returns: path to directory full of 0001.png, 0002.png, ...
    """
    import first_order_model.demo as fomm_demo  # type: ignore

    frames_dir = tmp_dir / "fomm_frames"
    frames_dir.mkdir(exist_ok=True)

    # Build CLI‑like args object
    class _Args:
        source_image = face_img
        driving_audio = audio_wav
        driving_video = ref_video
        result_dir = str(frames_dir)
        checkpoint = str(FOMM_CKPT)
        # … other FOMM flags default inside demo.py

    fomm_demo.main(_Args())
    return frames_dir


def run_diff2lip(frames_dir: Path, audio_wav: str, tmp_dir: Path) -> Path:
    """
    Diffuses viseme‑accurate mouth region over existing frames.
    Returns new dir of frames.  Falls back to Wav2Lip if GPU OOM or CKPT missing.
    """
    try:
        import diff2lip.inference as d2l  # type: ignore

        out_dir = tmp_dir / "d2l_frames"
        out_dir.mkdir(exist_ok=True)
        d2l.run_diffusion(
            ckpt=str(D2L_CKPT),
            frames=str(frames_dir),
            audio=audio_wav,
            out_dir=str(out_dir),
            steps=25,
        )
        return out_dir

    except (ImportError, RuntimeError) as e:
        print(f"[pipeline] Diff2Lip unavailable – falling back to Wav2Lip: {e}")
        return run_wav2lip(frames_dir, audio_wav, tmp_dir)


def run_sadtalker(face_img: str, audio_wav: str, tmp_dir: Path) -> Path:
    """
    Generates coarse talking‑head frames (head + expression) using SadTalker.
    """
    import sadtalker.inference as sad  # type: ignore

    out_dir = tmp_dir / "sadtalker_frames"
    out_dir.mkdir(exist_ok=True)
    sad.generate(
        cfg_path=str(SAD_CKPT),
        img=face_img,
        audio=audio_wav,
        out_dir=str(out_dir),
        enhancer_ckpt=str(GFPGAN_CKPT),
        still=True,
    )
    return out_dir


def run_wav2lip(frames_dir: Path, audio_wav: str, tmp_dir: Path) -> Path:
    """
    Refines mouth region with Wav2Lip GAN. Returns dir with final frames.
    """
    import wav2lip.inference as w2l  # type: ignore

    out_dir = tmp_dir / "w2l_frames"
    out_dir.mkdir(exist_ok=True)
    w2l.run_wav2lip(
        ckpt=str(W2L_CKPT),
        frames=str(frames_dir),
        audio=audio_wav,
        out_dir=str(out_dir),
    )
    return out_dir


def encode_mp4(frames_dir: Path, audio_wav: str, out_mp4: str) -> None:
    """
    Uses FFmpeg NVENC to combine PNG sequence + WAV into H.264 MP4.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-thread_queue_size",
        "1024",
        "-r",
        "25",
        "-i",
        f"{frames_dir}/%04d.png",
        "-i",
        audio_wav,
        "-c:v",
        "h264_nvenc",
        "-profile:v",
        "high",
        "-pix_fmt",
        "yuv420p",
        "-b:v",
        "6M",
        "-c:a",
        "aac",
        "-shortest",
        out_mp4,
    ]
    subprocess.check_call(cmd)


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def render_pipeline(
    *,
    face_image: str,
    audio: str,
    out_path: str,
    reference_video: Optional[str] = None,
) -> str:
    """
    High‑level entry used by Celery and the MCP server.

    Tries modern pipeline first (FOMM + Diff2Lip); if unavailable
    (e.g. not enough VRAM) drops to SadTalker + Wav2Lip.
    """
    tmp = Path(tempfile.mkdtemp(prefix="vidgen_"))

    try:
        if torch.cuda.is_available() and FOMM_CKPT.exists():
            print("[pipeline] Using FOMM + Diff2Lip")
            fomm_frames = run_fomm(face_image, audio, reference_video, tmp)
            final_frames = run_diff2lip(fomm_frames, audio, tmp)
        else:
            print("[pipeline] Using SadTalker + Wav2Lip fallback (CPU/GPU‑low)")
            sad_frames = run_sadtalker(face_image, audio, tmp)
            final_frames = run_wav2lip(sad_frames, audio, tmp)

        encode_mp4(final_frames, audio, out_path)
        return os.path.abspath(out_path)

    finally:
        # Optional: clean tmp dirs or leave for debugging
        pass


# --------------------------------------------------------------------------- #
# CLI helper for manual smoke tests                                           #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse, sys

    p = argparse.ArgumentParser()
    p.add_argument("--face", required=True, help="Path to PNG/JPG of face")
    p.add_argument("--audio", required=True, help="Path to wav file 16 kHz mono")
    p.add_argument("--out", required=True, help="Output MP4 path")
    p.add_argument("--driver", help="Optional driver MP4 for full body")
    args = p.parse_args()

    try:
        render_pipeline(
            face_image=args.face,
            audio=args.audio,
            out_path=args.out,
            reference_video=args.driver,
        )
    except Exception as exc:
        print("❌  Render failed:", exc, file=sys.stderr)
        sys.exit(1)
    print("✅  Done →", args.out)
