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

import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Any

import torch

# --------------------------------------------------------------------------- #
# Model roots (mounted as read‑only volume in Kubernetes)
# --------------------------------------------------------------------------- #
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "/models"))

FOMM_CKPT   = MODEL_ROOT / "fomm" / "vox-cpk.pth"
D2L_CKPT    = MODEL_ROOT / "diff2lip" / "Diff2Lip.pth"
SAD_CKPT    = MODEL_ROOT / "sadtalker" / "sadtalker.pth"
W2L_CKPT    = MODEL_ROOT / "wav2lip" / "wav2lip_gan.pth"
GFPGAN_CKPT = MODEL_ROOT / "gfpgan" / "GFPGANv1.3.pth"

# External dependencies base path
EXT_DEPS_DIR = Path(os.environ.get("EXT_DEPS_DIR", "external_deps"))

# --------------------------------------------------------------------------- #
# Helper function to dynamically load modules from file paths
# --------------------------------------------------------------------------- #

def load_module_from_path(module_name: str, file_path: Path) -> Any:
    """
    Dynamically load a Python module from a file path.

    This is necessary because external repos (SadTalker, Wav2Lip) are not
    proper Python packages and cannot be imported normally.

    Args:
        module_name: Name to assign to the loaded module
        file_path: Path to the Python file to load

    Returns:
        The loaded module object

    Raises:
        ImportError: If the module file doesn't exist or can't be loaded
    """
    if not file_path.exists():
        raise ImportError(f"Module file not found: {file_path}")

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from: {file_path}")

    module = importlib.util.module_from_spec(spec)

    # Add the module's directory to sys.path temporarily
    module_dir = str(file_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ImportError(f"Failed to execute module {file_path}: {e}") from e

    return module

# --------------------------------------------------------------------------- #
# Helpers (FOMM, Diff2Lip, SadTalker, Wav2Lip)                                #
# These can live in separate files; kept inline for readability.              #
# --------------------------------------------------------------------------- #


def run_fomm(face_img: str, audio_wav: str, ref_video: Optional[str], tmp_dir: Path) -> Path:
    """
    Generates a sequence of RGB frames (PNG) with head pose & expression.

    Returns: path to directory full of 0001.png, 0002.png, ...
    """
    # Try to load FOMM module dynamically
    try:
        fomm_path = EXT_DEPS_DIR / "first-order-model" / "demo.py"
        fomm_demo = load_module_from_path("fomm_demo", fomm_path)
    except (ImportError, FileNotFoundError) as e:
        raise RuntimeError(
            f"First Order Motion Model not found in {EXT_DEPS_DIR}/first-order-model. "
            f"Please install external dependencies: {e}"
        ) from e

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

    if hasattr(fomm_demo, 'main'):
        fomm_demo.main(_Args())
    else:
        raise RuntimeError("FOMM module loaded but 'main' function not found")

    return frames_dir


def run_diff2lip(frames_dir: Path, audio_wav: str, tmp_dir: Path) -> Path:
    """
    Diffuses viseme‑accurate mouth region over existing frames.
    Returns new dir of frames.  Falls back to Wav2Lip if GPU OOM or CKPT missing.
    """
    try:
        # Try to load Diff2Lip module dynamically
        diff2lip_path = EXT_DEPS_DIR / "Diff2Lip" / "inference.py"
        if not diff2lip_path.exists():
            # If Diff2Lip is not installed, fall back to Wav2Lip
            print(f"[pipeline] Diff2Lip not found at {diff2lip_path}, falling back to Wav2Lip")
            return run_wav2lip(frames_dir, audio_wav, tmp_dir)

        d2l = load_module_from_path("diff2lip_inference", diff2lip_path)

        out_dir = tmp_dir / "d2l_frames"
        out_dir.mkdir(exist_ok=True)

        if hasattr(d2l, 'run_diffusion'):
            d2l.run_diffusion(
                ckpt=str(D2L_CKPT),
                frames=str(frames_dir),
                audio=audio_wav,
                out_dir=str(out_dir),
                steps=25,
            )
        else:
            raise RuntimeError("Diff2Lip module loaded but 'run_diffusion' function not found")

        return out_dir

    except (ImportError, RuntimeError) as e:
        print(f"[pipeline] Diff2Lip unavailable – falling back to Wav2Lip: {e}")
        return run_wav2lip(frames_dir, audio_wav, tmp_dir)


def run_sadtalker(face_img: str, audio_wav: str, tmp_dir: Path) -> Path:
    """
    Generates coarse talking‑head frames (head + expression) using SadTalker.
    """
    # Try to load SadTalker module dynamically
    try:
        sadtalker_path = EXT_DEPS_DIR / "SadTalker" / "inference.py"
        sad = load_module_from_path("sadtalker_inference", sadtalker_path)
    except (ImportError, FileNotFoundError) as e:
        raise RuntimeError(
            f"SadTalker not found in {EXT_DEPS_DIR}/SadTalker. "
            f"Please install external dependencies: {e}"
        ) from e

    out_dir = tmp_dir / "sadtalker_frames"
    out_dir.mkdir(exist_ok=True)

    # Call SadTalker's generate function (adjust parameters based on actual API)
    if hasattr(sad, 'generate'):
        sad.generate(
            cfg_path=str(SAD_CKPT),
            img=face_img,
            audio=audio_wav,
            out_dir=str(out_dir),
            enhancer_ckpt=str(GFPGAN_CKPT),
            still=True,
        )
    else:
        # Fallback: create dummy frames (for testing without SadTalker)
        raise RuntimeError("SadTalker module loaded but 'generate' function not found")

    return out_dir


def run_wav2lip(frames_dir: Path, audio_wav: str, tmp_dir: Path) -> Path:
    """
    Refines mouth region with Wav2Lip GAN. Returns dir with final frames.
    """
    # Try to load Wav2Lip module dynamically
    try:
        wav2lip_path = EXT_DEPS_DIR / "Wav2Lip" / "inference.py"
        w2l = load_module_from_path("wav2lip_inference", wav2lip_path)
    except (ImportError, FileNotFoundError) as e:
        raise RuntimeError(
            f"Wav2Lip not found in {EXT_DEPS_DIR}/Wav2Lip. "
            f"Please install external dependencies: {e}"
        ) from e

    out_dir = tmp_dir / "w2l_frames"
    out_dir.mkdir(exist_ok=True)

    # Call Wav2Lip's inference function (adjust parameters based on actual API)
    if hasattr(w2l, 'run_wav2lip'):
        w2l.run_wav2lip(
            ckpt=str(W2L_CKPT),
            frames=str(frames_dir),
            audio=audio_wav,
            out_dir=str(out_dir),
        )
    else:
        raise RuntimeError("Wav2Lip module loaded but 'run_wav2lip' function not found")

    return out_dir


def enhance_with_gfpgan(frames_dir: Path, tmp_dir: Path) -> Path:
    """
    Enhances face quality using GFPGAN.
    Returns new directory with enhanced frames.
    """
    try:
        import gfpgan.inference as gfp  # type: ignore

        out_dir = tmp_dir / "gfpgan_frames"
        out_dir.mkdir(exist_ok=True)
        gfp.enhance_faces(
            ckpt=str(GFPGAN_CKPT),
            frames_dir=str(frames_dir),
            out_dir=str(out_dir),
            upscale=2,
            bg_upsampler='realesrgan'
        )
        return out_dir
    except (ImportError, RuntimeError) as e:
        print(f"[pipeline] GFPGAN enhancement failed, using original frames: {e}")
        return frames_dir


def encode_mp4(frames_dir: Path, audio_wav: str, out_mp4: str, quality_mode: str = "high_quality") -> None:
    """
    Uses FFmpeg to combine PNG sequence + WAV into H.264 MP4.
    Automatically selects best encoder: NVENC (GPU) or libx264 (CPU).

    Args:
        frames_dir: Directory containing frame sequence (%04d.png)
        audio_wav: Path to audio file
        out_mp4: Output video path
        quality_mode: "real_time" or "high_quality" to adjust encoding settings
    """
    import torch

    # Detect available encoder
    cuda_available = torch.cuda.is_available()

    # Base command
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
    ]

    # Video encoding settings based on quality mode and hardware
    if quality_mode == "real_time":
        # Fast encoding for streaming
        if cuda_available:
            cmd.extend([
                "-c:v", "h264_nvenc",
                "-preset", "p1",  # Fastest NVENC preset
                "-tune", "ll",    # Low latency
                "-profile:v", "baseline",
                "-b:v", "2M",     # Lower bitrate for streaming
            ])
        else:
            cmd.extend([
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-tune", "zerolatency",
                "-profile:v", "baseline",
                "-b:v", "2M",
            ])
    else:  # high_quality
        # Best quality encoding
        if cuda_available:
            cmd.extend([
                "-c:v", "h264_nvenc",
                "-preset", "p7",  # Best quality NVENC preset
                "-profile:v", "high",
                "-b:v", "6M",     # Higher bitrate for quality
            ])
        else:
            cmd.extend([
                "-c:v", "libx264",
                "-preset", "slow",
                "-profile:v", "high",
                "-crf", "18",     # Near-lossless quality
            ])

    # Common settings
    cmd.extend([
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        out_mp4,
    ])

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
    viseme_json: Optional[str] = None,
    quality_mode: str = "auto",
) -> str:
    """
    High‑level entry used by Celery and the MCP server.

    Args:
        face_image: Path to input portrait image (PNG/JPG)
        audio: Path to speech audio file (WAV/MP3)
        out_path: Output video file path
        reference_video: Optional driving video for head pose
        viseme_json: Optional phoneme/viseme alignment data
        quality_mode: "real_time", "high_quality", or "auto" (default)

    Quality Modes:
        real_time:    Fast processing (SadTalker + Wav2Lip, lower resolution)
                      Ideal for: Live streaming, news broadcasts, real-time chatbots
                      Target: <3s latency for 512x512 @ 25fps

        high_quality: Best quality (FOMM + Diff2Lip, full resolution + GFPGAN)
                      Ideal for: YouTube content, marketing videos, high-quality avatars
                      Target: Best quality, processing time not critical

        auto:         Automatically selects based on GPU availability and model checkpoints

    Returns:
        Absolute path to the generated video file
    """
    import logging
    logger = logging.getLogger("pipeline")

    tmp = Path(tempfile.mkdtemp(prefix="vidgen_"))
    logger.info(f"[pipeline] Starting render with quality_mode={quality_mode}")
    logger.info(f"[pipeline] Input: face={face_image}, audio={audio}")

    # Validate inputs
    if not Path(face_image).exists():
        raise FileNotFoundError(f"Face image not found: {face_image}")
    if not Path(audio).exists():
        raise FileNotFoundError(f"Audio file not found: {audio}")

    try:
        # Determine which pipeline to use
        cuda_available = torch.cuda.is_available()
        fomm_available = FOMM_CKPT.exists()
        diff2lip_available = D2L_CKPT.exists()
        sadtalker_available = SAD_CKPT.exists()
        wav2lip_available = W2L_CKPT.exists()

        # Auto-select quality mode if not specified
        if quality_mode == "auto":
            if cuda_available and fomm_available and diff2lip_available:
                quality_mode = "high_quality"
                logger.info("[pipeline] Auto-selected high_quality mode (GPU + FOMM available)")
            elif sadtalker_available and wav2lip_available:
                quality_mode = "real_time"
                logger.info("[pipeline] Auto-selected real_time mode (fallback models available)")
            else:
                raise RuntimeError("No suitable models available for rendering")

        # Execute pipeline based on quality mode
        if quality_mode == "high_quality":
            if not (cuda_available and fomm_available and diff2lip_available):
                raise RuntimeError(
                    "high_quality mode requires GPU and FOMM+Diff2Lip models. "
                    "Use real_time mode or ensure models are downloaded."
                )
            logger.info("[pipeline] Using HIGH QUALITY pipeline: FOMM + Diff2Lip + GFPGAN")
            fomm_frames = run_fomm(face_image, audio, reference_video, tmp)
            lip_frames = run_diff2lip(fomm_frames, audio, tmp)

            # Apply GFPGAN enhancement for high quality
            if GFPGAN_CKPT.exists():
                logger.info("[pipeline] Applying GFPGAN face enhancement")
                final_frames = enhance_with_gfpgan(lip_frames, tmp)
            else:
                logger.warning("[pipeline] GFPGAN not available, skipping enhancement")
                final_frames = lip_frames

        elif quality_mode == "real_time":
            if not (sadtalker_available and wav2lip_available):
                raise RuntimeError(
                    "real_time mode requires SadTalker and Wav2Lip models. "
                    "Ensure models are downloaded."
                )
            logger.info("[pipeline] Using REAL-TIME pipeline: SadTalker + Wav2Lip")
            sad_frames = run_sadtalker(face_image, audio, tmp)
            final_frames = run_wav2lip(sad_frames, audio, tmp)
            # Skip enhancement for real-time to reduce latency

        else:
            raise ValueError(
                f"Invalid quality_mode: {quality_mode}. "
                "Must be 'real_time', 'high_quality', or 'auto'"
            )

        # Encode final video
        logger.info(f"[pipeline] Encoding video to {out_path}")
        encode_mp4(final_frames, audio, out_path, quality_mode=quality_mode)

        result_path = os.path.abspath(out_path)
        logger.info(f"[pipeline] Render complete: {result_path}")
        return result_path

    except Exception as e:
        logger.error(f"[pipeline] Render failed: {e}", exc_info=True)
        raise
    finally:
        # Optional: clean tmp dirs or leave for debugging
        # import shutil
        # shutil.rmtree(tmp, ignore_errors=True)
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
