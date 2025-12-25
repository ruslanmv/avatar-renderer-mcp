"""
app/pipeline.py – FOMM + Diff2Lip (+ GFPGAN + Wav2Lip fallback)

Input:
    face_image       – path to PNG/JPG portrait (one face, frontish)
    audio            – path to WAV (we normalize to 16kHz mono)
    reference_video  – optional short driver MP4 (for motion)
    out_path         – target MP4 file

Returns:
    Absolute path to the finished MP4 (with audio).
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import glob
from pathlib import Path
from typing import Optional, Any

import cv2
import torch

# --------------------------------------------------------------------------- #
# HOTFIX: Monkeypatch torchvision for GFPGAN/BasicSR compatibility
# --------------------------------------------------------------------------- #
try:
    from torchvision.transforms import functional_tensor  # noqa: F401
except ImportError:
    try:
        import torchvision.transforms.functional as functional  # noqa: F401
        sys.modules["torchvision.transforms.functional_tensor"] = functional
    except Exception:
        pass

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

def _resolve_model_root() -> Path:
    env = os.environ.get("MODEL_ROOT")
    if env:
        return Path(env)
    cwd = Path.cwd()
    if (cwd / "pyproject.toml").exists() or (cwd / "Makefile").exists():
        return cwd / "models"
    return Path("/models")


MODEL_ROOT = _resolve_model_root()

FOMM_CKPT   = MODEL_ROOT / "fomm" / "vox-cpk.pth"
D2L_CKPT    = MODEL_ROOT / "diff2lip" / "Diff2Lip.pth"
W2L_CKPT    = MODEL_ROOT / "wav2lip" / "wav2lip_gan.pth"
GFPGAN_CKPT = MODEL_ROOT / "gfpgan" / "GFPGANv1.3.pth"

EXT_DEPS_DIR = Path(os.environ.get("EXT_DEPS_DIR", "external_deps"))

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _ffmpeg_binary_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _ffprobe_binary_available() -> bool:
    return shutil.which("ffprobe") is not None


def load_module_from_path(module_name: str, file_path: Path) -> Any:
    """Dynamically load a Python module from a file path."""
    if not file_path.exists():
        raise ImportError(f"Module file not found: {file_path}")

    module_dir = str(file_path.parent.resolve())
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec: {file_path}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ImportError(f"Failed to execute module {file_path}: {e}") from e

    return module


def _create_mpi_mock(target_dir: Path) -> Path:
    """
    Creates a fake mpi4py module so Diff2Lip doesn't crash on import.
    (Some forks import mpi4py even for single-GPU inference.)
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    mpi_file = target_dir / "mpi4py.py"
    if not mpi_file.exists():
        mpi_file.write_text(
            """
class MPI_Mock:
    def __getattr__(self, name):
        return self
    def Get_rank(self): return 0
    def Get_size(self): return 1
    def Barrier(self): pass

MPI = MPI_Mock()
COMM_WORLD = MPI_Mock()
""".lstrip()
        )
    return target_dir


def _ensure_audio_is_decodable(audio_path: str, tmp_dir: Path) -> str:
    """
    Normalize audio for downstream tools:
      - WAV
      - mono
      - 16kHz
    """
    in_path = Path(audio_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    if not _ffmpeg_binary_available():
        return str(in_path)

    fixed = tmp_dir / "audio_16k_mono.wav"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(in_path),
        "-ac", "1",
        "-ar", "16000",
        str(fixed),
    ]
    subprocess.check_call(cmd)
    return str(fixed)


def _video_has_audio(video_path: Path) -> bool:
    if not _ffprobe_binary_available():
        return False
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=codec_type",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            text=True,
        ).strip()
        return "audio" in out.splitlines()
    except Exception:
        return False


def _mux_audio_into_video(video_path: Path, audio_wav: str, out_path: Path) -> Path:
    """Force mux audio into an existing MP4 (used when an output is silent)."""
    if not _ffmpeg_binary_available():
        raise RuntimeError("ffmpeg not found")

    cmd = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-i", str(video_path),
        "-i", str(audio_wav),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        str(out_path),
    ]
    subprocess.check_call(cmd)
    return out_path


def _make_silent_video(frames_dir: Path, out_mp4: Path, fps: int = 25):
    """Create a silent MP4 from an image sequence (for tools that require video input)."""
    if not _ffmpeg_binary_available():
        raise RuntimeError("ffmpeg not found")

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-r", str(fps),
        "-i", str(frames_dir / "%04d.png"),
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(out_mp4),
    ]
    subprocess.check_call(cmd)

# --------------------------------------------------------------------------- #
# Pipeline stages
# --------------------------------------------------------------------------- #

def run_fomm(face_img: str, audio_wav: str, ref_video: Optional[str], tmp_dir: Path) -> Path:
    """Run First Order Motion Model to animate the face -> returns frames directory."""
    fomm_path = EXT_DEPS_DIR / "first-order-model" / "fomm_wrapper.py"
    if not fomm_path.exists():
        fomm_path = EXT_DEPS_DIR / "first-order-model" / "demo.py"
    if not fomm_path.exists():
        raise RuntimeError(f"FOMM entry not found under {EXT_DEPS_DIR / 'first-order-model'}")

    fomm_demo = load_module_from_path("fomm_demo", fomm_path)

    frames_dir = tmp_dir / "fomm_frames"
    frames_dir.mkdir(exist_ok=True)

    class _Args:
        source_image = face_img
        driving_audio = audio_wav
        driving_video = ref_video
        result_dir = str(frames_dir)
        checkpoint = str(FOMM_CKPT)
        cpu = not torch.cuda.is_available()
        relative = True
        adapt_scale = True

    if not hasattr(fomm_demo, "main"):
        raise RuntimeError("FOMM 'main' function not found")

    fomm_demo.main(_Args())
    return frames_dir


def run_wav2lip(frames_dir: Path, audio_wav: str, tmp_dir: Path) -> Path:
    """Run Wav2Lip fallback. Returns a video path if successful, else frames_dir."""
    wav2lip_dir = EXT_DEPS_DIR / "Wav2Lip"
    script = wav2lip_dir / "inference.py"
    if not script.exists():
        print("[pipeline] Wav2Lip missing.")
        return frames_dir

    silent_vid = tmp_dir / "wav2lip_input.mp4"
    out_vid = tmp_dir / "wav2lip_out.mp4"

    _make_silent_video(frames_dir, silent_vid)

    # Some forks hardcode temp/result.avi
    os.makedirs("temp", exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{str(wav2lip_dir.resolve())}{os.pathsep}{env.get('PYTHONPATH', '')}"

    cmd = [
        sys.executable, str(script),
        "--checkpoint_path", str(W2L_CKPT),
        "--face", str(silent_vid),
        "--audio", str(audio_wav),
        "--outfile", str(out_vid),
        "--resize_factor", "1",
    ]

    try:
        subprocess.check_call(cmd, env=env)
        if out_vid.exists() and out_vid.stat().st_size > 0:
            return out_vid
    except Exception as e:
        print(f"[pipeline] Wav2Lip failed: {e}")

    return frames_dir


def run_diff2lip(frames_dir: Path, audio_wav: str, tmp_dir: Path) -> Path:
    """
    Run Diff2Lip (HQ) correctly for the zip version you posted.

    FIXES:
      - Do NOT use the broken shim CLI (--face/--audio/--outfile) because generate.py
        from the zip expects --video_path/--audio_path/--out_path/--model_path, etc.
      - Force the correct guided-diffusion (Diff2Lip's bundled guided-diffusion),
        otherwise Python may import external_deps/guided-diffusion which lacks tfg_*.
      - Use cwd=Diff2Lip root so relative paths like configs/... resolve safely.
      - If output mp4 is silent, mux audio in.
    """
    d2l_root = EXT_DEPS_DIR / "Diff2Lip"
    generate_py = d2l_root / "generate.py"
    if not generate_py.exists():
        print("[pipeline] Diff2Lip generate.py missing. Falling back to Wav2Lip.")
        return run_wav2lip(frames_dir, audio_wav, tmp_dir)

    # Diff2Lip wants a VIDEO input: convert frames -> silent mp4
    d2l_in_video = tmp_dir / "d2l_input.mp4"
    _make_silent_video(frames_dir, d2l_in_video)

    out_dir = tmp_dir / "d2l_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / "result.mp4"

    # Build environment to guarantee correct imports
    env = os.environ.copy()
    current_pp = env.get("PYTHONPATH", "")

    mock_dir = _create_mpi_mock(tmp_dir / "mock_libs")

    # IMPORTANT: prefer Diff2Lip bundled guided-diffusion
    bundled_gd = d2l_root / "guided-diffusion"
    python_path_list = [str(mock_dir.resolve())]

    if bundled_gd.exists():
        python_path_list.append(str(bundled_gd.resolve()))
    python_path_list.append(str(d2l_root.resolve()))

    env["PYTHONPATH"] = f"{os.pathsep.join(python_path_list)}{os.pathsep}{current_pp}"

    # generate.py arguments (zip version)
    cmd = [
        sys.executable, str(generate_py),
        "--model_path", str(D2L_CKPT),
        "--video_path", str(d2l_in_video),
        "--audio_path", str(audio_wav),
        "--out_path", str(out_mp4),
        "--sample_path", str(out_dir),
        "--save_orig", "False",
    ]

    print(f"[pipeline] Diff2Lip running: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd, env=env, cwd=str(d2l_root))
    except Exception as e:
        print(f"[pipeline] Diff2Lip failed: {e}. Falling back to Wav2Lip.")
        return run_wav2lip(frames_dir, audio_wav, tmp_dir)

    if not out_mp4.exists() or out_mp4.stat().st_size == 0:
        print("[pipeline] Diff2Lip finished but result.mp4 missing/empty. Falling back to Wav2Lip.")
        return run_wav2lip(frames_dir, audio_wav, tmp_dir)

    # Some environments produce silent mp4 with torchvision; enforce mux if needed
    if not _video_has_audio(out_mp4):
        print("[pipeline] Diff2Lip result.mp4 has NO audio. Muxing audio in...")
        muxed = out_dir / "result_with_audio.mp4"
        _mux_audio_into_video(out_mp4, audio_wav, muxed)
        return muxed

    return out_mp4


def enhance_with_gfpgan(frames_or_vid: Any, tmp_dir: Path) -> Path:
    """
    Enhance face quality using GFPGANer.
    Returns a FRAMES directory (PNG sequence). If input is a video, it extracts frames first.
    """
    try:
        from gfpgan import GFPGANer
    except ImportError:
        print("[pipeline] GFPGAN not installed.")
        return Path(frames_or_vid)

    input_path = Path(frames_or_vid)

    # If input is a video, extract frames (audio will be re-added later in encode_mp4)
    if input_path.is_file():
        video_frames_dir = tmp_dir / "gfpgan_in_frames"
        video_frames_dir.mkdir(exist_ok=True)
        subprocess.check_call([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(input_path),
            str(video_frames_dir / "%04d.png"),
        ])
        input_path = video_frames_dir

    if not input_path.is_dir():
        return input_path

    img_list = sorted(glob.glob(str(input_path / "*.png")))
    if not img_list:
        return input_path

    out_dir = tmp_dir / "gfpgan_frames"
    out_dir.mkdir(exist_ok=True)

    restorer = GFPGANer(
        model_path=str(GFPGAN_CKPT),
        upscale=2,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )

    print(f"[pipeline] Enhancing {len(img_list)} frames with GFPGAN...")
    for img_path in img_list:
        basename = os.path.basename(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        _, _, output = restorer.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
        )
        cv2.imwrite(str(out_dir / basename), output)

    return out_dir


def encode_mp4(frames_dir: Path, audio_wav: str, out_mp4: str, fps: int = 25):
    """
    Encode PNG frames + audio into MP4 (production-safe):
      - explicit stream mapping
      - standard AAC audio rate
      - shortest
    """
    if not _ffmpeg_binary_available():
        raise RuntimeError("ffmpeg not found")

    frames_dir = Path(frames_dir)
    if not frames_dir.is_dir():
        raise RuntimeError(f"encode_mp4 expected a frames directory, got: {frames_dir}")

    frame_files = sorted(glob.glob(str(frames_dir / "*.png")))
    if not frame_files:
        raise RuntimeError(f"No PNG frames found in {frames_dir}")

    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "warning",
        "-thread_queue_size", "512",
        "-r", str(fps),
        "-i", str(frames_dir / "%04d.png"),
        "-thread_queue_size", "512",
        "-i", str(audio_wav),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-ar", "44100",
        "-shortest",
        str(out_mp4),
    ]
    subprocess.check_call(cmd)

    # Quick verification (non-fatal)
    if _ffprobe_binary_available():
        try:
            streams = subprocess.check_output(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "stream=codec_type",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(out_mp4),
                ],
                text=True,
            ).strip().splitlines()
            if "audio" not in streams:
                print("[pipeline] ⚠️  Output has no audio stream (silent).")
        except Exception as e:
            print(f"[pipeline] ⚠️  ffprobe verification failed: {e}")

# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def render_pipeline(
    *,
    face_image: str,
    audio: str,
    out_path: str,
    reference_video: Optional[str] = None,
    viseme_json: Optional[str] = None,  # kept for API compatibility
    quality_mode: str = "auto",
) -> str:
    """
    Main pipeline: FOMM → Lip Sync (Diff2Lip/Wav2Lip) → GFPGAN(optional) → Final MP4

    Returns absolute path to finished MP4 with audio.
    """
    import logging
    logger = logging.getLogger("pipeline")

    tmp = Path(tempfile.mkdtemp(prefix="vidgen_"))

    face_image_p = Path(face_image)
    if not face_image_p.exists():
        raise FileNotFoundError(f"Face image not found: {face_image}")

    audio_fixed = _ensure_audio_is_decodable(audio, tmp)
    out_path_abs = str(Path(out_path).resolve())
    logger.info(f"Render: {face_image} + {audio_fixed} -> {out_path_abs}")

    # 1) FOMM frames
    fomm_frames = run_fomm(str(face_image_p), audio_fixed, reference_video, tmp)

    # 2) Lip sync selection
    has_d2l = (EXT_DEPS_DIR / "Diff2Lip" / "generate.py").exists()
    use_d2l = (
        quality_mode == "high_quality"
        or (quality_mode == "auto" and has_d2l and torch.cuda.is_available())
    )

    if use_d2l:
        logger.info("Using Diff2Lip")
        lip_result = run_diff2lip(fomm_frames, audio_fixed, tmp)
    else:
        logger.info("Using Wav2Lip")
        lip_result = run_wav2lip(fomm_frames, audio_fixed, tmp)

    # 3) GFPGAN (optional) -> always produces FRAMES if it runs
    final_result: Path | str = lip_result
    if GFPGAN_CKPT.exists():
        logger.info("Enhancing with GFPGAN")
        final_result = enhance_with_gfpgan(lip_result, tmp)

    # 4) Finalize output
    final_result_p = Path(final_result)

    if final_result_p.is_file():
        # If we ended on a video, ensure it has audio, else mux.
        if not _video_has_audio(final_result_p):
            logger.warning("Final video is missing audio. Muxing audio in...")
            muxed = tmp / "final_muxed.mp4"
            _mux_audio_into_video(final_result_p, audio_fixed, muxed)
            shutil.copy(str(muxed), out_path_abs)
        else:
            shutil.copy(str(final_result_p), out_path_abs)
    else:
        # Frames directory -> encode with audio
        encode_mp4(final_result_p, audio_fixed, out_path_abs)

    if not Path(out_path_abs).exists():
        raise RuntimeError(f"❌ Output file not created: {out_path_abs}")

    size_mb = Path(out_path_abs).stat().st_size / (1024 * 1024)
    logger.info(f"✅ Pipeline complete: {out_path_abs} ({size_mb:.2f} MB)")
    return out_path_abs


# --------------------------------------------------------------------------- #
# Runtime checks
# --------------------------------------------------------------------------- #

def _resolve_deps_check():
    if not (EXT_DEPS_DIR / "first-order-model").exists():
        print("WARNING: 'first-order-model' not found in external_deps.")
    if not _ffmpeg_binary_available():
        print("WARNING: ffmpeg not found in PATH. Encoding/muxing will fail.")
    if not (EXT_DEPS_DIR / "Diff2Lip" / "generate.py").exists():
        print("WARNING: Diff2Lip generate.py not found (will fallback to Wav2Lip).")

_resolve_deps_check()
