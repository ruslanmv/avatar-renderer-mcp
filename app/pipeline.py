"""
pipeline.py – FOMM + Diff2Lip (+ Wav2Lip fallback) + optional GFPGAN

Input:
    face_image       – path to PNG/JPG portrait
    audio            – path to WAV (ideally 16kHz mono; but ffmpeg will decode other)
    reference_video  – optional driver MP4
    out_path         – target MP4 file

Returns:
    Absolute path to the finished MP4.
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
from typing import Optional, Any, Tuple, List

import torch

# --------------------------------------------------------------------------- #
# Torchvision monkeypatch for BasicSR/GFPGAN compatibility (safe no-op if not needed)
# --------------------------------------------------------------------------- #
try:
    from torchvision.transforms import functional_tensor  # noqa: F401
except Exception:
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
    if (cwd / "pyproject.toml").exists() or (cwd / "Makefile").exists() or (cwd / "app").exists():
        return cwd / "models"
    return Path("/models")


MODEL_ROOT = _resolve_model_root()

FOMM_CKPT = MODEL_ROOT / "fomm" / "vox-cpk.pth"

# Diff2Lip: support BOTH checkpoints (paper + legacy)
D2L_PAPER_CKPT = MODEL_ROOT / "diff2lip" / "e7.24.1.3_model260000_paper.pt"
D2L_LEGACY_CKPT = MODEL_ROOT / "diff2lip" / "Diff2Lip.pth"

W2L_CKPT = MODEL_ROOT / "wav2lip" / "wav2lip_gan.pth"
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
    """
    Dynamically load a Python module from a file path.
    Injects the parent directory to sys.path so internal imports work.
    """
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


def _run_cmd_capture_tail(cmd: List[str], env: Optional[dict] = None, tail_lines: int = 30) -> Tuple[int, str]:
    """
    Run subprocess, capture combined stdout/stderr, return (returncode, tail_text).
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    assert proc.stdout is not None
    lines: List[str] = []
    for line in proc.stdout:
        lines.append(line.rstrip("\n"))
        if len(lines) > 5000:
            lines = lines[-2500:]
    proc.wait()
    tail = "\n".join(lines[-tail_lines:])
    return proc.returncode, tail


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _audio_to_16k_mono(in_audio: str, out_wav: str) -> None:
    """
    Some pipelines behave better with 16k mono WAV.
    """
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", in_audio,
        "-ac", "1",
        "-ar", "16000",
        out_wav,
    ]
    subprocess.check_call(cmd)


def _make_silent_video_from_frames(frames_dir: Path, out_mp4: Path, fps: int = 25) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-r", str(fps),
        "-i", str(frames_dir / "%04d.png"),
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(out_mp4),
    ]
    subprocess.check_call(cmd)


def _remux_video_with_audio(video_in: Path, audio_in: Path, out_mp4: Path) -> None:
    """
    ALWAYS remux final output to guarantee audio exists.
    Keeps video stream as-is and encodes audio to AAC.
    Audio duration is preserved (not truncated to video length).
    """
    cmd = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-i", str(video_in),
        "-i", str(audio_in),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(out_mp4),
    ]
    subprocess.check_call(cmd)


def _verify_has_audio(path: Path) -> bool:
    if not _ffprobe_binary_available():
        return True
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "stream=codec_type",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            text=True,
        )
        streams = [s.strip() for s in out.splitlines() if s.strip()]
        return "audio" in streams
    except Exception:
        return True


# --------------------------------------------------------------------------- #
# ✅ FIX: Robust mpi4py mock (package) with COMM_WORLD.bcast and friends
# --------------------------------------------------------------------------- #

def _create_mpi4py_mock(mock_root: Path) -> None:
    """
    Create a minimal mpi4py package that satisfies guided-diffusion's dist_util.py.

    guided_diffusion/dist_util.py expects:
      from mpi4py import MPI
      comm = MPI.COMM_WORLD
      comm.Get_rank(), comm.Get_size(), comm.bcast(), comm.Barrier(), etc.
      AND comm.rank, comm.size as properties
    """
    pkg = mock_root / "mpi4py"
    _ensure_dir(pkg)

    init_py = pkg / "__init__.py"

    # Minimal but compatible API surface
    init_py.write_text(
        """\
# Auto-generated mpi4py mock for single-process inference
# This is only meant to satisfy guided-diffusion / Diff2Lip imports.

class _Op:
    pass

class _Comm:
    # Properties (used by dist_util.py)
    @property
    def rank(self):
        return 0
    
    @property
    def size(self):
        return 1
    
    # Methods (also part of MPI API)
    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def bcast(self, obj, root=0):
        # Single process: broadcast is identity
        return obj

    def Barrier(self):
        return None

    def allreduce(self, obj, op=None):
        # Single process: reduce is identity
        return obj

    def gather(self, obj, root=0):
        return [obj]

    def scatter(self, obj, root=0):
        if isinstance(obj, (list, tuple)) and obj:
            return obj[0]
        return obj

    def Abort(self, code=1):
        raise SystemExit(code)

class _MPI:
    COMM_WORLD = _Comm()
    SUM = _Op()
    MAX = _Op()
    MIN = _Op()

MPI = _MPI()
""",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Stage 1: FOMM
# --------------------------------------------------------------------------- #

def run_fomm(face_img: str, audio_wav: str, ref_video: Optional[str], tmp_dir: Path) -> Path:
    """
    Run First Order Motion Model to animate the face.
    Produces frames %04d.png in frames_dir.
    """
    fomm_wrapper = EXT_DEPS_DIR / "first-order-model" / "fomm_wrapper.py"
    fomm_demo_py = EXT_DEPS_DIR / "first-order-model" / "demo.py"
    fomm_path = fomm_wrapper if fomm_wrapper.exists() else fomm_demo_py
    if not fomm_path.exists():
        raise RuntimeError(f"FOMM not found at {EXT_DEPS_DIR / 'first-order-model'}")

    fomm_demo = load_module_from_path("fomm_demo", fomm_path)

    frames_dir = tmp_dir / "fomm_frames"
    _ensure_dir(frames_dir)

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
        raise RuntimeError("FOMM module loaded but 'main' function not found")

    fomm_demo.main(_Args())
    return frames_dir


# --------------------------------------------------------------------------- #
# Stage 2: Diff2Lip (subprocess) with BOTH checkpoint styles
# --------------------------------------------------------------------------- #

def _diff2lip_script() -> Path:
    return EXT_DEPS_DIR / "Diff2Lip" / "generate.py"


def _pick_diff2lip_ckpts() -> List[Path]:
    ckpts: List[Path] = []
    if D2L_PAPER_CKPT.exists():
        ckpts.append(D2L_PAPER_CKPT)
    if D2L_LEGACY_CKPT.exists():
        ckpts.append(D2L_LEGACY_CKPT)
    return ckpts


def _diff2lip_flags_for_checkpoint(ckpt: Path) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    """
    Best-effort flags matching the "paper" model.
    Your fingerprint shows both models look identical (same tensor shapes),
    so we can use the same flags for both.
    """
    model_flags = [
        "--attention_resolutions", "32,16,8",
        "--class_cond", "False",
        "--learn_sigma", "True",
        "--num_channels", "128",
        "--num_head_channels", "64",
        "--num_res_blocks", "2",
        "--resblock_updown", "True",
        "--use_fp16", "True",
        "--use_scale_shift_norm", "False",
    ]
    diffusion_flags = [
        "--predict_xstart", "False",
        "--diffusion_steps", "1000",
        "--noise_schedule", "linear",
        "--rescale_timesteps", "False",
    ]
    sample_flags = [
        "--sampling_seed", "7",
        "--sampling_input_type", "gt",
        "--sampling_ref_type", "gt",
        "--timestep_respacing", "ddim25",
        "--use_ddim", "True",
        f"--model_path={str(ckpt)}",
    ]
    data_flags = [
        "--nframes", "5",
        "--nrefer", "1",
        "--image_size", "128",
        "--sampling_batch_size", "16",
    ]
    tfg_flags = [
        "--face_hide_percentage", "0.5",
        "--use_ref", "True",
        "--use_audio", "True",
        "--audio_as_style", "True",
    ]
    return model_flags, diffusion_flags, sample_flags, data_flags, tfg_flags


def run_diff2lip(frames_dir: Path, audio_in: str, tmp_dir: Path) -> Path:
    """
    Run Diff2Lip generate.py as subprocess.
    Returns: output_video_path
    
    Note: Diff2Lip internally uses 16kHz mono audio for inference,
    but we don't return that audio - the caller should use the original
    high-quality audio for final output.
    """
    gen = _diff2lip_script()
    if not gen.exists():
        raise RuntimeError(f"Diff2Lip generate.py not found at: {gen}")

    if not _ffmpeg_binary_available():
        raise RuntimeError("ffmpeg not found; Diff2Lip requires ffmpeg")

    # Convert frames -> mp4
    d2l_input_mp4 = tmp_dir / "d2l_input.mp4"
    _make_silent_video_from_frames(frames_dir, d2l_input_mp4, fps=25)

    # Convert audio to 16k mono for Diff2Lip inference
    # This is only used internally by Diff2Lip for lip-sync
    audio_16k = tmp_dir / "audio_16k_mono.wav"
    _audio_to_16k_mono(audio_in, str(audio_16k))

    # ✅ Inject mpi4py mock BEFORE Diff2Lip imports happen
    mock_root = tmp_dir / "mock_libs"
    _ensure_dir(mock_root)
    _create_mpi4py_mock(mock_root)

    # Ensure Diff2Lip can import its guided_diffusion (the one inside Diff2Lip repo)
    env = os.environ.copy()
    d2l_root = (EXT_DEPS_DIR / "Diff2Lip").resolve()
    guided_root = (EXT_DEPS_DIR / "Diff2Lip" / "guided-diffusion").resolve()

    env["PYTHONPATH"] = (
        f"{str(mock_root.resolve())}{os.pathsep}"
        f"{str(d2l_root)}{os.pathsep}"
        f"{str(guided_root)}{os.pathsep}"
        f"{env.get('PYTHONPATH', '')}"
    )

    # (Optional) avoid some MPI init assumptions
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env.setdefault("MASTER_PORT", "29500")

    out_dir = tmp_dir / "d2l_out"
    _ensure_dir(out_dir)
    out_mp4 = out_dir / "result.mp4"

    ckpts = _pick_diff2lip_ckpts()
    if not ckpts:
        raise RuntimeError("No Diff2Lip checkpoints found. Put one in models/diff2lip/")

    last_tail = ""
    for ckpt in ckpts:
        model_flags, diffusion_flags, sample_flags, data_flags, tfg_flags = _diff2lip_flags_for_checkpoint(ckpt)

        gen_flags = [
            "--generate_from_filelist", "0",
            "--video_path", str(d2l_input_mp4),
            "--audio_path", str(audio_16k),
            "--out_path", str(out_mp4),
            "--sample_path", str(out_dir),
            "--save_orig", "False",
            "--face_det_batch_size", "16",
            "--pads", "0,0,0,0",
            "--is_voxceleb2", "False",
        ]

        all_args = model_flags + diffusion_flags + sample_flags + data_flags + tfg_flags + gen_flags
        
        # Simple subprocess call - PyAV should be installed
        cmd = [sys.executable, str(gen)] + all_args
        
        print(f"[pipeline] Diff2Lip trying ckpt: {ckpt.name}")
        print(f"[pipeline] Diff2Lip running: {' '.join(cmd)}")

        if out_mp4.exists():
            out_mp4.unlink()

        code, tail = _run_cmd_capture_tail(cmd, env=env, tail_lines=60)
        if code == 0 and out_mp4.exists() and out_mp4.stat().st_size > 0:
            # ✅ Return just the video - caller uses original audio
            print(f"[pipeline] Diff2Lip succeeded with {ckpt.name}")
            return out_mp4

        last_tail = tail
        print("[pipeline] Diff2Lip failed output (tail):")
        print(last_tail)

    raise RuntimeError(f"Diff2Lip failed for all checkpoints. Last tail:\n{last_tail}")


# --------------------------------------------------------------------------- #
# Stage 3: Wav2Lip fallback
# --------------------------------------------------------------------------- #

def run_wav2lip(frames_dir: Path, audio_wav: str, tmp_dir: Path) -> Path:
    wav2lip_dir = EXT_DEPS_DIR / "Wav2Lip"
    script = wav2lip_dir / "inference.py"
    if not script.exists():
        print("[pipeline] Wav2Lip missing; returning frames without lip-sync.")
        return frames_dir

    if not _ffmpeg_binary_available():
        print("[pipeline] ffmpeg not available; returning frames without lip-sync.")
        return frames_dir

    silent_vid = tmp_dir / "wav2lip_input.mp4"
    out_vid = tmp_dir / "wav2lip_out.mp4"

    _make_silent_video_from_frames(frames_dir, silent_vid, fps=25)

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
        print("[pipeline] Wav2Lip finished but output missing/empty; returning frames.")
        return frames_dir
    except Exception as e:
        print(f"[pipeline] Wav2Lip failed: {e}")
        return frames_dir


# --------------------------------------------------------------------------- #
# Stage 4: GFPGAN enhancement (frames only)
# --------------------------------------------------------------------------- #

def enhance_with_gfpgan(frames_or_video: Any, tmp_dir: Path) -> Path:
    try:
        from gfpgan import GFPGANer
    except Exception:
        print("[pipeline] GFPGAN not installed; skipping.")
        return Path(frames_or_video)

    import cv2

    out_dir = tmp_dir / "gfpgan_frames"
    _ensure_dir(out_dir)

    input_path = Path(frames_or_video)

    if input_path.is_file():
        extracted = tmp_dir / "video_extract_frames"
        _ensure_dir(extracted)
        subprocess.check_call([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(input_path),
            str(extracted / "%04d.png"),
        ])
        input_path = extracted

    if not input_path.is_dir():
        return input_path

    try:
        restorer = GFPGANer(
            model_path=str(GFPGAN_CKPT),
            upscale=2,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )
    except Exception as e:
        print(f"[pipeline] Failed to init GFPGAN: {e}")
        return input_path

    pngs = sorted(glob.glob(str(input_path / "*.png")))
    print(f"[pipeline] Enhancing {len(pngs)} frames with GFPGAN...")

    for p in pngs:
        basename = os.path.basename(p)
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        _, _, out_img = restorer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        cv2.imwrite(str(out_dir / basename), out_img)

    return out_dir


# --------------------------------------------------------------------------- #
# Final encode (frames + audio)
# --------------------------------------------------------------------------- #

def encode_mp4(frames_dir: Path, audio_wav: str, out_mp4: str, fps: int = 25) -> None:
    """
    Encode frames to MP4 with audio.
    Audio duration is preserved (video extends last frame if needed).
    """
    if not _ffmpeg_binary_available():
        raise RuntimeError("ffmpeg not found on PATH")

    frame_files = sorted(glob.glob(str(frames_dir / "*.png")))
    if not frame_files:
        raise RuntimeError(f"No PNG frames found in {frames_dir}")

    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "warning",
        "-thread_queue_size", "1024",
        "-r", str(fps),
        "-i", f"{frames_dir}/%04d.png",
        "-i", audio_wav,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(out_mp4),
    ]
    subprocess.check_call(cmd)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def render_pipeline(
    *,
    face_image: str,
    audio: str,
    out_path: str,
    reference_video: Optional[str] = None,
    viseme_json: Optional[str] = None,
    quality_mode: str = "auto",
    force_wav2lip: bool = False,
) -> str:
    """
    FOMM -> (Diff2Lip or Wav2Lip) -> optional GFPGAN -> final mp4 with guaranteed audio

    Args:
        face_image: Path to portrait image
        audio: Path to audio file
        out_path: Output video path
        reference_video: Optional reference video for motion
        viseme_json: Optional viseme data (currently unused)
        quality_mode: Quality mode (auto, real_time, high_quality)
        force_wav2lip: If True, use Wav2Lip instead of Diff2Lip regardless of quality_mode

    Audio handling:
    - Diff2Lip uses 16kHz mono internally for lip-sync inference
    - Final output uses the ORIGINAL high-quality audio for best quality
    """
    tmp = Path(tempfile.mkdtemp(prefix="vidgen_"))

    if not Path(face_image).exists():
        raise FileNotFoundError(f"Face image not found: {face_image}")
    if not Path(audio).exists():
        raise FileNotFoundError(f"Audio file not found: {audio}")

    fomm_frames = run_fomm(face_image, audio, reference_video, tmp)

    # ✅ ALWAYS use original high-quality audio for final output
    final_audio = audio

    lip_result: Path

    # Decide which lip-sync method to use
    if force_wav2lip:
        print("[pipeline] Wav2Lip forced via --wav2lip flag")
        lip_result = run_wav2lip(fomm_frames, audio, tmp)
    else:
        want_d2l = (quality_mode in ("high_quality", "auto")) and torch.cuda.is_available()
        d2l_available = _diff2lip_script().exists() and (D2L_PAPER_CKPT.exists() or D2L_LEGACY_CKPT.exists())

        if want_d2l and d2l_available:
            try:
                # Diff2Lip uses 16kHz mono internally, but we don't use that for final output
                lip_result = run_diff2lip(fomm_frames, audio, tmp)
                # ✅ DON'T overwrite final_audio - keep using original
                print(f"[pipeline] Using original audio for final output (not 16kHz version)")
            except Exception as e:
                print(f"[pipeline] Diff2Lip failed: {e}. Falling back to Wav2Lip.")
                lip_result = run_wav2lip(fomm_frames, audio, tmp)
        else:
            lip_result = run_wav2lip(fomm_frames, audio, tmp)

    final_result: Path = lip_result
    if GFPGAN_CKPT.exists():
        final_result = enhance_with_gfpgan(lip_result, tmp)

    out_path_abs = str(Path(out_path).resolve())
    out_path_p = Path(out_path_abs)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)

    if final_result.is_file() and final_result.suffix.lower() == ".mp4":
        # Video file - remux with original high-quality audio
        tmp_remux = tmp / "remux.mp4"
        print(f"[pipeline] Remuxing video with original audio: {final_audio}")
        _remux_video_with_audio(final_result, Path(final_audio), tmp_remux)
        shutil.copy(str(tmp_remux), out_path_abs)
    else:
        # Directory of frames - encode with original high-quality audio
        print(f"[pipeline] Encoding frames with original audio: {final_audio}")
        encode_mp4(final_result, final_audio, str(out_path_abs), fps=25)

    if not _verify_has_audio(Path(out_path_abs)):
        print("[pipeline] ⚠️ WARNING: final output seems to have no audio stream after remux.")
    else:
        print("[pipeline] ✅ Audio stream verified in final output")

    return out_path_abs