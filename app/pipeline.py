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
        "-shortest",
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
# ✅ NEW: Create a patcher script that patches torchvision.io at runtime
# --------------------------------------------------------------------------- #

def _create_torchvision_patcher(mock_root: Path) -> Path:
    """
    Create a patcher module that patches torchvision.io.write_video at runtime.
    This gets imported before generate.py runs.
    """
    patcher_file = mock_root / "patch_torchvision.py"
    
    patcher_file.write_text(
        """\
\"\"\"
Runtime patcher for torchvision.io.write_video to avoid PyAV dependency.
This must be imported before torchvision.io is used.
\"\"\"

import os
import subprocess
import tempfile
import shutil
from pathlib import Path

def patched_write_video(
    filename,
    video_array,
    fps,
    video_codec='libx264',
    options=None,
    audio_array=None,
    audio_fps=None,
    audio_codec='aac',
    audio_options=None
):
    \"\"\"
    Write video using ffmpeg instead of PyAV.
    
    Args:
        filename: Output video path
        video_array: torch.Tensor of shape (T, H, W, C) with values 0-255
        fps: Video framerate
        video_codec: Video codec (default: libx264)
        options: Video encoding options (ignored)
        audio_array: torch.Tensor of shape (C, N) audio samples
        audio_fps: Audio sample rate
        audio_codec: Audio codec (default: aac)
        audio_options: Audio encoding options (ignored)
    \"\"\"
    import numpy as np
    import torch
    
    # Convert tensors to numpy if needed
    if isinstance(video_array, torch.Tensor):
        video_array = video_array.cpu().numpy()
    if audio_array is not None and isinstance(audio_array, torch.Tensor):
        audio_array = audio_array.cpu().numpy()
    
    # Ensure uint8
    if video_array.dtype != np.uint8:
        video_array = video_array.astype(np.uint8)
    
    filename = str(filename)
    temp_dir = tempfile.mkdtemp(prefix="tv_write_")
    
    try:
        # Step 1: Write frames to temp directory
        frames_dir = Path(temp_dir) / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        for i, frame in enumerate(video_array):
            # frame is (H, W, C)
            try:
                from PIL import Image
                img = Image.fromarray(frame)
                img.save(frames_dir / f"{i:05d}.png")
            except ImportError:
                # Fallback to cv2 if PIL not available
                import cv2
                cv2.imwrite(str(frames_dir / f"{i:05d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        # Step 2: Create video with ffmpeg
        if audio_array is not None and audio_fps is not None:
            # Write audio to temp WAV file
            audio_path = Path(temp_dir) / "audio.wav"
            
            # audio_array is (C, N), we need (N,) or (N, C) for wavfile
            if audio_array.ndim == 2:
                audio_data = audio_array.T
            else:
                audio_data = audio_array
            
            # Ensure proper format for WAV
            if audio_data.dtype in (np.float32, np.float64):
                audio_data = (audio_data * 32767).astype(np.int16)
            
            try:
                import scipy.io.wavfile as wavfile
                wavfile.write(str(audio_path), int(audio_fps), audio_data)
            except ImportError:
                # Fallback: use ffmpeg to create audio from raw PCM
                print("[patch] scipy not available, using ffmpeg for audio encoding")
                raw_audio = Path(temp_dir) / "audio.raw"
                audio_data.tofile(str(raw_audio))
                subprocess.check_call([
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "s16le",
                    "-ar", str(int(audio_fps)),
                    "-ac", str(audio_array.shape[0] if audio_array.ndim == 2 else 1),
                    "-i", str(raw_audio),
                    str(audio_path)
                ])
            
            # Combine video and audio
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-r", str(fps),
                "-i", str(frames_dir / "%05d.png"),
                "-i", str(audio_path),
                "-c:v", video_codec,
                "-pix_fmt", "yuv420p",
                "-c:a", audio_codec,
                "-b:a", "192k",
                "-shortest",
                filename
            ]
        else:
            # Video only
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-r", str(fps),
                "-i", str(frames_dir / "%05d.png"),
                "-c:v", video_codec,
                "-pix_fmt", "yuv420p",
                filename
            ]
        
        subprocess.check_call(cmd)
        
    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


# Apply the patch
import torchvision.io
torchvision.io.write_video = patched_write_video
print("[patch] Successfully patched torchvision.io.write_video to use ffmpeg instead of PyAV")
""",
        encoding="utf-8",
    )
    
    return patcher_file


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
    """
    gen = _diff2lip_script()
    if not gen.exists():
        raise RuntimeError(f"Diff2Lip generate.py not found at: {gen}")

    if not _ffmpeg_binary_available():
        raise RuntimeError("ffmpeg not found; Diff2Lip requires ffmpeg")

    # Convert frames -> mp4
    d2l_input_mp4 = tmp_dir / "d2l_input.mp4"
    _make_silent_video_from_frames(frames_dir, d2l_input_mp4, fps=25)

    # Convert audio to 16k mono
    audio_16k = tmp_dir / "audio_16k_mono.wav"
    _audio_to_16k_mono(audio_in, str(audio_16k))

    # ✅ Inject mpi4py mock AND torchvision patcher BEFORE Diff2Lip imports happen
    mock_root = tmp_dir / "mock_libs"
    _ensure_dir(mock_root)
    _create_mpi4py_mock(mock_root)
    patcher_file = _create_torchvision_patcher(mock_root)

    # Ensure Diff2Lip can import its guided_diffusion (the one inside Diff2Lip repo)
    env = os.environ.copy()
    d2l_root = (EXT_DEPS_DIR / "Diff2Lip").resolve()
    guided_root = (EXT_DEPS_DIR / "Diff2Lip" / "guided-diffusion").resolve()

    # ✅ CRITICAL: Put mock_root FIRST in PYTHONPATH so our mocks take precedence
    env["PYTHONPATH"] = (
        f"{str(mock_root.resolve())}{os.pathsep}"
        f"{str(d2l_root)}{os.pathsep}"
        f"{str(guided_root)}{os.pathsep}"
        f"{env.get('PYTHONPATH', '')}"
    )

    # ✅ Use PYTHONSTARTUP or -c to import patcher before running generate.py
    # We'll use python -c to import the patcher then exec the script
    startup_code = f"import sys; sys.path.insert(0, '{str(mock_root.resolve())}'); import patch_torchvision"

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

        # ✅ Run python with -c to import patcher, then exec the generate.py script
        all_args = model_flags + diffusion_flags + sample_flags + data_flags + tfg_flags + gen_flags
        args_str = " ".join(f'"{arg}"' if " " in arg else arg for arg in all_args)
        
        exec_code = f"{startup_code}; import sys; sys.argv = ['{str(gen)}'] + {repr(model_flags + diffusion_flags + sample_flags + data_flags + tfg_flags + gen_flags)}; exec(open('{str(gen)}').read())"
        
        cmd = [sys.executable, "-c", exec_code]
        
        print(f"[pipeline] Diff2Lip trying ckpt: {ckpt.name}")
        print(f"[pipeline] Diff2Lip running with torchvision.io patch")

        if out_mp4.exists():
            out_mp4.unlink()

        code, tail = _run_cmd_capture_tail(cmd, env=env, tail_lines=60)
        if code == 0 and out_mp4.exists() and out_mp4.stat().st_size > 0:
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
        "-shortest",
        "-movflags", "+faststart",
        out_mp4,
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
) -> str:
    """
    FOMM -> (Diff2Lip or Wav2Lip) -> optional GFPGAN -> final mp4 with guaranteed audio
    """
    tmp = Path(tempfile.mkdtemp(prefix="vidgen_"))

    if not Path(face_image).exists():
        raise FileNotFoundError(f"Face image not found: {face_image}")
    if not Path(audio).exists():
        raise FileNotFoundError(f"Audio file not found: {audio}")

    fomm_frames = run_fomm(face_image, audio, reference_video, tmp)

    lip_result: Path
    want_d2l = (quality_mode in ("high_quality", "auto")) and torch.cuda.is_available()
    d2l_available = _diff2lip_script().exists() and (D2L_PAPER_CKPT.exists() or D2L_LEGACY_CKPT.exists())

    if want_d2l and d2l_available:
        try:
            lip_result = run_diff2lip(fomm_frames, audio, tmp)
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
        tmp_remux = tmp / "remux.mp4"
        _remux_video_with_audio(final_result, Path(audio), tmp_remux)
        shutil.copy(str(tmp_remux), out_path_abs)
    else:
        tmp_encoded = tmp / "encoded.mp4"
        encode_mp4(final_result, audio, str(tmp_encoded), fps=25)

        tmp_remux = tmp / "remux.mp4"
        _remux_video_with_audio(tmp_encoded, Path(audio), tmp_remux)
        shutil.copy(str(tmp_remux), out_path_abs)

    if not _verify_has_audio(Path(out_path_abs)):
        print("[pipeline] ⚠️ WARNING: final output seems to have no audio stream after remux.")

    return out_path_abs