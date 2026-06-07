"""
lipsync.py — In-process Wav2Lip lip-sync (GPU-friendly, for the ZeroGPU Space).

Produces a *talking* face (mouth motion synced to audio) from a single portrait
image + an audio clip, WITHOUT FOMM/SadTalker and without subprocesses (so it
actually uses the GPU that ZeroGPU attaches inside @spaces.GPU).

Approach (adapted from Rudrabha/Wav2Lip inference):
  1. Lazily set up Wav2Lip: clone the repo (for its `audio` + model defs) and
     download `wav2lip_gan.pth` from the ruslanmv/avatar-renderer model repo.
  2. Detect the face box once with OpenCV's bundled Haar cascade (no s3fd weights).
  3. Build mel chunks (one per output frame) using Wav2Lip's own audio module so
     the spectrogram matches training.
  4. Run the Wav2Lip model in-process (GPU if available) and paste predicted
     mouths back into the portrait, frame by frame.
  5. Encode frames + original audio to MP4 via ffmpeg.

Any failure raises, and the caller (app.render.render) falls back to the safe
ffmpeg renderer so the product never hard-fails.
"""

from __future__ import annotations

import logging
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

log = logging.getLogger("avatar-renderer.lipsync")

WAV2LIP_REPO = os.getenv("WAV2LIP_REPO", "https://github.com/Rudrabha/Wav2Lip")
WAV2LIP_DIR = Path(os.getenv("WAV2LIP_DIR", "/tmp/Wav2Lip"))
LIPSYNC_CACHE = Path(os.getenv("LIPSYNC_CACHE", "/tmp/lipsync_models"))
MODEL_HF_REPO = os.getenv("LIPSYNC_HF_REPO", "ruslanmv/avatar-renderer")
MODEL_HF_FILE = os.getenv("LIPSYNC_HF_FILE", "wav2lip/wav2lip_gan.pth")

IMG_SIZE = 96            # Wav2Lip face crop size
MEL_STEP_SIZE = 16
FPS = 25

# Remove "mouth-within-a-mouth" remnants before restoration. Default OFF: the
# hard mouth-core mask already prevents the double-lip artifact, and the cleanup
# darkening can smudge lips/chin. Enable only for problematic smiley sources.
_MOUTH_CLEANUP = os.getenv("LIPSYNC_MOUTH_CLEANUP", "0") == "1"


def _cleanup_frame(frame, face_box=None):
    """Lazy wrapper around the mouth-artifact cleanup (avoids import cost upfront)."""
    from .enhancements.mouth_artifact_cleanup import cleanup_frame as _cf

    return _cf(frame, face_box=face_box)


def _setup() -> Path:
    """Clone Wav2Lip code + fetch the checkpoint. Returns the checkpoint path."""
    if not (WAV2LIP_DIR / "models" / "wav2lip.py").exists():
        log.info("Cloning Wav2Lip into %s", WAV2LIP_DIR)
        subprocess.run(
            ["git", "clone", "--depth=1", WAV2LIP_REPO, str(WAV2LIP_DIR)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
    if str(WAV2LIP_DIR) not in sys.path:
        sys.path.insert(0, str(WAV2LIP_DIR))

    # Make Wav2Lip's audio.py compatible with modern librosa (>=0.10), which made
    # librosa.filters.mel keyword-only. The repo calls it positionally.
    audio_py = WAV2LIP_DIR / "audio.py"
    try:
        t = audio_py.read_text()
        if "librosa.filters.mel(hp.sample_rate, hp.n_fft," in t:
            t = t.replace(
                "librosa.filters.mel(hp.sample_rate, hp.n_fft,",
                "librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft,",
            )
            audio_py.write_text(t)
            log.info("Patched Wav2Lip audio.py for modern librosa.")
    except Exception as exc:  # non-fatal; may already be compatible
        log.warning("Could not patch Wav2Lip audio.py: %s", exc)

    LIPSYNC_CACHE.mkdir(parents=True, exist_ok=True)
    ckpt = LIPSYNC_CACHE / "wav2lip_gan.pth"
    if not ckpt.exists():
        from huggingface_hub import hf_hub_download
        log.info("Downloading %s from %s", MODEL_HF_FILE, MODEL_HF_REPO)
        p = hf_hub_download(repo_id=MODEL_HF_REPO, filename=MODEL_HF_FILE, repo_type="model")
        import shutil
        shutil.copy(p, ckpt)
    return ckpt


def ensure_setup() -> bool:
    """Pre-clone the repo + download weights (call at startup, outside the GPU
    window) so the first GPU inference doesn't spend its budget downloading."""
    try:
        _setup()
        return True
    except Exception as exc:
        log.warning("Wav2Lip setup not ready: %s", exc)
        return False


def _detect_face_box(img, pad: float = 0.25):
    """Return (x1, y1, x2, y2) face box via OpenCV Haar; whole image if none."""
    import cv2

    h, w = img.shape[:2]
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return 0, 0, w, h
    x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
    px, py = int(fw * pad), int(fh * pad)
    # Cast to plain python ints — numpy ints break cv2 geometry calls
    # (cv2.ellipse / getRotationMatrix2D raise "Can't parse 'center'").
    return (
        int(max(0, x - px)), int(max(0, y - py)),
        int(min(w, x + fw + px)), int(min(h, y + fh + py)),
    )


def _get_gfpgan(device: str):
    """Return a GFPGAN restorer (sharpens the Wav2Lip mouth/face), or None."""
    try:
        # basicsr (a GFPGAN dep) imports torchvision.transforms.functional_tensor,
        # removed in torchvision>=0.17. Alias it to functional before importing.
        import sys
        try:
            import torchvision.transforms.functional_tensor  # noqa: F401
        except Exception:
            import torchvision.transforms.functional as _F
            sys.modules["torchvision.transforms.functional_tensor"] = _F

        from gfpgan import GFPGANer
        from huggingface_hub import hf_hub_download

        weights = LIPSYNC_CACHE / "GFPGANv1.3.pth"
        if not weights.exists():
            p = hf_hub_download(repo_id=MODEL_HF_REPO, filename="gfpgan/GFPGANv1.3.pth", repo_type="model")
            import shutil
            shutil.copy(p, weights)
        restorer = GFPGANer(
            model_path=str(weights), upscale=1, arch="clean",
            channel_multiplier=2, bg_upsampler=None, device=device,
        )
        log.info("GFPGAN restorer ready.")
        return restorer
    except Exception as exc:
        log.warning("GFPGAN unavailable (%s); skipping face restoration.", exc)
        return None


def _load_model(ckpt: Path, device: str):
    import torch
    from models import Wav2Lip  # from the cloned repo

    model = Wav2Lip()
    checkpoint = torch.load(str(ckpt), map_location=device)
    state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    return model.to(device).eval()


def wav2lip_render(
    *,
    face_image: str,
    audio: str,
    out_path: str,
    enhancements: "list | None" = None,
    use_gfpgan: "bool | None" = None,
    head_motion: bool = False,
) -> str:
    """Render a lip-synced talking-face MP4. Raises on failure (caller falls back)."""
    import cv2
    import numpy as np
    import torch

    ckpt = _setup()
    import audio as w2l_audio  # Wav2Lip's audio module (mel matches training)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Wav2Lip running on %s", device)

    full = cv2.imread(face_image)
    if full is None:
        raise RuntimeError(f"Could not read image: {face_image}")
    x1, y1, x2, y2 = _detect_face_box(full)
    log.info("Face box: (%d,%d,%d,%d) of image %dx%d", x1, y1, x2, y2, full.shape[1], full.shape[0])
    face = full[y1:y2, x1:x2]
    face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

    # Audio → mel → per-frame chunks
    tmp = Path(tempfile.mkdtemp(prefix="w2l_"))
    wav_path = tmp / "audio16k.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", audio, "-ar", "16000", "-ac", "1", str(wav_path)],
        check=True,
    )
    wav = w2l_audio.load_wav(str(wav_path), 16000)

    # Trim leading/trailing silence so the mouth doesn't keep moving after the
    # speech ends. The SAME trimmed span is used for the final audio mux, so
    # video and audio start/end together.
    import librosa

    audio_for_mux = audio
    try:
        _, idx = librosa.effects.trim(wav, top_db=30)
        s0, s1 = int(idx[0]), int(idx[1])
        if s1 - s0 > 16000 * 0.2:  # keep only if >0.2s of speech remains
            wav = wav[s0:s1]
            start_sec = s0 / 16000.0
            dur_sec = (s1 - s0) / 16000.0
            trimmed = tmp / "trimmed_audio.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start_sec:.3f}",
                 "-t", f"{dur_sec:.3f}", "-i", audio, str(trimmed)],
                check=True,
            )
            audio_for_mux = str(trimmed)
            log.info("Trimmed silence: %.2fs..%.2fs (%.2fs of speech)", start_sec, s1 / 16000.0, dur_sec)
    except Exception as exc:
        log.warning("Silence trim skipped: %s", exc)

    mel = w2l_audio.melspectrogram(wav)
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise RuntimeError("Mel contains NaN; check audio.")

    mel_chunks = []
    mel_idx_multiplier = 80.0 / FPS
    i = 0
    while True:
        start = int(i * mel_idx_multiplier)
        if start + MEL_STEP_SIZE > mel.shape[1]:
            mel_chunks.append(mel[:, mel.shape[1] - MEL_STEP_SIZE:])
            break
        mel_chunks.append(mel[:, start:start + MEL_STEP_SIZE])
        i += 1

    # Per-frame silence gating: freeze the mouth (keep the original resting face)
    # during quiet frames, so the mouth doesn't keep moving when there's no speech.
    spf = 16000.0 / FPS
    rms = []
    for i in range(len(mel_chunks)):
        seg = wav[int(i * spf):int((i + 1) * spf)]
        rms.append(float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0)
    peak = max(rms) if rms else 0.0
    sil_thr = peak * 0.12
    silent = [r < sil_thr for r in rms] if peak > 0 else [False] * len(mel_chunks)

    model = _load_model(ckpt, device)
    want_gfpgan = use_gfpgan if use_gfpgan is not None else (os.getenv("LIPSYNC_GFPGAN", "1") == "1")
    restorer = _get_gfpgan(device) if want_gfpgan else None

    frames_dir = tmp / "frames"
    frames_dir.mkdir()
    img_masked = face_resized.copy()
    img_masked[IMG_SIZE // 2:] = 0  # mask lower half (Wav2Lip convention)

    # Blend the Wav2Lip mouth back, but GUARANTEE the mouth interior is 100% owned
    # by the lip-sync output. A hard "core" over the mouth (set to 1) prevents the
    # original closed/smiling mouth from bleeding through the feather — which is
    # what causes the "mouth-within-a-mouth" double-lip artifact. Feathering is
    # kept only in the surrounding skin (philtrum/cheeks/chin).
    bw, bh = x2 - x1, y2 - y1

    def _mouth_mask(height, width, oy=0, ox=0):
        m = np.zeros((height, width), np.float32)
        m[oy + int(bh * 0.45):oy + bh, ox + int(bw * 0.08):ox + int(bw * 0.92)] = 1.0
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=max(2.0, bw * 0.05), sigmaY=max(2.0, bh * 0.045))
        core = np.zeros((height, width), np.float32)
        core[oy + int(bh * 0.55):oy + int(bh * 0.97), ox + int(bw * 0.18):ox + int(bw * 0.82)] = 1.0
        return np.maximum(m, core)[..., None]

    blend_mask = _mouth_mask(bh, bw)
    orig_region = full[y1:y2, x1:x2].astype(np.float32)

    # Anti-flicker: GFPGAN the portrait ONCE → a static sharp base. We then update
    # ONLY the mouth band from each per-frame result, so the forehead/eyes/cheeks
    # stay pixel-stable (no "boiling"/shimmer) and only the mouth moves.
    base_still = full.copy()
    if restorer is not None:
        try:
            _, _, base_still = restorer.enhance(
                full, has_aligned=False, only_center_face=True, paste_back=True
            )
        except Exception as exc:
            log.warning("GFPGAN base enhance failed (%s).", exc)
    # Full-image mask, hard core over the mouth (feather only in skin) — same
    # guarantee at composite time so the static base's closed mouth can't show.
    comp_mask = _mouth_mask(full.shape[0], full.shape[1], oy=y1, ox=x1)

    # Whole-head motion mask (feathered ellipse over the head). When head_motion
    # is on, the head moves subtly while the BACKGROUND stays static (the head is
    # warped and composited back over the static base only inside this mask).
    head_mask = None
    if head_motion:
        hm = np.zeros(full.shape[:2], np.float32)
        cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)
        cv2.ellipse(hm, (cx, cy), (int(bw * 0.90), int(bh * 1.10)), 0, 0, 360, 1.0, -1)
        head_mask = cv2.GaussianBlur(hm, (0, 0), sigmaX=bw * 0.07, sigmaY=bh * 0.07)[..., None]

    batch = 64
    written = 0
    # Wav2Lip channel order is [masked, reference] (lower half of the masked copy
    # is zeroed; the model inpaints the mouth from the audio).
    face6 = np.concatenate([img_masked, face_resized], axis=2)
    for b in range(0, len(mel_chunks), batch):
        chunk = mel_chunks[b:b + batch]
        img_in = np.concatenate([face6[None]] * len(chunk), axis=0) / 255.0
        mel_in = np.array([m for m in chunk])[..., None]
        img_t = torch.FloatTensor(np.transpose(img_in, (0, 3, 1, 2))).to(device)
        mel_t = torch.FloatTensor(np.transpose(mel_in, (0, 3, 1, 2))).to(device)
        with torch.no_grad():
            pred = model(mel_t, img_t)
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        for p in pred:
            gen = cv2.resize(p.astype(np.uint8), (bw, bh)).astype(np.float32)
            frame = full.copy()
            open_mouth = not (written < len(silent) and silent[written])
            if open_mouth:
                blended = orig_region * (1.0 - blend_mask) + gen * blend_mask
                frame[y1:y2, x1:x2] = blended.astype(np.uint8)
                # Remove any double-lip remnant BEFORE restoration (so GFPGAN
                # doesn't sharpen a bad mouth texture).
                if _MOUTH_CLEANUP:
                    try:
                        frame = _cleanup_frame(frame, face_box=(x1, y1, x2, y2))
                    except Exception:
                        pass
            if restorer is not None:
                try:
                    _, _, frame = restorer.enhance(
                        frame, has_aligned=False, only_center_face=True, paste_back=True
                    )
                except Exception as exc:
                    if written == 0:
                        log.warning("GFPGAN enhance failed (%s); using unrestored frames.", exc)
                    restorer = None
            # Composite ONLY the mouth band onto the static base → kills flicker.
            if frame.shape[:2] != base_still.shape[:2]:
                frame = cv2.resize(frame, (base_still.shape[1], base_still.shape[0]))
            out_frame = (base_still.astype(np.float32) * (1.0 - comp_mask)
                         + frame.astype(np.float32) * comp_mask).astype(np.uint8)

            # Whole-head motion on a STATIC background: warp the head and
            # composite it back over the static base only inside the head mask.
            if head_mask is not None:
                t = written / float(FPS)
                dx = 3.0 * math.sin(2 * math.pi * 0.25 * t)
                dy = 2.0 * math.sin(2 * math.pi * 0.18 * t + 1.0)
                ang = 0.8 * math.sin(2 * math.pi * 0.20 * t)
                sc = 1.0 + 0.006 * math.sin(2 * math.pi * 0.15 * t)
                cx, cy = float((x1 + x2) // 2), float((y1 + y2) // 2)
                M = cv2.getRotationMatrix2D((cx, cy), ang, sc)
                M[0, 2] += dx
                M[1, 2] += dy
                warped = cv2.warpAffine(
                    out_frame, M, (out_frame.shape[1], out_frame.shape[0]),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
                )
                out_frame = (base_still.astype(np.float32) * (1.0 - head_mask)
                             + warped.astype(np.float32) * head_mask).astype(np.uint8)

            cv2.imwrite(str(frames_dir / f"{written:05d}.png"), out_frame)
            written += 1

    if written == 0:
        raise RuntimeError("Wav2Lip produced no frames.")

    # Optional naturalness add-ons (eye blink/gaze, head motion, emotion) applied
    # as frame post-processing so users can toggle and compare them.
    final_dir = frames_dir
    if enhancements:
        try:
            from .enhancements import EnhancementContext, registry

            ctx = EnhancementContext(
                face_image=face_image, audio_path=audio_for_mux,
                frames_dir=frames_dir, tmp_dir=tmp, fps=FPS, device=device,
            )
            enabled = set(enhancements)
            ctx = registry.apply_stage(ctx, "pre_motion", enabled)
            ctx = registry.apply_stage(ctx, "post_process", enabled)
            if ctx.frames_dir and ctx.frames_dir.exists():
                final_dir = ctx.frames_dir
            if ctx.applied_enhancements:
                log.info("Applied add-ons: %s", ", ".join(ctx.applied_enhancements))
        except Exception as exc:
            log.warning("Add-on enhancements failed (%s); using base frames.", exc)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(FPS), "-i", str(final_dir / "%05d.png"),
            "-i", audio_for_mux,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
            out_path,
        ],
        check=True,
    )
    log.info("Wav2Lip wrote %d frames -> %s", written, out_path)
    return str(Path(out_path).resolve())
