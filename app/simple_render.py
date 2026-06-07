"""
simple_render.py – Dependency-light avatar video renderer (ffmpeg only).

This produces a real, downloadable MP4 from a portrait image + an audio clip
WITHOUT torch, model checkpoints, or a GPU. It animates the still portrait with
a gentle Ken Burns zoom and muxes the audio, so the full product loop
(upload → render → download) works on a free CPU Hugging Face Space.

It is NOT lip-sync. It's the "demo / preview" backend used when the heavy ML
pipeline isn't available; the real FOMM/Diff2Lip/Wav2Lip pipeline
(``app.pipeline.render_pipeline``) takes over automatically on a GPU Space with
models present (see ``app.render.render``).

Only requirement: the ``ffmpeg`` binary on PATH (already in hf/Dockerfile).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger("avatar-renderer.simple")


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found on PATH; cannot render.")
    return exe


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def simple_render(*, face_image: str, audio: str, out_path: str, size: int = 720) -> str:
    """Render a portrait + audio into an MP4. Returns the absolute output path.

    Tries a gentle zoom (Ken Burns) first; falls back to a clean static frame if
    the fancy filter chain isn't supported by the installed ffmpeg.
    """
    face = Path(face_image)
    aud = Path(audio)
    if not face.exists():
        raise FileNotFoundError(f"Face image not found: {face}")
    if not aud.exists():
        raise FileNotFoundError(f"Audio file not found: {aud}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _ffmpeg()

    # Square, even dimensions; pad to fit the portrait without distortion.
    pad = (
        f"scale={size}:{size}:force_original_aspect_ratio=decrease,"
        f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
    )

    # 1) Preferred: subtle Ken Burns zoom so the result feels alive.
    kenburns = (
        f"[0:v]{pad},zoompan=z='min(zoom+0.0005,1.10)':"
        f"d=1:s={size}x{size}:fps=25,format=yuv420p[v]"
    )
    fancy = [
        ffmpeg, "-y", "-loglevel", "error",
        "-loop", "1", "-i", str(face),
        "-i", str(aud),
        "-filter_complex", kenburns,
        "-map", "[v]", "-map", "1:a",
        "-c:v", "libx264", "-tune", "stillimage", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p", "-shortest", "-movflags", "+faststart",
        str(out),
    ]

    # 2) Fallback: plain static portrait + audio (maximally compatible).
    plain = [
        ffmpeg, "-y", "-loglevel", "error",
        "-loop", "1", "-i", str(face),
        "-i", str(aud),
        "-vf", f"{pad},format=yuv420p",
        "-c:v", "libx264", "-tune", "stillimage", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p", "-shortest", "-movflags", "+faststart",
        str(out),
    ]

    try:
        _run(fancy)
    except subprocess.CalledProcessError as exc:
        log.warning("Ken Burns render failed (%s); using static fallback.",
                    exc.stderr.decode("utf-8", "ignore")[-200:] if exc.stderr else exc)
        _run(plain)

    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError("simple_render produced no output")
    log.info("simple_render -> %s (%d bytes)", out, out.stat().st_size)
    return str(out.resolve())
