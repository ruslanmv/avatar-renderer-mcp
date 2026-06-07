"""
space_app.py — Avatar Renderer · ZeroGPU Gradio Space
=====================================================

This is the Hugging Face Space entry point (app_file in README.md). It runs as a
Gradio SDK Space on ZeroGPU hardware:

  • `generate()` is decorated with `@spaces.GPU`, so ZeroGPU attaches a GPU for
    the duration of each inference call.
  • Inference reuses the repo pipeline via `app.render.render`, which tries the
    full GPU lip-sync pipeline and gracefully falls back to the ffmpeg renderer
    so a video is ALWAYS returned (the Space never hard-fails).
  • A named API endpoint (api_name="predict") is exposed so the Vercel frontend
    can run inferences via @gradio/client.

The file is named `space_app.py` (not `app.py`) on purpose: the repo ships an
`app/` Python package, and a same-named `app.py` at the root would shadow it.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import gradio as gr

# ZeroGPU runtime provides `spaces`; locally we fall back to a no-op decorator.
try:
    import spaces  # type: ignore
except Exception:  # pragma: no cover - only hit off-ZeroGPU
    class _SpacesShim:
        @staticmethod
        def GPU(*args, **kwargs):
            def _wrap(fn):
                return fn
            return _wrap(args[0]) if args and callable(args[0]) else _wrap
    spaces = _SpacesShim()  # type: ignore

from app.render import render  # lazy/heavy imports happen inside render()

_OUT_DIR = Path(tempfile.gettempdir()) / "zerogpu-jobs"
_OUT_DIR.mkdir(parents=True, exist_ok=True)

GPU_DURATION = int(os.getenv("ZEROGPU_DURATION", "120"))


@spaces.GPU(duration=GPU_DURATION)
def generate(image_path: str, audio_path: str, quality_mode: str = "auto") -> str:
    """Generate a talking-avatar video on GPU and return the output file path.

    Runs inside ZeroGPU's GPU allocation. `render()` selects the full ML pipeline
    when models + CUDA are available, otherwise the ffmpeg fallback.
    """
    if not image_path or not audio_path:
        raise gr.Error("Please provide both a portrait image and an audio file.")

    out_path = str(_OUT_DIR / f"{uuid.uuid4()}.mp4")
    try:
        return render(
            face_image=image_path,
            audio=audio_path,
            out_path=out_path,
            quality_mode=quality_mode or "auto",
        )
    except Exception as exc:  # surface a clean error in the UI/API
        raise gr.Error(f"Rendering failed: {exc}") from exc


# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
_DESCRIPTION = """
# 🎭 Avatar Renderer — ZeroGPU
Turn a **portrait photo** + an **audio clip** into a talking-avatar video.
Runs on Hugging Face **ZeroGPU**. Also callable as an API from the web app.
"""


def build_ui() -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="cyan", secondary_hue="blue")
    with gr.Blocks(title="Avatar Renderer — ZeroGPU", theme=theme) as demo:
        gr.Markdown(_DESCRIPTION)
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="filepath", label="Portrait image")
                audio = gr.Audio(type="filepath", label="Audio")
                quality = gr.Dropdown(
                    choices=["auto", "real_time", "high_quality"],
                    value="auto",
                    label="Quality mode",
                )
                btn = gr.Button("Generate video", variant="primary")
            with gr.Column(scale=1):
                out = gr.Video(label="Result", autoplay=True)

        # Named endpoint for @gradio/client (the Vercel frontend calls "/predict").
        btn.click(generate, inputs=[image, audio, quality], outputs=out, api_name="predict")

    return demo


demo = build_ui()

if __name__ == "__main__":
    demo.queue(max_size=20).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
