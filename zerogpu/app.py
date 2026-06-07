"""
zerogpu/app.py – OPTIONAL ZeroGPU Gradio wrapper (Phase 5).

This is a *separate* Hugging Face Space (Gradio SDK, ZeroGPU hardware) that
reuses the existing rendering pipeline. It is not imported by the FastAPI
backend and adds nothing to the main app — deploy it only if you specifically
want user-quota-based ZeroGPU inference.

How it works:
    - Gradio exposes a `render(avatar, audio, quality_mode)` function.
    - `@spaces.GPU` tells ZeroGPU to attach a GPU for the duration of the call.
    - The function calls the repo's `render_pipeline(...)` and returns an MP4.

Deploy:
    1. Create a new Space → SDK: Gradio → Hardware: ZeroGPU.
    2. Include this repo (or vendor app/ + external_deps/) so `app.pipeline`
       importable, and use `zerogpu/requirements.txt`.
    3. The FastAPI backend can call this Space's API with the user's HF token so
       their ZeroGPU quota applies.

Note: the heavy model/checkpoint setup is identical to the main Space; reuse
`hf/start.sh`'s download logic or pre-bake the models.
"""

from __future__ import annotations

import os
import sys
import tempfile
import uuid
from pathlib import Path

# Make the repo root importable (so `from app.pipeline import ...` works whether
# this file is run from zerogpu/ or the repo root).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import gradio as gr

try:
    import spaces  # provided by the ZeroGPU runtime
except ImportError:  # local dev / non-ZeroGPU: no-op decorator
    class _Spaces:
        @staticmethod
        def GPU(*args, **kwargs):  # noqa: N802 - mirror the real API
            def _decorator(fn):
                return fn

            # Support both @spaces.GPU and @spaces.GPU(duration=...)
            if args and callable(args[0]):
                return args[0]
            return _decorator

    spaces = _Spaces()  # type: ignore

from app.pipeline import render_pipeline  # noqa: E402

_OUT_DIR = Path(os.environ.get("ZEROGPU_OUT_DIR", tempfile.gettempdir())) / "zerogpu-jobs"
_OUT_DIR.mkdir(parents=True, exist_ok=True)


@spaces.GPU(duration=300)
def render(avatar_path: str, audio_path: str, quality_mode: str = "auto") -> str:
    """Render a talking-avatar MP4 and return the output file path."""
    if not avatar_path or not audio_path:
        raise gr.Error("Please provide both an avatar image and an audio file.")

    out_path = str(_OUT_DIR / f"{uuid.uuid4()}.mp4")
    return render_pipeline(
        face_image=avatar_path,
        audio=audio_path,
        out_path=out_path,
        quality_mode=quality_mode,
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Avatar Renderer (ZeroGPU)") as demo:
        gr.Markdown("# Avatar Renderer — ZeroGPU\nUpload a portrait and audio to generate a talking avatar.")
        with gr.Row():
            avatar = gr.Image(type="filepath", label="Avatar image")
            audio = gr.Audio(type="filepath", label="Audio")
        quality = gr.Dropdown(
            choices=["auto", "real_time", "high_quality"],
            value="auto",
            label="Quality mode",
        )
        out = gr.Video(label="Result")
        gr.Button("Render", variant="primary").click(
            render, inputs=[avatar, audio, quality], outputs=out
        )
    return demo


if __name__ == "__main__":
    build_ui().queue().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
