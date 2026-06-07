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

import logging
import os
import tempfile
import uuid
from pathlib import Path

# Surface app render-path logs (otherwise INFO from our modules is suppressed).
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

import gradio as gr

from app.render import render  # lazy/heavy imports happen inside render()

# Pre-fetch the Wav2Lip code + weights at startup (outside the GPU window) so the
# first inference call doesn't spend its GPU budget cloning/downloading. Safe to
# fail — render() falls back to the ffmpeg renderer.
try:
    from app.lipsync import ensure_setup

    ensure_setup()
except Exception:  # never block startup on this
    pass

_OUT_DIR = Path(tempfile.gettempdir()) / "zerogpu-jobs"
_OUT_DIR.mkdir(parents=True, exist_ok=True)

GPU_DURATION = int(os.getenv("ZEROGPU_DURATION", "120"))

# Use the real ZeroGPU decorator only on actual ZeroGPU hardware (env set by HF).
# On cpu-basic (e.g. while waiting for a ZeroGPU slot) it is a no-op, so the app
# still runs and serves the ffmpeg fallback instead of crashing.
_ON_ZEROGPU = os.environ.get("SPACES_ZERO_GPU", "").lower() in ("true", "1")
if _ON_ZEROGPU:
    import spaces  # provided by the ZeroGPU runtime

    def gpu(fn):
        return spaces.GPU(duration=GPU_DURATION)(fn)
else:
    def gpu(fn):
        return fn


# Natural-motion add-ons users can toggle to compare (label -> enhancement id).
ADDON_CHOICES = [
    ("Head motion (breathing/sway)", "gesture_animation"),
    ("Eye blink & gaze", "eye_gaze_blink"),
    ("Emotion analysis", "emotion_expressions"),
]


@gpu
def generate(image_path: str, audio_path: str, quality_mode: str = "auto", addons=None) -> str:
    """Generate a talking-avatar video on GPU and return the output file path.

    `addons` is a list of enhancement ids (or labels) to apply for a more natural,
    less robotic result. Runs inside ZeroGPU's GPU allocation.
    """
    if not image_path or not audio_path:
        raise gr.Error("Please provide both a portrait image and an audio file.")

    # Accept either ids or human labels from the UI/clients.
    label_to_id = {label: eid for label, eid in ADDON_CHOICES}
    addons = [label_to_id.get(a, a) for a in (addons or [])]

    out_path = str(_OUT_DIR / f"{uuid.uuid4()}.mp4")
    try:
        return render(
            face_image=image_path,
            audio=audio_path,
            out_path=out_path,
            quality_mode=quality_mode or "auto",
            enhancements=addons or None,
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


_HERE = Path(__file__).resolve().parent
_DEMO_VIDEO = str(_HERE / "assets" / "demo.mp4")
_EX_IMAGE = str(_HERE / "assets" / "alice.png")
_EX_AUDIO = str(_HERE / "assets" / "hello.wav")


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
                addons = gr.CheckboxGroup(
                    choices=[label for label, _ in ADDON_CHOICES],
                    value=[],
                    label="Naturalness add-ons (toggle to compare)",
                    info="Add subtle head motion, blinking and gaze to look less robotic.",
                )
                btn = gr.Button("Generate video", variant="primary")
            with gr.Column(scale=1):
                # Pre-loaded so visitors see a sample result the moment they open
                # the Space (without spending any GPU quota).
                out = gr.Video(
                    label="Result (sample shown — generate your own on the left)",
                    value=_DEMO_VIDEO if Path(_DEMO_VIDEO).exists() else None,
                    autoplay=True,
                )

        # One-click example using bundled sample inputs.
        if Path(_EX_IMAGE).exists() and Path(_EX_AUDIO).exists():
            gr.Examples(
                examples=[[_EX_IMAGE, _EX_AUDIO, "auto"]],
                inputs=[image, audio, quality],
                label="Try this example",
            )

        # Named endpoint for @gradio/client (the Vercel frontend calls "/predict").
        btn.click(generate, inputs=[image, audio, quality, addons], outputs=out, api_name="predict")

    return demo


demo = build_ui()

if __name__ == "__main__":
    demo.queue(max_size=20).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
