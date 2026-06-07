"""
space_app.py — Avatar Renderer · ZeroGPU Gradio Space
=====================================================

Hugging Face Space entry point (app_file in README.md), running on ZeroGPU:

  • Text-to-speech (edge-tts): type text + pick a voice/speed/pitch and the Space
    synthesizes the audio, OR upload your own audio.
  • Lip-sync render reuses app.render.render (in-process Wav2Lip + GFPGAN, with a
    safe ffmpeg fallback) inside the @spaces.GPU allocation.
  • Optional naturalness add-ons (head motion, blink/gaze) you can toggle to
    compare more vs. less robotic results.
  • Named API endpoint (api_name="predict") for the Vercel frontend via
    @gradio/client.

Named space_app.py (not app.py) so it doesn't shadow the repo's app/ package.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import uuid
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("space_app")

import gradio as gr

from app.render import render  # lazy/heavy imports happen inside render()

# Pre-fetch Wav2Lip code + weights at startup (outside the GPU window).
try:
    from app.lipsync import ensure_setup

    ensure_setup()
except Exception:
    pass

_OUT_DIR = Path(tempfile.gettempdir()) / "zerogpu-jobs"
_OUT_DIR.mkdir(parents=True, exist_ok=True)

GPU_DURATION = int(os.getenv("ZEROGPU_DURATION", "120"))

_ON_ZEROGPU = os.environ.get("SPACES_ZERO_GPU", "").lower() in ("true", "1")
if _ON_ZEROGPU:
    import spaces

    def gpu(fn):
        return spaces.GPU(duration=GPU_DURATION)(fn)
else:
    def gpu(fn):
        return fn


# Naturalness add-ons — (label shown in UI, enhancement id sent to backend).
# Tuple choices make the CheckboxGroup accept ids over the API too.
ADDON_CHOICES = [
    ("Head motion (breathing/sway)", "gesture_animation"),
    ("Eye blink & gaze", "eye_gaze_blink"),
    ("Emotion analysis", "emotion_expressions"),
]

# Curated edge-tts neural voices (label, voice id) across languages.
VOICE_CHOICES = [
    ("English (US) — Aria, female", "en-US-AriaNeural"),
    ("English (US) — Guy, male", "en-US-GuyNeural"),
    ("English (UK) — Sonia, female", "en-GB-SoniaNeural"),
    ("English (UK) — Ryan, male", "en-GB-RyanNeural"),
    ("Spanish (ES) — Elvira, female", "es-ES-ElviraNeural"),
    ("Spanish (MX) — Jorge, male", "es-MX-JorgeNeural"),
    ("French — Denise, female", "fr-FR-DeniseNeural"),
    ("German — Katja, female", "de-DE-KatjaNeural"),
    ("Italian — Elsa, female", "it-IT-ElsaNeural"),
    ("Portuguese (BR) — Francisca, female", "pt-BR-FranciscaNeural"),
    ("Hindi — Swara, female", "hi-IN-SwaraNeural"),
    ("Chinese — Xiaoxiao, female", "zh-CN-XiaoxiaoNeural"),
    ("Japanese — Nanami, female", "ja-JP-NanamiNeural"),
    ("Arabic — Salma, female", "ar-EG-SalmaNeural"),
    ("Russian — Svetlana, female", "ru-RU-SvetlanaNeural"),
]
_DEFAULT_VOICE = "en-US-AriaNeural"


def synthesize_tts(text: str, voice: str, speed_pct: int, pitch_hz: int) -> str:
    """Convert text to an MP3 with edge-tts; returns the file path."""
    import edge_tts

    out = str(_OUT_DIR / f"tts_{uuid.uuid4()}.mp3")
    rate = f"{int(speed_pct):+d}%"
    pitch = f"{int(pitch_hz):+d}Hz"

    async def _go():
        comm = edge_tts.Communicate(text, voice or _DEFAULT_VOICE, rate=rate, pitch=pitch)
        await comm.save(out)

    asyncio.run(_go())
    log.info("TTS: %d chars, voice=%s, rate=%s, pitch=%s -> %s", len(text), voice, rate, pitch, out)
    return out


METHOD_CHOICES = [
    ("Best — mouth + GFPGAN (sharp)", "wav2lip_gfpgan"),
    ("Full face — head motion, static background", "fullface"),
    ("Basic lip-sync (no GFPGAN)", "wav2lip"),
    ("Simple (no lip-sync)", "simple"),
    ("Auto", "auto"),
]


@gpu
def _gpu_render(image_path: str, audio_path: str, addons, method, quality_mode) -> str:
    """GPU-allocated render (Wav2Lip + GFPGAN + add-ons / chosen method/tier)."""
    out_path = str(_OUT_DIR / f"{uuid.uuid4()}.mp4")
    return render(
        face_image=image_path,
        audio=audio_path,
        out_path=out_path,
        enhancements=(addons or None),
        method=(method or "auto"),
        quality_mode=(quality_mode or "standard"),
    )


def generate(image_path, audio_path, text, voice, speed, pitch, quality_mode, addons, method="auto"):
    """UI/API handler: optional TTS from text, then GPU lip-sync render.

    Audio source: if `text` is provided it is synthesized (edge-tts) and used;
    otherwise the uploaded `audio_path` is used. TTS runs outside the GPU window.
    """
    if not image_path:
        raise gr.Error("Please provide a portrait image.")

    if text and text.strip():
        try:
            audio_path = synthesize_tts(text.strip(), voice, speed or 0, pitch or 0)
        except Exception as exc:
            raise gr.Error(f"Text-to-speech failed: {exc}") from exc

    if not audio_path:
        raise gr.Error("Type some text to speak, or upload an audio file.")

    # Map any labels to ids (defensive — tuple choices already send ids).
    label_to_id = {label: eid for label, eid in ADDON_CHOICES}
    addons = [label_to_id.get(a, a) for a in (addons or [])]

    try:
        return _gpu_render(image_path, audio_path, addons, method or "auto", quality_mode or "standard")
    except Exception as exc:
        raise gr.Error(f"Rendering failed: {exc}") from exc


_DESCRIPTION = """
# 🎭 Avatar Renderer — ZeroGPU
Turn a **portrait photo** into a talking-avatar video. **Type text** (synthesized
to speech) or **upload audio**. Toggle the naturalness add-ons to compare results.
Runs on Hugging Face **ZeroGPU**; also callable as an API from the web app.
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

                with gr.Tab("Type text (TTS)"):
                    text = gr.Textbox(
                        label="Text to speak",
                        placeholder="Hello! I'm your AI avatar.",
                        lines=3,
                    )
                    voice = gr.Dropdown(choices=VOICE_CHOICES, value=_DEFAULT_VOICE, label="Voice")
                    with gr.Row():
                        speed = gr.Slider(-50, 50, value=0, step=5, label="Speed %")
                        pitch = gr.Slider(-20, 20, value=0, step=2, label="Pitch (Hz)")
                with gr.Tab("Upload audio"):
                    audio = gr.Audio(type="filepath", label="Audio (used if no text above)")

                quality = gr.Dropdown(
                    choices=["auto", "real_time", "high_quality"], value="auto", label="Quality mode"
                )
                method = gr.Dropdown(
                    choices=METHOD_CHOICES, value="wav2lip_gfpgan", label="Generation method",
                    info="Compare approaches: best mouth, full-face head motion, basic, or simple.",
                )
                addons = gr.CheckboxGroup(
                    choices=ADDON_CHOICES,
                    value=[],
                    label="Naturalness add-ons (toggle to compare)",
                    info="Subtle head motion, blinking and gaze to look less robotic.",
                )
                btn = gr.Button("Generate video", variant="primary")
            with gr.Column(scale=1):
                out = gr.Video(
                    label="Result (sample shown — generate your own on the left)",
                    value=_DEMO_VIDEO if Path(_DEMO_VIDEO).exists() else None,
                    autoplay=True,
                )

        if Path(_EX_IMAGE).exists() and Path(_EX_AUDIO).exists():
            gr.Examples(
                examples=[[_EX_IMAGE, "Hello! Welcome to the avatar renderer demo.", _DEFAULT_VOICE]],
                inputs=[image, text, voice],
                label="Try this example (text-to-speech)",
            )

        btn.click(
            generate,
            inputs=[image, audio, text, voice, speed, pitch, quality, addons, method],
            outputs=out,
            api_name="predict",
        )

    return demo


demo = build_ui()

if __name__ == "__main__":
    demo.queue(max_size=20).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
