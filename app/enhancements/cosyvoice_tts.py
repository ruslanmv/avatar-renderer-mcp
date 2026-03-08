"""
Enhancement #7: CosyVoice Emotional Prosody TTS
=================================================

Integrates CosyVoice (by Alibaba) for emotion-conditioned text-to-speech.
When Enhancement #1 detects emotion, CosyVoice can generate speech with
matching emotional prosody (happy, sad, angry tones).

Pipeline stage: tts (runs before the avatar rendering pipeline)
Priority: 10

This is an ADDITIVE TTS option alongside Chatterbox. It's used when:
  1. Emotion-aware speech is requested
  2. CosyVoice server is available
  3. The user opts in via the "cosyvoice_tts" enhancement flag

Models/Dependencies:
  - CosyVoice server running at COSYVOICE_URL (default: http://localhost:5001)
  - OR external_deps/CosyVoice/ with local model
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from . import Enhancement, EnhancementContext, registry

logger = logging.getLogger(__name__)

_COSYVOICE_URL = os.environ.get("COSYVOICE_URL", "http://localhost:5001")
_EXT_DEPS_DIR = Path(os.environ.get("EXT_DEPS_DIR", "external_deps"))
_COSYVOICE_REPO = _EXT_DEPS_DIR / "CosyVoice"

# Emotion → CosyVoice style prompt mapping
EMOTION_STYLE_MAP = {
    "happy": "cheerful and warm",
    "sad": "melancholic and gentle",
    "angry": "firm and intense",
    "surprised": "excited and animated",
    "fearful": "nervous and trembling",
    "disgusted": "cold and dismissive",
    "neutral": "calm and clear",
}


def _cosyvoice_server_available() -> bool:
    """Check if CosyVoice server is reachable."""
    try:
        import urllib.request
        req = urllib.request.Request(f"{_COSYVOICE_URL}/health", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def _cosyvoice_local_available() -> bool:
    """Check if CosyVoice can be run locally."""
    return _COSYVOICE_REPO.exists() and (
        (_COSYVOICE_REPO / "inference.py").exists() or
        (_COSYVOICE_REPO / "cosyvoice" / "cli" / "cosyvoice.py").exists()
    )


def _generate_via_server(
    text: str,
    output_path: str,
    style_prompt: str = "calm and clear",
    language: str = "en",
) -> bool:
    """Generate speech via CosyVoice HTTP API."""
    try:
        import json
        import urllib.request

        payload = json.dumps({
            "text": text,
            "style_prompt": style_prompt,
            "language": language,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{_COSYVOICE_URL}/v1/audio/speech",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            audio_data = resp.read()
            Path(output_path).write_bytes(audio_data)
            return True

    except Exception as e:
        logger.warning("CosyVoice server request failed: %s", e)
        return False


def _generate_via_local(
    text: str,
    output_path: str,
    style_prompt: str = "calm and clear",
) -> bool:
    """Generate speech via local CosyVoice inference."""
    try:
        inference_script = _COSYVOICE_REPO / "inference.py"
        if not inference_script.exists():
            inference_script = _COSYVOICE_REPO / "cosyvoice" / "cli" / "cosyvoice.py"

        env = os.environ.copy()
        env["PYTHONPATH"] = (
            f"{str(_COSYVOICE_REPO.resolve())}{os.pathsep}"
            f"{env.get('PYTHONPATH', '')}"
        )

        cmd = [
            sys.executable, str(inference_script),
            "--text", text,
            "--output", output_path,
            "--style_prompt", style_prompt,
        ]

        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
        return result.returncode == 0 and Path(output_path).exists()

    except Exception as e:
        logger.warning("CosyVoice local inference failed: %s", e)
        return False


class CosyVoiceTTSEnhancement(Enhancement):
    """CosyVoice: emotion-conditioned multilingual text-to-speech."""

    @property
    def name(self) -> str:
        return "cosyvoice_tts"

    @property
    def stage(self) -> str:
        return "tts"

    @property
    def priority(self) -> int:
        return 10

    @property
    def description(self) -> str:
        return (
            "CosyVoice emotional prosody TTS by Alibaba. Generates speech with "
            "emotion-matching tone (happy, sad, angry). Adds expressive speech "
            "alongside or replacing Chatterbox TTS."
        )

    def is_available(self) -> bool:
        return _cosyvoice_server_available() or _cosyvoice_local_available()

    def apply(self, ctx: EnhancementContext) -> EnhancementContext:
        # Only apply if there's text to synthesize and no audio yet
        if not ctx.transcript:
            logger.info("CosyVoice skipped: no transcript to synthesize")
            return ctx

        # Don't override existing audio
        if ctx.audio_path and Path(ctx.audio_path).exists():
            logger.info("CosyVoice skipped: audio already provided")
            return ctx

        # Determine emotional style from context
        emotion = ctx.detected_emotion or "neutral"
        style_prompt = EMOTION_STYLE_MAP.get(emotion, EMOTION_STYLE_MAP["neutral"])

        output_path = str(ctx.tmp_dir / "cosyvoice_output.wav")

        # Try server first, then local
        success = False
        if _cosyvoice_server_available():
            success = _generate_via_server(ctx.transcript, output_path, style_prompt)
        if not success and _cosyvoice_local_available():
            success = _generate_via_local(ctx.transcript, output_path, style_prompt)

        if success:
            ctx.audio_path = output_path
            logger.info("CosyVoice generated emotional speech (emotion=%s)", emotion)
        else:
            logger.warning("CosyVoice TTS failed; audio_path unchanged")

        return ctx


# Auto-register
registry.register(CosyVoiceTTSEnhancement())
