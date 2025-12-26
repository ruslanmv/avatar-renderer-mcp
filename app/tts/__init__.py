"""TTS (Text-to-Speech) module for Avatar Renderer MCP.

This module provides text-to-speech capabilities using the Chatterbox TTS service.
"""

from .chatterbox_client import (
    ChatterboxTtsError,
    chatterbox_health,
    chatterbox_health_async,
    tts_wav_base64,
    tts_wav_base64_async,
    tts_wav_bytes,
    tts_wav_bytes_async,
)

__all__ = [
    "ChatterboxTtsError",
    "tts_wav_bytes",
    "tts_wav_base64",
    "chatterbox_health",
    "tts_wav_bytes_async",
    "tts_wav_base64_async",
    "chatterbox_health_async",
]
