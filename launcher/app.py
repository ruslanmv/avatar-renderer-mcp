#!/usr/bin/env python3
"""
Avatar Renderer Launcher - Modern Web-Based UI using Eel

A user-friendly alternative to the tkinter GUI that provides a modern,
web-based interface for generating talking avatar videos.
"""

import os
import sys
import json
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent directory to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import eel
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")
TTS_ENDPOINT = "/text-to-audio"
DEFAULT_AVATAR = ROOT_DIR / "tests/assets/alice.png"
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

SUPPORTED_LANGUAGES = {
    "en": "🇬🇧 English", "es": "🇪🇸 Spanish", "it": "🇮🇹 Italian", "fr": "🇫🇷 French",
    "de": "🇩🇪 German", "ru": "🇷🇺 Russian", "pt": "🇵🇹 Portuguese", "pl": "🇵🇱 Polish",
    "nl": "🇳🇱 Dutch", "sv": "🇸🇪 Swedish", "no": "🇳🇴 Norwegian", "da": "🇩🇰 Danish",
    "fi": "🇫🇮 Finnish", "el": "🇬🇷 Greek", "tr": "🇹🇷 Turkish", "ar": "🇸🇦 Arabic",
    "he": "🇮🇱 Hebrew", "hi": "🇮🇳 Hindi", "ja": "🇯🇵 Japanese", "ko": "🇰🇷 Korean",
    "zh": "🇨🇳 Chinese", "ms": "🇲🇾 Malay", "sw": "🇹🇿 Swahili",
}

VOICE_PROFILES = {
    "sophia": {"name": "🎀 Sophia – Friendly (Female)", "voice": "female", "temperature": 0.7, "cfg_weight": 0.4, "exaggeration": 0.35, "speed": 1.0},
    "emma": {"name": "🎀 Emma – Calm (Female)", "voice": "female", "temperature": 0.6, "cfg_weight": 0.5, "exaggeration": 0.3, "speed": 0.95},
    "luna": {"name": "🎀 Luna – Energetic (Female)", "voice": "female", "temperature": 0.8, "cfg_weight": 0.3, "exaggeration": 0.45, "speed": 1.05},
    "marcus": {"name": "👔 Marcus – Professional (Male)", "voice": "male", "temperature": 0.7, "cfg_weight": 0.4, "exaggeration": 0.3, "speed": 1.0},
    "ethan": {"name": "👔 Ethan – Warm (Male)", "voice": "male", "temperature": 0.65, "cfg_weight": 0.5, "exaggeration": 0.35, "speed": 0.95},
    "neutral": {"name": "⚪ Neutral Voice", "voice": "neutral", "temperature": 0.7, "cfg_weight": 0.4, "exaggeration": 0.35, "speed": 1.0},
}

SAMPLE_TEXTS = {
    "en": {
        "greeting": "Hello! I'm your AI assistant. How can I help you today?",
        "professional": "Thank you for contacting support. I'll be happy to assist you."
    },
    "es": {
        "greeting": "¡Hola! Soy tu asistente de IA. ¿Cómo puedo ayudarte hoy?",
        "professional": "Gracias por contactar a soporte. Estaré encantado de asistirte."
    },
    "it": {
        "greeting": "Ciao! Sono il tuo assistente AI. Come posso aiutarti oggi?",
        "professional": "Grazie per aver contattato il supporto."
    },
    "fr": {
        "greeting": "Bonjour! Je suis votre assistant IA.",
        "professional": "Merci d'avoir contacté notre équipe de support."
    },
}

QUALITY_MODES = {
    "auto": "🔄 Auto (Automatic)",
    "real_time": "⚡ Real-time (Fast)",
    "high_quality": "✨ High Quality (Best)",
}

# ============================================================================
# Application State
# ============================================================================

class AppState:
    """Global application state."""
    def __init__(self):
        self.is_generating = False
        self.avatar_image_path: Optional[Path] = DEFAULT_AVATAR
        self.generated_audio_path: Optional[Path] = None
        self.output_video_path: Optional[Path] = None

state = AppState()

# ============================================================================
# Eel Exposed Functions (Called from JavaScript)
# ============================================================================

@eel.expose
def get_config():
    """Return initial configuration to the frontend."""
    return {
        "languages": SUPPORTED_LANGUAGES,
        "voices": VOICE_PROFILES,
        "samples": SAMPLE_TEXTS,
        "quality_modes": QUALITY_MODES,
        "default_avatar": str(DEFAULT_AVATAR),
        "output_dir": str(OUTPUT_DIR),
        "api_url": API_BASE_URL,
    }

@eel.expose
def check_api_health():
    """Check if the backend API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health/live", timeout=3)
        return {"status": "online", "code": response.status_code}
    except Exception as e:
        return {"status": "offline", "error": str(e)}

@eel.expose
def generate_audio(text: str, language: str, voice_profile: str):
    """
    Generate audio from text.

    Args:
        text: Text to convert to speech
        language: Language code (e.g., 'en', 'es')
        voice_profile: Voice profile key (e.g., 'sophia', 'marcus')

    Returns:
        Dict with status and audio path or error
    """
    try:
        if not text.strip():
            return {"status": "error", "error": "Please enter some text"}

        # Get voice settings
        voice_data = VOICE_PROFILES.get(voice_profile, VOICE_PROFILES["neutral"])

        payload = {
            "text": text,
            "language": language,
            "voice": voice_data["voice"],
            "temperature": voice_data["temperature"],
            "speed": voice_data["speed"],
            "output_format": "file"
        }

        response = requests.post(f"{API_BASE_URL}{TTS_ENDPOINT}", json=payload, timeout=30)
        response.raise_for_status()

        data = response.json()
        if data["status"] != "success":
            return {"status": "error", "error": data.get("error", "Unknown error")}

        # Get audio path from response
        audio_path = Path(data["audio_path"])
        if not audio_path.exists():
            return {"status": "error", "error": "Audio file not found on server"}

        # Copy to local output
        local_path = OUTPUT_DIR / f"audio_{int(time.time())}.wav"
        local_path.write_bytes(audio_path.read_bytes())

        state.generated_audio_path = local_path

        return {
            "status": "success",
            "audio_path": str(local_path),
            "message": "Audio generated successfully!"
        }

    except Exception as e:
        return {"status": "error", "error": f"Audio generation failed: {str(e)}"}

@eel.expose
def generate_video(avatar_path: str, quality_mode: str):
    """
    Generate talking avatar video.

    Args:
        avatar_path: Path to avatar image
        quality_mode: Quality mode key (e.g., 'auto', 'real_time', 'high_quality')

    Returns:
        Dict with status and video path or error
    """
    try:
        if state.is_generating:
            return {"status": "error", "error": "Generation already in progress"}

        avatar_path_obj = Path(avatar_path) if avatar_path else state.avatar_image_path
        if not avatar_path_obj or not avatar_path_obj.exists():
            return {"status": "error", "error": "Please select a valid avatar image"}

        if not state.generated_audio_path or not state.generated_audio_path.exists():
            return {"status": "error", "error": "Please generate audio first"}

        state.is_generating = True

        # Update frontend
        eel.update_status("Initializing pipeline...")

        # Lazy import to avoid loading ML libraries at startup
        try:
            from app.pipeline import render_pipeline
        except ImportError as e:
            state.is_generating = False
            return {"status": "error", "error": f"Failed to import pipeline: {str(e)}"}

        # Prepare output path
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_out = OUTPUT_DIR / f"avatar_{timestamp}.mp4"

        # Update frontend
        eel.update_status(f"Rendering video ({QUALITY_MODES[quality_mode]})...")

        # Run pipeline
        result_path = render_pipeline(
            face_image=str(avatar_path_obj),
            audio=str(state.generated_audio_path),
            out_path=str(video_out),
            quality_mode=quality_mode
        )

        state.output_video_path = Path(result_path)
        state.is_generating = False

        return {
            "status": "success",
            "video_path": str(result_path),
            "message": "Video generated successfully!"
        }

    except Exception as e:
        state.is_generating = False
        return {"status": "error", "error": f"Video generation failed: {str(e)}"}

@eel.expose
def set_avatar_path(path: str):
    """Set the avatar image path."""
    try:
        avatar_path = Path(path)
        if avatar_path.exists():
            state.avatar_image_path = avatar_path
            return {"status": "success", "path": str(avatar_path)}
        else:
            return {"status": "error", "error": "File not found"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@eel.expose
def get_sample_text(language: str, sample_type: str):
    """Get sample text for a language."""
    samples = SAMPLE_TEXTS.get(language, SAMPLE_TEXTS["en"])
    return samples.get(sample_type, "")

# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """
    Main entry point for the launcher application.

    Starts the Eel web server and opens the UI.
    """
    try:
        print("[INFO] Starting Avatar Renderer Launcher...")

        # Initialize Eel with the web folder
        web_dir = Path(__file__).parent / "web"
        eel.init(str(web_dir))

        print(f"[INFO] Web directory: {web_dir}")
        print(f"[INFO] API URL: {API_BASE_URL}")
        print(f"[INFO] Output directory: {OUTPUT_DIR}")

        # Start the application
        print("[INFO] Opening launcher UI...")

        # Start Eel (opens browser automatically)
        eel.start(
            'index.html',
            size=(1200, 900),
            port=8080,
            mode='chrome',  # Try Chrome app mode first
            close_callback=lambda page, sockets: print("\n[INFO] Launcher closed")
        )

        return 0

    except KeyboardInterrupt:
        print("\n[INFO] Launcher interrupted by user")
        return 0
    except Exception as e:
        print(f"[ERROR] Launcher startup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
