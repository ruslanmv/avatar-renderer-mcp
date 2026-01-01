#!/usr/bin/env python3
"""
Avatar Renderer Launcher - Modern Web-Based UI using Eel
Production Ready Version
"""

import os
import sys
import json
import time
import shutil
import platform
import subprocess
from pathlib import Path
from typing import Optional

# Add parent directory to path to allow importing app modules
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
WEB_DIR = Path(__file__).parent / "web"

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
(WEB_DIR / "cache").mkdir(exist_ok=True) # For temporary browser previews

SUPPORTED_LANGUAGES = {
    "en": "🇬🇧 English", "es": "🇪🇸 Spanish", "it": "🇮🇹 Italian", "fr": "🇫🇷 French",
    "de": "🇩🇪 German", "ru": "🇷🇺 Russian", "pt": "🇵🇹 Portuguese", "pl": "🇵🇱 Polish",
    "nl": "🇳🇱 Dutch", "sv": "🇸🇪 Swedish", "no": "🇳🇴 Norwegian", "da": "🇩🇰 Danish",
    "fi": "🇫🇮 Finnish", "el": "🇬🇷 Greek", "tr": "🇹🇷 Turkish", "ar": "🇸🇦 Arabic",
    "he": "🇮🇱 Hebrew", "hi": "🇮🇳 Hindi", "ja": "🇯🇵 Japanese", "ko": "🇰🇷 Korean",
    "zh": "🇨🇳 Chinese", "ms": "🇲🇾 Malay", "sw": "🇹🇿 Swahili",
}

VOICE_PROFILES = {
    "sophia": {"name": "🎀 Sophia – Friendly (Female)", "voice": "female", "temperature": 0.7, "speed": 1.0},
    "emma": {"name": "🎀 Emma – Calm (Female)", "voice": "female", "temperature": 0.6, "speed": 0.95},
    "marcus": {"name": "👔 Marcus – Professional (Male)", "voice": "male", "temperature": 0.7, "speed": 1.0},
    "neutral": {"name": "⚪ Neutral Voice", "voice": "neutral", "temperature": 0.7, "speed": 1.0},
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
}

QUALITY_MODES = {
    "high_quality": "✨ High Quality (Best Visuals)",
    "real_time": "⚡ Real-time (Fastest)",
    "auto": "🔄 Auto (Balanced)",
}

# ============================================================================
# Application State
# ============================================================================

class AppState:
    def __init__(self):
        self.is_generating = False
        self.avatar_image_path: Optional[Path] = DEFAULT_AVATAR if DEFAULT_AVATAR.exists() else None
        self.generated_audio_path: Optional[Path] = None
        self.output_video_path: Optional[Path] = None

state = AppState()

# ============================================================================
# Helper Functions
# ============================================================================

def open_path_in_os(path: str):
    """Cross-platform method to open a file or directory."""
    path = str(path)
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])

# ============================================================================
# Eel Exposed Functions
# ============================================================================

@eel.expose
def get_config():
    return {
        "languages": SUPPORTED_LANGUAGES,
        "voices": VOICE_PROFILES,
        "samples": SAMPLE_TEXTS,
        "quality_modes": QUALITY_MODES,
        "default_avatar": str(state.avatar_image_path) if state.avatar_image_path else "",
        "output_dir": str(OUTPUT_DIR),
        "api_url": API_BASE_URL,
    }

@eel.expose
def check_api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health/live", timeout=2) 
        if response.status_code == 200:
            return {"status": "online", "code": 200}
        
        response = requests.get(f"{API_BASE_URL}/docs", timeout=2)
        return {"status": "online", "code": response.status_code}
    except Exception as e:
        return {"status": "offline", "error": str(e)}

@eel.expose
def open_output_folder():
    """Opens the output directory in the OS file explorer."""
    try:
        open_path_in_os(OUTPUT_DIR)
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@eel.expose
def open_video_file():
    """Opens the last generated video in the OS default player."""
    if state.output_video_path and state.output_video_path.exists():
        open_path_in_os(state.output_video_path)
    else:
        return {"status": "error", "error": "No video found"}

@eel.expose
def delete_generated_video():
    """Deletes the current generated video and its cache."""
    try:
        # 1. Delete the actual output file if it exists in state
        if state.output_video_path and state.output_video_path.exists():
            try:
                state.output_video_path.unlink()
            except Exception as e:
                print(f"[WARN] Failed to delete output video: {e}")
        
        # 2. Delete the cache file used for preview
        cache_video = WEB_DIR / "cache" / "preview_video.mp4"
        if cache_video.exists():
            try:
                cache_video.unlink()
            except Exception as e:
                print(f"[WARN] Failed to delete cache video: {e}")

        # 3. Reset state
        state.output_video_path = None
        return {"status": "success", "message": "Video deleted successfully"}
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

@eel.expose
def generate_audio(text: str, language: str, voice_profile: str):
    try:
        if not text.strip():
            return {"status": "error", "error": "Please enter some text"}

        voice_data = VOICE_PROFILES.get(voice_profile, VOICE_PROFILES["neutral"])

        payload = {
            "text": text,
            "language": language,
            "voice": voice_data["voice"],
            "temperature": voice_data["temperature"],
            "speed": voice_data["speed"],
            "output_format": "file"
        }

        # Call API
        response = requests.post(f"{API_BASE_URL}{TTS_ENDPOINT}", json=payload, timeout=30)
        
        # Parse Response
        if response.status_code == 200:
            data = response.json()
            if "audio_path" in data and data["audio_path"]:
                 server_audio_path = Path(data["audio_path"])
                 if server_audio_path.exists():
                     audio_content = server_audio_path.read_bytes()
                 else:
                     return {"status": "error", "error": f"Audio file not found at {server_audio_path}"}
            elif "audio_base64" in data and data["audio_base64"]:
                 import base64
                 audio_content = base64.b64decode(data["audio_base64"])
            else:
                 return {"status": "error", "error": "API response missing audio data"}
        else:
            try:
                err_detail = response.json().get('detail', response.text)
            except:
                err_detail = response.text
            return {"status": "error", "error": f"API Error ({response.status_code}): {err_detail}"}

        # Save to Output Dir
        filename = f"audio_{int(time.time())}.wav"
        local_path = OUTPUT_DIR / filename
        local_path.write_bytes(audio_content)
        state.generated_audio_path = local_path

        # Create a copy in web/cache for browser preview
        cache_path = WEB_DIR / "cache" / "preview_audio.wav"
        cache_path.write_bytes(audio_content)

        return {
            "status": "success",
            "audio_path": str(local_path),
            "preview_url": "cache/preview_audio.wav?t=" + str(time.time()),
            "message": "Audio generated successfully!"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": f"Audio generation failed: {str(e)}"}

@eel.expose
def generate_video(avatar_path: str, quality_mode: str):
    try:
        if state.is_generating:
            return {"status": "error", "error": "Generation in progress"}

        img_path = Path(avatar_path) if avatar_path else state.avatar_image_path
        
        if not img_path or not img_path.exists():
            return {"status": "error", "error": "Invalid avatar image path"}
        
        if not state.generated_audio_path or not state.generated_audio_path.exists():
            return {"status": "error", "error": "Audio file missing. Generate audio first."}

        state.is_generating = True
        eel.update_status("Initializing AI engine...")

        try:
            from app.pipeline import render_pipeline 
        except ImportError as e:
            state.is_generating = False
            return {"status": "error", "error": f"Backend pipeline import failed: {e}"}

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_out = OUTPUT_DIR / f"avatar_{timestamp}.mp4"

        eel.update_status(f"Rendering ({QUALITY_MODES.get(quality_mode, quality_mode)})... This may take time.")

        # --- EXECUTE PIPELINE ---
        result_path = render_pipeline(
            face_image=str(img_path),
            audio=str(state.generated_audio_path),
            out_path=str(video_out),
            quality_mode=quality_mode
        )

        state.output_video_path = Path(result_path)
        state.is_generating = False
        
        cache_video = WEB_DIR / "cache" / "preview_video.mp4"
        shutil.copy(result_path, cache_video)

        return {
            "status": "success",
            "video_path": str(result_path),
            "preview_url": "cache/preview_video.mp4?t=" + str(time.time()),
            "message": "Video generated!"
        }

    except Exception as e:
        state.is_generating = False
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": f"Render failed: {str(e)}"}

@eel.expose
def set_avatar_path(path: str):
    try:
        clean_path = path.strip('"').strip("'")
        p = Path(clean_path)
        
        if p.exists() and p.is_file():
            state.avatar_image_path = p
            cache_img = WEB_DIR / "cache" / "preview_avatar.png"
            shutil.copy(p, cache_img)
            
            return {
                "status": "success", 
                "path": str(p),
                "preview_url": "cache/preview_avatar.png?t=" + str(time.time())
            }
        return {"status": "error", "error": "File not found"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@eel.expose
def get_sample_text(language: str, sample_type: str):
    samples = SAMPLE_TEXTS.get(language, SAMPLE_TEXTS.get("en", {}))
    return samples.get(sample_type, "")

# ============================================================================
# Main
# ============================================================================

def main():
    try:
        print("[INFO] Starting Launcher...")
        eel.init(str(WEB_DIR))
        
        cache_dir = WEB_DIR / "cache"
        if cache_dir.exists():
            for f in cache_dir.glob("*"):
                try: f.unlink()
                except: pass

        eel.start(
            'index.html',
            size=(1280, 850),
            port=8080,
            mode='chrome',
            cmdline_args=['--disable-http-cache'] 
        )
    except Exception as e:
        print(f"[FATAL] {e}")

if __name__ == "__main__":
    main()