#!/usr/bin/env python3
"""
demo_avatars.py - Bulk Avatar Generator

Generates a complete set of demo videos for different personas (Professional, Creator, NPC, etc.)
using specific scripts and their corresponding images from notebook_assets/avatars.

Usage:
  python demo_avatars.py
"""

from __future__ import annotations

import os
import time
import requests
from pathlib import Path

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

TTS_SERVER_URL = "http://localhost:4123/v1/audio/speech"
ASSETS_DIR = Path("notebook_assets/avatars")
OUTPUT_DIR = Path("output/demos")

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Avatar Definitions
# -----------------------------------------------------------------------------

AVATARS = [
    {
        "name": "professional",
        "image": "professional.png",
        "voice": "female",
        "speed": 1.0,
        "text": (
            "Good morning. I’m your AI assistant, here to provide clear and reliable support. "
            "I can answer questions, guide next steps, and respond professionally at any time. "
            "Please let me know how I can assist you today."
        )
    },
    {
        "name": "creator",
        "image": "creator.png",
        "voice": "female",
        "speed": 1.05,  # Slightly faster/energetic
        "text": (
            "Hey! Thanks for being here. "
            "I’m excited to share something new with you today. "
            "Let’s create content that feels natural, engaging, and truly alive."
        )
    },
    {
        "name": "educator",
        "image": "educator.png",
        "voice": "female",
        "speed": 0.95,  # Slightly slower/clearer
        "text": (
            "Welcome. Today we’ll explore this topic together, step by step. "
            "Take your time, and feel free to pause or replay at any moment. "
            "Learning works best when it feels clear and comfortable."
        )
    },
    {
        "name": "npc",
        "image": "npc.png",
        "voice": "male",
        "speed": 0.9,  # Dramatic/Slow
        "text": (
            "So… you finally arrived. "
            "The path ahead isn’t easy, and every choice you make will matter. "
            "If you’re ready, we can begin."
        )
    },
    {
        "name": "brand",
        "image": "brand.png",
        "voice": "female",
        "speed": 1.0,
        "text": (
            "Welcome, and thank you for joining us. "
            "We’re proud to bring you an experience built on quality and innovation. "
            "Let’s discover what’s possible together."
        )
    },
    {
        "name": "custom",
        "image": "custom.png",
        "voice": "neutral",
        "speed": 1.0,
        "text": (
            "Hello. I’m a dynamic AI avatar designed to communicate naturally. "
            "I can express emotion, adapt my tone, and speak clearly. "
            "This is a demonstration of realistic avatar rendering."
        )
    }
]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def print_banner() -> None:
    print("=" * 60)
    print("🚀 Bulk Avatar Generator")
    print("=" * 60)
    print(f"📂 Assets: {ASSETS_DIR}")
    print(f"📂 Output: {OUTPUT_DIR}")
    print("=" * 60)
    print()

def generate_audio(text: str, voice: str, speed: float, output_path: Path) -> Path:
    """Sends text to local TTS server."""
    print(f"  🎤 Generating audio ({voice}, {speed}x)...")
    
    payload = {
        "input": text,
        "language": "en",
        "voice": voice,
        "speed": speed,
        "temperature": 0.7,
        "stream": False
    }

    try:
        response = requests.post(TTS_SERVER_URL, json=payload, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"TTS Error {response.status_code}: {response.text}")
        
        with open(output_path, "wb") as f:
            f.write(response.content)
        return output_path
        
    except requests.exceptions.ConnectionError:
        print("\n❌ CRITICAL: Chatterbox TTS Server is not running on port 4123.")
        raise

def generate_video(image_path: Path, audio_path: Path, output_path: Path) -> None:
    """Runs the rendering pipeline."""
    print(f"  🎬 Rendering video...")
    
    # Lazy import to avoid loading heavy libs until needed
    from app.pipeline import render_pipeline

    render_pipeline(
        face_image=str(image_path),
        audio=str(audio_path),
        out_path=str(output_path),
        quality_mode="real_time",  # Use 'high_quality' if you have a strong GPU and time
        force_wav2lip=False
    )

# -----------------------------------------------------------------------------
# Main Loop
# -----------------------------------------------------------------------------

def main():
    print_banner()

    # check if server is up before starting loop
    try:
        requests.get(TTS_SERVER_URL.replace("/v1/audio/speech", "/health"), timeout=2)
    except:
        print("❌ Error: TTS Server is offline. Run 'python app/tts/chatterbox_server.py'")
        return

    total_start = time.time()

    for idx, avatar in enumerate(AVATARS, 1):
        name = avatar['name']
        image_file = ASSETS_DIR / avatar['image']
        audio_file = OUTPUT_DIR / f"{name}.wav"
        video_file = OUTPUT_DIR / f"{name}.mp4"

        print(f"[{idx}/{len(AVATARS)}] Processing: {name.upper()}")

        if not image_file.exists():
            print(f"  ⚠️  Image not found: {image_file}, skipping...")
            continue

        try:
            # 1. Generate Audio
            generate_audio(avatar['text'], avatar['voice'], avatar['speed'], audio_file)

            # 2. Generate Video
            start_vid = time.time()
            generate_video(image_file, audio_file, video_file)
            duration = time.time() - start_vid
            
            print(f"  ✅ Saved to: {video_file} ({duration:.1f}s)")
            
            # Optional cleanup
            if audio_file.exists():
                os.remove(audio_file)

        except Exception as e:
            print(f"  ❌ Failed: {e}")
        
        print("-" * 60)

    total_time = time.time() - total_start
    print(f"\n🎉 All tasks completed in {total_time:.1f}s")
    print(f"📂 Check {OUTPUT_DIR} for results.")

if __name__ == "__main__":
    main()