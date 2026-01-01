#!/usr/bin/env python3
"""
demo_tts.py - Avatar Renderer + TTS Demo

1. Generates audio from text using the running Chatterbox TTS Server.
2. Generates a talking avatar video using that audio + a static image.

Usage:
  # Default (Hello World)
  python demo_tts.py

  # Custom text and image
  python demo_tts.py --text "Welcome to my AI blog!" --image tests/assets/alice.png

  # High quality mode
  python demo_tts.py --quality high_quality
"""

from __future__ import annotations

import argparse
import os
import time
import requests
from pathlib import Path

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

TTS_SERVER_URL = "http://localhost:4123/v1/audio/speech"
TEMP_AUDIO_FILENAME = "temp_tts_output.wav"


def print_banner() -> None:
    print("=" * 60)
    print("🗣️  Avatar Renderer + TTS Demo")
    print("=" * 60)
    print()


def human_mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.2f} MB"


def check_file(label: str, path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"{label} is a directory, expected a file: {path}")


def generate_audio_from_text(text: str, output_path: Path) -> Path:
    """
    Sends text to the local TTS server and saves the WAV file.
    """
    print(f"🎤 Generating audio for: '{text}'")
    print(f"   Server: {TTS_SERVER_URL}")

    payload = {
        "input": text,
        "language": "en",
        "voice": "female",  # Options: neutral, female, male
        "speed": 1.0,
        "temperature": 0.7,
        "stream": False
    }

    try:
        start_time = time.time()
        response = requests.post(TTS_SERVER_URL, json=payload)
        
        if response.status_code != 200:
            raise RuntimeError(f"TTS Server Error {response.status_code}: {response.text}")

        with open(output_path, "wb") as f:
            f.write(response.content)
            
        elapsed = time.time() - start_time
        size = human_mb(output_path.stat().st_size)
        print(f"   ✅ Audio generated in {elapsed:.2f}s ({size})")
        return output_path

    except requests.exceptions.ConnectionError:
        print("\n❌ CRITICAL ERROR: Could not connect to TTS Server.")
        print("   Please ensure the server is running in a separate terminal:")
        print("   $ python app/tts/chatterbox_server.py\n")
        raise


def generate_video(image_path: Path, audio_path: Path, output_path: Path, quality_mode: str, force_wav2lip: bool = False) -> Path:
    """
    Generates the avatar video using the pipeline.
    """
    print("\n🎬 Generating avatar video...")
    print(f"   Image:   {image_path}")
    print(f"   Audio:   {audio_path}")
    print(f"   Output:  {output_path}")
    print(f"   Quality: {quality_mode}")
    print()
    print("⏳ Processing (this may take a moment)...\n")

    start_time = time.time()

    # Import pipeline here to avoid heavy loading if TTS fails earlier
    from app.pipeline import render_pipeline

    result = render_pipeline(
        face_image=str(image_path.resolve()),
        audio=str(audio_path.resolve()),
        out_path=str(output_path.resolve()),
        quality_mode=quality_mode,
        force_wav2lip=force_wav2lip,
    )

    elapsed = time.time() - start_time
    result_path = Path(result)

    print("✅ Video Success!")
    print(f"   ⏱️  Time: {elapsed:.1f}s")
    print(f"   📁 Output: {result_path}")

    if result_path.exists():
        print(f"   📊 Size: {human_mb(result_path.stat().st_size)}")

    return result_path


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Avatar Renderer + TTS Demo")
    p.add_argument("--text", type=str, default="Hello! This is a fully generated avatar video test.",
                   help="Text to speak")
    p.add_argument("--image", type=Path, default=Path("tests/assets/alice.png"),
                   help="Path to portrait image (png/jpg)")
    p.add_argument("--out", type=Path, default=Path("demo_tts_result.mp4"),
                   help="Output mp4 path")
    p.add_argument("--quality", default="auto", choices=["auto", "real_time", "high_quality"],
                   help="Quality mode")
    p.add_argument("--wav2lip", action="store_true",
                   help="Force use of Wav2Lip instead of Diff2Lip for lip-sync")
    return p


def main() -> int:
    print_banner()
    args = build_argparser().parse_args()

    # 1. Validation
    print("📁 Checking input image...")
    try:
        check_file("Image", args.image)
        print(f"   ✅ Found: {args.image}")
    except Exception as e:
        print(f"   ❌ {e}")
        return 2

    print()

    # 2. Generate Audio (Step 1)
    audio_path = Path(TEMP_AUDIO_FILENAME)
    try:
        generate_audio_from_text(args.text, audio_path)
    except Exception as e:
        print(f"❌ TTS Generation Failed: {e}")
        return 1

    # 3. Generate Video (Step 2)
    try:
        result = generate_video(args.image, audio_path, args.out, args.quality, args.wav2lip)
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ Video Generation Failed")
        print("=" * 60)
        print(f"\nError: {e}\n")
        import traceback
        traceback.print_exc()
        return 1

    # Cleanup (Optional - comment out if you want to keep the audio)
    # if audio_path.exists():
    #     os.remove(audio_path)

    print("\n" + "=" * 60)
    print("🎉 Demo Complete!")
    print("=" * 60)
    print(f"\nResult saved to: {result.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())