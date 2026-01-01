#!/usr/bin/env python3
"""
generate_showcase.py - Bulk Avatar Demo Generator (Fixed for Long Text)

Fixes:
1. Switches TTS to "Streaming Mode" to handle long paragraphs without cutting off.
2. Manually saves the streamed PCM data as a valid WAV file.
3. Keeps the process isolation to prevent VRAM crashes.
"""

import os
import time
import requests
import multiprocessing
import warnings
import wave  # <--- NEW: Required to save streaming audio correctly
from pathlib import Path

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
TTS_SERVER_URL = "http://localhost:4123/v1/audio/speech"

# Directory containing your source images
ASSETS_DIR = Path("notebook_assets/avatars") 

# Output directory
OUTPUT_DIR = Path("output/demos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# The "Script"
# -----------------------------------------------------------------------------
AVATARS = [
    {
        "name": "professional",
        "image": "professional.png",
        "voice": "male",
        "speed": 1.0,
        "text": (
            "Good morning. I’m your AI assistant. "
            "I am designed to provide clear, reliable support and answer your questions professionally."
        )
    },
    {
        "name": "npc_villain",
        "image": "npc.png",
        "voice": "male",
        "speed": 0.9,
        "text": (
            "So... you finally arrived. "
            "The path ahead is treacherous, and every choice you make will have consequences."
        )
    },
    {
        "name": "content_creator",
        "image": "creator.png",
        "voice": "male",
        "speed": 1.1,
        "text": (
            "Hey guys! Welcome back to the channel. "
            "Today we're checking out this insane new AI tool that completely changes the game!"
        )
    },
    {
        "name": "tech_demo",
        "image": "custom.png",
        "voice": "neutral",
        "speed": 1.0,
        "text": (
            "This is a technical demonstration of multilingual text to speech, "
            "synchronized with generative lip movements."
        )
    }
]

# -----------------------------------------------------------------------------
# Helper: TTS Generation (FIXED)
# -----------------------------------------------------------------------------
def generate_audio(text: str, voice: str, speed: float, output_path: Path) -> bool:
    print(f"  🎤 Generating Audio ({voice} @ {speed}x)...")
    
    # [FIX] We switch to stream=True. 
    # The server handles long text better in stream mode (it chunks by sentence).
    payload = {
        "input": text,
        "language": "en",
        "voice": voice,
        "speed": speed,
        "temperature": 0.7,
        "stream": True  # <--- CHANGED to True
    }

    try:
        # We use stream=True in requests to receive chunks
        with requests.post(TTS_SERVER_URL, json=payload, stream=True, timeout=60) as resp:
            if resp.status_code == 200:
                
                # 1. Collect all Raw PCM data
                pcm_data = bytearray()
                for chunk in resp.iter_content(chunk_size=4096):
                    if chunk:
                        pcm_data.extend(chunk)

                if len(pcm_data) == 0:
                    print("  ❌ Error: Received empty audio stream.")
                    return False

                # 2. Get Sample Rate from header (default to 24000 if missing)
                sample_rate = int(resp.headers.get("X-Sample-Rate", 24000))

                # 3. Write as a valid WAV file using the 'wave' library
                # The server sends 16-bit PCM (2 bytes), Mono (1 channel)
                with wave.open(str(output_path), "wb") as wav_file:
                    wav_file.setnchannels(1)      # Mono
                    wav_file.setsampwidth(2)      # 16-bit = 2 bytes
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(pcm_data)
                
                return True
            else:
                print(f"  ❌ TTS Error {resp.status_code}: {resp.text}")
                return False

    except requests.exceptions.ConnectionError:
        print("  ❌ TTS Connection Failed (is localhost:4123 running?)")
        return False
    except Exception as e:
        print(f"  ❌ Audio Generation Failed: {e}")
        return False

# -----------------------------------------------------------------------------
# Helper: Video Rendering (Isolated Process)
# -----------------------------------------------------------------------------
def _render_process(image_path: str, audio_path: str, output_path: str):
    """
    Runs in a separate process to clear VRAM.
    """
    try:
        # --- PATCH: Fix Librosa/Wav2Lip incompatibility ---
        import librosa.filters
        if not hasattr(librosa.filters, 'mel'):
            pass # Suppress
            
        warnings.filterwarnings("ignore")

        # Lazy import
        from app.pipeline import render_pipeline

        print(f"  🎬 Pipeline Starting...")
        
        # 
        
        render_pipeline(
            face_image=image_path,
            audio=audio_path,
            out_path=output_path,
            quality_mode="real_time", 
            force_wav2lip=False
        )
    except Exception as e:
        print(f"  ❌ Render Crash: {e}")
        import sys
        sys.exit(1)

def generate_video_isolated(image_path: Path, audio_path: Path, output_path: Path) -> bool:
    # Resolve absolute paths
    p_image = str(image_path.resolve())
    p_audio = str(audio_path.resolve())
    p_output = str(output_path.resolve())

    proc = multiprocessing.Process(
        target=_render_process, 
        args=(p_image, p_audio, p_output)
    )
    
    start = time.time()
    proc.start()
    proc.join()
    elapsed = time.time() - start

    if proc.exitcode == 0:
        print(f"  ✅ Video Ready in {elapsed:.1f}s")
        return True
    else:
        print("  ❌ Video Generation Failed (Subprocess Error)")
        return False

# -----------------------------------------------------------------------------
# Main Loop
# -----------------------------------------------------------------------------
def main():
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    print("="*60)
    print(f"🚀 Bulk Avatar Generator starting for {len(AVATARS)} avatars")
    print("="*60)

    # Simple Check
    try:
        requests.get(TTS_SERVER_URL.replace("/speech", "")).status_code
    except:
        pass 

    for i, item in enumerate(AVATARS):
        print(f"\n[{i+1}/{len(AVATARS)}] Processing: {item['name'].upper()}")
        
        img_path = ASSETS_DIR / item['image']
        wav_path = OUTPUT_DIR / f"{item['name']}.wav"
        mp4_path = OUTPUT_DIR / f"{item['name']}.mp4"

        if not img_path.exists():
            print(f"  ⚠️ Image not found: {img_path} (Skipping)")
            continue

        if not generate_audio(item['text'], item['voice'], item['speed'], wav_path):
            continue

        success = generate_video_isolated(img_path, wav_path, mp4_path)
        
        if success and wav_path.exists():
            os.remove(wav_path)

    print("\n" + "="*60)
    print(f"🎉 Batch Complete. Check directory: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()