#!/usr/bin/env python3
"""
launcher.py - Desktop Application Launcher for Avatar Renderer MCP

This launcher creates a desktop application using Eel that integrates the
Vercel-compatible frontend with the real Avatar Renderer backend.

Features:
- Desktop application using Eel (Python + Web UI)
- Integrates with real avatar rendering pipeline
- Supports all avatar types from the frontend
- Real-time processing and video generation

Requirements:
    pip install eel python-dotenv

Usage:
    python launcher.py
    python launcher.py --port 8080
    python launcher.py --mode chrome
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import eel
except ImportError:
    print("❌ Error: Eel is not installed.")
    print("   Install it with: pip install eel python-dotenv")
    print("   Or: pip install -e \".[launcher]\"")
    sys.exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    print("❌ Error: python-dotenv is not installed.")
    print("   Install it with: pip install eel python-dotenv")
    print("   Or: pip install -e \".[launcher]\"")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Global variables
PROCESSING = False
CURRENT_TASK = None
TASK_PROGRESS = 0

# Configuration
OUTPUT_DIR = Path("generated_videos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Eel configuration
WEB_DIR = "web"  # We'll create a simple web interface


def setup_web_directory() -> None:
    """Set up the web directory for Eel."""
    web_path = Path(WEB_DIR)
    web_path.mkdir(exist_ok=True)

    # Create index.html
    index_html = web_path / "index.html"
    if not index_html.exists():
        index_html.write_text(create_html_interface())

    # Create styles
    css_file = web_path / "style.css"
    if not css_file.exists():
        css_file.write_text(create_css_styles())

    # Create JavaScript
    js_file = web_path / "main.js"
    if not js_file.exists():
        js_file.write_text(create_javascript())


def create_html_interface() -> str:
    """Create the HTML interface for the desktop app."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Avatar Renderer MCP - Desktop App</title>
    <link rel="stylesheet" href="style.css">
    <script type="text/javascript" src="/eel.js"></script>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎬 Avatar Renderer MCP</h1>
            <p class="subtitle">Transform static images into dynamic AI-powered avatars</p>
        </header>

        <main>
            <!-- Step 1: Choose Avatar -->
            <section class="card">
                <h2>Step 1: Choose Your Avatar</h2>
                <div class="avatar-grid">
                    <div class="avatar-option" data-avatar="professional">
                        <div class="avatar-icon">👔</div>
                        <h3>Professional</h3>
                        <p>Support • Sales • HR</p>
                    </div>
                    <div class="avatar-option" data-avatar="creator">
                        <div class="avatar-icon">🎨</div>
                        <h3>Creator</h3>
                        <p>Influencer • Ads</p>
                    </div>
                    <div class="avatar-option" data-avatar="educator">
                        <div class="avatar-icon">📚</div>
                        <h3>Educator</h3>
                        <p>Courses • Tutors</p>
                    </div>
                    <div class="avatar-option" data-avatar="game_npc">
                        <div class="avatar-icon">🎮</div>
                        <h3>Game NPC</h3>
                        <p>Dialogue • Lore</p>
                    </div>
                    <div class="avatar-option" data-avatar="brand">
                        <div class="avatar-icon">🏢</div>
                        <h3>Brand</h3>
                        <p>Retail • Events</p>
                    </div>
                    <div class="avatar-option" data-avatar="custom">
                        <div class="avatar-icon">✨</div>
                        <h3>Custom</h3>
                        <p>Bring your own</p>
                    </div>
                </div>

                <div class="upload-section">
                    <h3>Or Upload Your Own Avatar Image</h3>
                    <input type="file" id="avatar-upload" accept="image/png,image/jpeg,image/jpg">
                    <label for="avatar-upload" class="upload-btn">
                        📁 Choose Image (PNG/JPG)
                    </label>
                    <div id="upload-preview"></div>
                </div>
            </section>

            <!-- Step 2: Add Script/Audio -->
            <section class="card">
                <h2>Step 2: Add Script or Audio</h2>
                <div class="input-group">
                    <label for="script-text">Enter Script Text:</label>
                    <textarea
                        id="script-text"
                        rows="4"
                        placeholder="Hello! I'm your AI avatar, ready to bring your content to life..."
                    ></textarea>
                </div>

                <div class="upload-section">
                    <h3>Or Upload Audio File</h3>
                    <input type="file" id="audio-upload" accept="audio/wav,audio/mp3">
                    <label for="audio-upload" class="upload-btn">
                        🎵 Choose Audio (WAV/MP3)
                    </label>
                    <div id="audio-preview"></div>
                </div>

                <div class="quality-selector">
                    <label for="quality-mode">Quality Mode:</label>
                    <select id="quality-mode">
                        <option value="auto">Auto</option>
                        <option value="real_time">Real Time (Faster)</option>
                        <option value="high_quality">High Quality (Slower)</option>
                    </select>
                </div>
            </section>

            <!-- Step 3: Generate -->
            <section class="card">
                <h2>Step 3: Generate Your Avatar</h2>
                <button id="generate-btn" class="generate-btn" onclick="generateAvatar()">
                    🎬 Bring to Life
                </button>
                <button id="reset-btn" class="reset-btn" onclick="resetForm()">
                    🔄 Reset
                </button>

                <div id="status-message"></div>
                <div id="progress-container" style="display: none;">
                    <div class="progress-bar">
                        <div id="progress-fill"></div>
                    </div>
                    <p id="progress-text">Processing...</p>
                </div>
            </section>

            <!-- Results -->
            <section class="card" id="results-section" style="display: none;">
                <h2>Your AI Avatar</h2>
                <div id="video-container">
                    <video id="result-video" controls></video>
                </div>
                <div class="result-actions">
                    <button onclick="downloadVideo()" class="download-btn">
                        💾 Download MP4
                    </button>
                    <button onclick="createAnother()" class="secondary-btn">
                        ➕ Create Another
                    </button>
                </div>
            </section>
        </main>

        <footer>
            <p>© 2025 Avatar Renderer MCP. All rights reserved.</p>
            <p>Powered by FOMM + Diff2Lip + Wav2Lip</p>
        </footer>
    </div>

    <script src="main.js"></script>
</body>
</html>
"""


def create_css_styles() -> str:
    """Create CSS styles for the desktop app."""
    return """* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: #333;
    min-height: 100vh;
    padding: 20px;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
}

header {
    text-align: center;
    color: white;
    margin-bottom: 40px;
}

header h1 {
    font-size: 3em;
    margin-bottom: 10px;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
}

.subtitle {
    font-size: 1.2em;
    opacity: 0.9;
}

.card {
    background: white;
    border-radius: 12px;
    padding: 30px;
    margin-bottom: 20px;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
}

.card h2 {
    color: #667eea;
    margin-bottom: 20px;
    font-size: 1.8em;
}

.avatar-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 15px;
    margin-bottom: 30px;
}

.avatar-option {
    border: 2px solid #e0e0e0;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
}

.avatar-option:hover {
    border-color: #667eea;
    transform: translateY(-5px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
}

.avatar-option.selected {
    border-color: #667eea;
    background: #f0f4ff;
}

.avatar-icon {
    font-size: 3em;
    margin-bottom: 10px;
}

.avatar-option h3 {
    font-size: 1.1em;
    margin-bottom: 5px;
    color: #333;
}

.avatar-option p {
    font-size: 0.9em;
    color: #666;
}

.upload-section {
    margin-top: 30px;
    padding-top: 30px;
    border-top: 1px solid #e0e0e0;
}

.upload-section h3 {
    font-size: 1.2em;
    margin-bottom: 15px;
    color: #555;
}

input[type="file"] {
    display: none;
}

.upload-btn {
    display: inline-block;
    padding: 12px 24px;
    background: #667eea;
    color: white;
    border-radius: 6px;
    cursor: pointer;
    transition: background 0.3s ease;
}

.upload-btn:hover {
    background: #5568d3;
}

.input-group {
    margin-bottom: 20px;
}

.input-group label {
    display: block;
    margin-bottom: 8px;
    font-weight: 600;
    color: #555;
}

textarea {
    width: 100%;
    padding: 12px;
    border: 2px solid #e0e0e0;
    border-radius: 6px;
    font-size: 1em;
    font-family: inherit;
    resize: vertical;
}

textarea:focus {
    outline: none;
    border-color: #667eea;
}

.quality-selector {
    margin-top: 20px;
}

.quality-selector label {
    display: block;
    margin-bottom: 8px;
    font-weight: 600;
    color: #555;
}

select {
    padding: 10px;
    border: 2px solid #e0e0e0;
    border-radius: 6px;
    font-size: 1em;
    min-width: 200px;
}

select:focus {
    outline: none;
    border-color: #667eea;
}

.generate-btn, .reset-btn, .download-btn, .secondary-btn {
    padding: 15px 40px;
    font-size: 1.1em;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    margin-right: 10px;
    margin-bottom: 10px;
    transition: all 0.3s ease;
}

.generate-btn {
    background: #667eea;
    color: white;
}

.generate-btn:hover {
    background: #5568d3;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.generate-btn:disabled {
    background: #ccc;
    cursor: not-allowed;
    transform: none;
}

.reset-btn, .secondary-btn {
    background: #f0f0f0;
    color: #333;
}

.reset-btn:hover, .secondary-btn:hover {
    background: #e0e0e0;
}

.download-btn {
    background: #4caf50;
    color: white;
}

.download-btn:hover {
    background: #45a049;
}

#status-message {
    margin-top: 20px;
    padding: 15px;
    border-radius: 6px;
    display: none;
}

#status-message.success {
    background: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
    display: block;
}

#status-message.error {
    background: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c6cb;
    display: block;
}

#status-message.info {
    background: #d1ecf1;
    color: #0c5460;
    border: 1px solid #bee5eb;
    display: block;
}

.progress-bar {
    width: 100%;
    height: 30px;
    background: #e0e0e0;
    border-radius: 15px;
    overflow: hidden;
    margin: 20px 0 10px 0;
}

#progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    width: 0%;
    transition: width 0.3s ease;
}

#progress-text {
    text-align: center;
    color: #666;
    font-weight: 600;
}

#video-container {
    margin: 20px 0;
    text-align: center;
}

#result-video {
    max-width: 100%;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

.result-actions {
    text-align: center;
    margin-top: 20px;
}

footer {
    text-align: center;
    color: white;
    margin-top: 40px;
    opacity: 0.9;
}

footer p {
    margin: 5px 0;
}

#upload-preview, #audio-preview {
    margin-top: 15px;
    padding: 10px;
    background: #f9f9f9;
    border-radius: 6px;
    display: none;
}

#upload-preview.show, #audio-preview.show {
    display: block;
}
"""


def create_javascript() -> str:
    """Create JavaScript for the desktop app."""
    return """let selectedAvatar = null;
let uploadedImage = null;
let uploadedAudio = null;
let currentVideoPath = null;

// Avatar selection
document.querySelectorAll('.avatar-option').forEach(option => {
    option.addEventListener('click', function() {
        document.querySelectorAll('.avatar-option').forEach(opt => {
            opt.classList.remove('selected');
        });
        this.classList.add('selected');
        selectedAvatar = this.dataset.avatar;
        uploadedImage = null;
        document.getElementById('upload-preview').classList.remove('show');
    });
});

// Image upload
document.getElementById('avatar-upload').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(event) {
            uploadedImage = event.target.result;
            selectedAvatar = null;
            document.querySelectorAll('.avatar-option').forEach(opt => {
                opt.classList.remove('selected');
            });
            const preview = document.getElementById('upload-preview');
            preview.innerHTML = `<img src="${event.target.result}" style="max-width: 200px; border-radius: 6px;">
                                <p>✅ Image uploaded: ${file.name}</p>`;
            preview.classList.add('show');
        };
        reader.readAsDataURL(file);
    }
});

// Audio upload
document.getElementById('audio-upload').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(event) {
            uploadedAudio = event.target.result;
            const preview = document.getElementById('audio-preview');
            preview.innerHTML = `<audio controls src="${event.target.result}"></audio>
                                <p>✅ Audio uploaded: ${file.name}</p>`;
            preview.classList.add('show');
        };
        reader.readAsDataURL(file);
    }
});

async function generateAvatar() {
    const scriptText = document.getElementById('script-text').value;
    const qualityMode = document.getElementById('quality-mode').value;

    // Validation
    if (!selectedAvatar && !uploadedImage) {
        showStatus('Please select or upload an avatar image', 'error');
        return;
    }

    if (!scriptText && !uploadedAudio) {
        showStatus('Please enter a script or upload an audio file', 'error');
        return;
    }

    // Disable button
    const generateBtn = document.getElementById('generate-btn');
    generateBtn.disabled = true;
    generateBtn.textContent = '⏳ Processing...';

    // Show progress
    document.getElementById('progress-container').style.display = 'block';
    showStatus('Starting avatar generation...', 'info');

    try {
        // Call Python backend
        const result = await eel.generate_avatar_video(
            selectedAvatar,
            uploadedImage,
            scriptText,
            uploadedAudio,
            qualityMode
        )();

        if (result.success) {
            showStatus('✅ Avatar generated successfully!', 'success');
            displayResult(result.video_path);
        } else {
            showStatus(`❌ Error: ${result.error}`, 'error');
        }
    } catch (error) {
        showStatus(`❌ Error: ${error}`, 'error');
    } finally {
        generateBtn.disabled = false;
        generateBtn.textContent = '🎬 Bring to Life';
        document.getElementById('progress-container').style.display = 'none';
    }
}

function showStatus(message, type) {
    const statusDiv = document.getElementById('status-message');
    statusDiv.textContent = message;
    statusDiv.className = type;
}

function displayResult(videoPath) {
    currentVideoPath = videoPath;
    const video = document.getElementById('result-video');
    video.src = `file://${videoPath}`;
    document.getElementById('results-section').style.display = 'block';
    video.scrollIntoView({ behavior: 'smooth' });
}

async function downloadVideo() {
    if (currentVideoPath) {
        await eel.open_file_location(currentVideoPath)();
    }
}

function createAnother() {
    resetForm();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function resetForm() {
    selectedAvatar = null;
    uploadedImage = null;
    uploadedAudio = null;
    currentVideoPath = null;

    document.querySelectorAll('.avatar-option').forEach(opt => {
        opt.classList.remove('selected');
    });

    document.getElementById('script-text').value = '';
    document.getElementById('avatar-upload').value = '';
    document.getElementById('audio-upload').value = '';
    document.getElementById('quality-mode').value = 'auto';
    document.getElementById('upload-preview').classList.remove('show');
    document.getElementById('audio-preview').classList.remove('show');
    document.getElementById('status-message').style.display = 'none';
    document.getElementById('results-section').style.display = 'none';
}

// Update progress bar
eel.expose(updateProgress);
function updateProgress(progress, message) {
    document.getElementById('progress-fill').style.width = `${progress}%`;
    document.getElementById('progress-text').textContent = message;
}
"""


# Eel exposed functions
@eel.expose
def generate_avatar_video(
    avatar_type: Optional[str],
    uploaded_image: Optional[str],
    script_text: str,
    uploaded_audio: Optional[str],
    quality_mode: str,
) -> Dict[str, Any]:
    """Generate avatar video using the pipeline."""
    try:
        global PROCESSING, TASK_PROGRESS
        PROCESSING = True
        TASK_PROGRESS = 0

        eel.updateProgress(10, "Preparing assets...")

        # Determine image path
        if uploaded_image:
            # Decode base64 image
            image_data = uploaded_image.split(",")[1]
            image_bytes = base64.b64decode(image_data)
            image_path = tempfile.NamedTemporaryFile(
                delete=False, suffix=".png"
            ).name
            with open(image_path, "wb") as f:
                f.write(image_bytes)
        else:
            # Use preset avatar
            from assets import AVATAR_CONFIGS

            if avatar_type in AVATAR_CONFIGS:
                image_path = AVATAR_CONFIGS[avatar_type]["image"]
            else:
                return {"success": False, "error": "Invalid avatar type"}

        eel.updateProgress(30, "Processing audio...")

        # Determine audio path
        if uploaded_audio:
            # Decode base64 audio
            audio_data = uploaded_audio.split(",")[1]
            audio_bytes = base64.b64decode(audio_data)
            audio_path = tempfile.NamedTemporaryFile(
                delete=False, suffix=".wav"
            ).name
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)
        else:
            # Use placeholder audio (TODO: integrate TTS)
            audio_path = "tests/assets/hello.wav"

        eel.updateProgress(50, "Generating avatar video...")

        # Generate output path
        timestamp = int(time.time())
        output_path = OUTPUT_DIR / f"avatar_{timestamp}.mp4"

        # Run the pipeline
        from app.pipeline import render_pipeline

        result = render_pipeline(
            face_image=str(Path(image_path).resolve()),
            audio=str(Path(audio_path).resolve()),
            out_path=str(output_path.resolve()),
            quality_mode=quality_mode,
        )

        eel.updateProgress(100, "Complete!")

        PROCESSING = False
        return {"success": True, "video_path": str(result)}

    except Exception as e:
        PROCESSING = False
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e)}


@eel.expose
def open_file_location(file_path: str) -> None:
    """Open file location in system file explorer."""
    import platform
    import subprocess

    file_path = Path(file_path)
    folder = file_path.parent

    system = platform.system()
    if system == "Windows":
        subprocess.run(["explorer", str(folder)])
    elif system == "Darwin":  # macOS
        subprocess.run(["open", str(folder)])
    else:  # Linux
        subprocess.run(["xdg-open", str(folder)])


def build_argparser() -> argparse.ArgumentParser:
    """Build argument parser."""
    p = argparse.ArgumentParser(
        description="Avatar Renderer MCP Desktop Launcher"
    )
    p.add_argument(
        "--port", type=int, default=8050, help="Port for the web server"
    )
    p.add_argument(
        "--mode",
        choices=["chrome", "chrome-app", "edge", "firefox", "default"],
        default="chrome",
        help="Browser mode for Eel",
    )
    p.add_argument(
        "--size",
        nargs=2,
        type=int,
        default=[1200, 900],
        help="Window size (width height)",
    )
    return p


def main() -> int:
    """Main entry point."""
    print("=" * 70)
    print("🎬 Avatar Renderer MCP - Desktop Launcher")
    print("=" * 70)
    print()

    args = build_argparser().parse_args()

    print("🔧 Setting up web directory...")
    setup_web_directory()

    print("🚀 Initializing Eel...")
    eel.init(WEB_DIR)

    print(f"🌐 Starting application on port {args.port}...")
    print(f"📐 Window size: {args.size[0]}x{args.size[1]}")
    print(f"🌍 Browser mode: {args.mode}")
    print()
    print("✅ Application ready!")
    print("💡 The desktop app will open in a new window.")
    print()

    try:
        eel.start(
            "index.html",
            mode=args.mode,
            port=args.port,
            size=args.size,
            block=True,
        )
    except (SystemExit, KeyboardInterrupt):
        print("\n\n👋 Application closed.")
        return 0
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
