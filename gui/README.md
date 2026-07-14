# Avatar Renderer GUI (Deprecated)

> **Deprecated:** This legacy Tk desktop GUI is no longer the supported product UI.
> Use the modern web launcher instead:
>
> ```bash
> make launch
> ```
>
> The Tk GUI is kept only for low-level debugging and may be removed in a future release.

## Legacy documentation

A graphical user interface for generating talking avatar videos from text input.

## Features

- **Multilingual Support**: 23 languages supported (English, Spanish, Italian, French, German, Russian, etc.)
- **Multiple Voice Profiles**: Female, Male, and Neutral voices with different characteristics
- **Avatar Customization**: Use any portrait image (PNG/JPG)
- **Quality Modes**: Auto, Real-time (fast), or High Quality (best)
- **Complete Pipeline**: Text → Audio → Video in one click

## Requirements

```bash
pip install requests
```

## Usage

### 1. Start the API Server

First, start the FastAPI server:

```bash
# From project root
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

### 2. Run the legacy GUI for debugging only

For normal use, run:

```bash
make launch
```

If you are debugging the deprecated Tk module directly, run:

```bash
python -m gui
```

## Quick Start

1. **Select Language**: Choose from 23 supported languages
2. **Choose Voice**: Pick a voice profile (female, male, or neutral)
3. **Enter Text**: Type or select sample text
4. **Select Avatar**: Choose a portrait image (default: tests/assets/alice.png)
5. **Set Quality**: Choose rendering quality (Auto/Real-time/High Quality)
6. **Generate**: Click "Generate Video" and wait for processing
7. **View Result**: Open the output folder to see your video

## Pipeline Steps

The GUI executes the following pipeline:

1. **Text-to-Speech (TTS)**: Converts your text to audio using the Chatterbox TTS server
2. **Avatar Rendering**: Generates lip-synced video using the audio and avatar image
3. **Enhancement**: Applies face enhancement and quality improvements
4. **Output**: Saves the final video to the output directory

## Configuration

You can customize the following via environment variables:

- `API_URL`: API server URL (default: http://localhost:8000)

## Supported Languages

🇬🇧 English, 🇪🇸 Spanish, 🇮🇹 Italian, 🇫🇷 French, 🇩🇪 German, 🇷🇺 Russian, 🇵🇹 Portuguese, 🇵🇱 Polish, 🇳🇱 Dutch, 🇸🇪 Swedish, 🇳🇴 Norwegian, 🇩🇰 Danish, 🇫🇮 Finnish, 🇬🇷 Greek, 🇹🇷 Turkish, 🇸🇦 Arabic, 🇮🇱 Hebrew, 🇮🇳 Hindi, 🇯🇵 Japanese, 🇰🇷 Korean, 🇨🇳 Chinese, 🇲🇾 Malay, 🇹🇿 Swahili

## Voice Profiles

- **Sophia** (Female): Friendly Assistant - Warm and professional
- **Emma** (Female): Calm Professional - Composed and clear
- **Luna** (Female): Energetic & Warm - Lively and expressive
- **Marcus** (Male): Professional - Clear and authoritative
- **Ethan** (Male): Warm Baritone - Deep and reassuring
- **Neutral**: Default synthesized voice

## Troubleshooting

### API Connection Failed

- Ensure the FastAPI server is running: `uvicorn app.api:app --port 8000`
- Check the API URL in the status bar
- Verify the server is accessible: `curl http://localhost:8000/health/live`

### TTS Service Unavailable

- Check if the Chatterbox TTS server is running
- Verify TTS configuration in `app/settings.py`
- Test the TTS endpoint: `curl http://localhost:8000/health/tts`

### Video Generation Failed

- Ensure all required models are downloaded: `make download-models`
- Check GPU availability for high-quality mode
- Review the generation log in the GUI for specific errors

## Output Files

Generated files are saved in the `output/` directory (configurable):

- `audio_YYYYMMDD_HHMMSS.wav`: Generated audio file
- `avatar_YYYYMMDD_HHMMSS.mp4`: Final video file

## License

Apache-2.0
