# Avatar Renderer MCP - Launcher & Assets Documentation

This document describes the new launcher application and assets generation tools for the Avatar Renderer MCP project.

## Overview

Two new Python scripts have been added to enhance the Avatar Renderer MCP functionality:

1. **`assets.py`** - Generate all demo videos for the frontend
2. **`launcher.py`** - Desktop application launcher using Eel

---

## 📦 Installation

### Install Launcher Dependencies

```bash
# Using pip
pip install eel python-dotenv

# Or using the optional dependency group
pip install -e ".[launcher]"

# Or install everything
pip install -e ".[all]"
```

---

## 🎬 Assets Generator (`assets.py`)

### Purpose

Generate demo videos for all avatar personas displayed in the Vercel frontend:

- **Professional** - Support, Sales, HR
- **Creator** - Influencer, Ads
- **Educator** - Courses, Tutors
- **Game NPC** - Dialogue, Lore
- **Brand** - Retail, Events
- **Custom** - Bring your own

### Usage

```bash
# Generate all demo videos (saves to ./demo_videos/)
python assets.py

# Generate specific avatars only
python assets.py --avatar professional creator educator

# Use high quality mode
python assets.py --quality high_quality

# Custom output directory (e.g., for Vercel frontend)
python assets.py --output-dir ./frontend/public/videos

# Generate with TTS (requires TTS setup)
python assets.py --use-tts
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--avatar NAMES` | Generate specific avatar(s) only | All avatars |
| `--output-dir DIR` | Output directory for videos | `./demo_videos` |
| `--quality MODE` | Quality mode: `auto`, `real_time`, `high_quality` | `auto` |
| `--use-tts` | Generate audio from scripts using TTS | `False` |

### Avatar Configurations

The script includes predefined configurations for each avatar type:

```python
AVATAR_CONFIGS = {
    "professional": {
        "image": "external_deps/SadTalker/examples/source_image/art_1.png",
        "script": "Hello! I'm your professional assistant...",
    },
    # ... more avatars
}
```

### Output

Videos are generated in MP4 format and saved to the output directory:

```
demo_videos/
├── professional_demo.mp4
├── creator_demo.mp4
├── educator_demo.mp4
├── game_npc_demo.mp4
├── brand_demo.mp4
└── custom_demo.mp4
```

### Deploying to Vercel Frontend

1. Generate videos for the frontend:
   ```bash
   python assets.py --output-dir ./frontend/public/videos --quality high_quality
   ```

2. Copy videos to frontend public directory:
   ```bash
   cp demo_videos/*.mp4 frontend/public/videos/
   ```

3. Update your frontend to reference these videos:
   ```javascript
   const demoVideos = {
     professional: '/videos/professional_demo.mp4',
     creator: '/videos/creator_demo.mp4',
     // ...
   };
   ```

4. Deploy to Vercel:
   ```bash
   cd frontend
   vercel deploy
   ```

---

## 🖥️ Desktop Launcher (`launcher.py`)

### Purpose

Create a desktop application that:
- Uses the Vercel-compatible frontend adapted for desktop
- Connects to the real Avatar Renderer backend
- Provides a native application experience
- Supports real-time video generation

### Features

- **Desktop UI**: Web-based interface using Eel (Python + HTML/CSS/JS)
- **Real Backend**: Uses the actual `app.pipeline.render_pipeline`
- **Avatar Selection**: Choose from preset avatars or upload custom images
- **Audio Input**: Enter text scripts or upload audio files
- **Quality Modes**: Auto, Real-time, or High Quality
- **Video Export**: Download generated MP4 files

### Usage

```bash
# Launch the desktop application (default settings)
python launcher.py

# Custom port
python launcher.py --port 8080

# Use different browser mode
python launcher.py --mode chrome-app

# Custom window size
python launcher.py --size 1400 1000
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--port PORT` | Port for the web server | `8050` |
| `--mode MODE` | Browser mode: `chrome`, `chrome-app`, `edge`, `firefox`, `default` | `chrome` |
| `--size W H` | Window size (width height) | `1200 900` |

### Application Structure

The launcher creates a `web/` directory with:

```
web/
├── index.html    # Main UI interface
├── style.css     # Styling
└── main.js       # Frontend JavaScript
```

Generated videos are saved to:

```
generated_videos/
└── avatar_<timestamp>.mp4
```

### How It Works

1. **Frontend**: HTML/CSS/JavaScript interface presented via Eel
2. **Backend**: Python functions exposed via `@eel.expose` decorator
3. **Communication**: JavaScript calls Python functions seamlessly
4. **Pipeline**: Uses `app.pipeline.render_pipeline` for video generation

### Key Functions

#### Python (Backend)

```python
@eel.expose
def generate_avatar_video(avatar_type, uploaded_image, script_text, uploaded_audio, quality_mode):
    """Generate avatar video using the pipeline."""
    # ... uses app.pipeline.render_pipeline
    return {"success": True, "video_path": result}

@eel.expose
def open_file_location(file_path):
    """Open file location in system file explorer."""
    # ... opens folder containing generated video
```

#### JavaScript (Frontend)

```javascript
async function generateAvatar() {
    // Call Python backend
    const result = await eel.generate_avatar_video(
        selectedAvatar,
        uploadedImage,
        scriptText,
        uploadedAudio,
        qualityMode
    )();

    // Display result
    displayResult(result.video_path);
}
```

---

## 🔧 Configuration

### Environment Variables

Both scripts respect standard Avatar Renderer MCP environment variables:

```bash
# Model directory
export MODEL_ROOT=/path/to/models

# External dependencies directory
export EXT_DEPS_DIR=/path/to/external_deps
```

### Dependencies

The launcher requires additional packages defined in `pyproject.toml`:

```toml
[project.optional-dependencies]
launcher = [
    "eel>=0.16.0,<1.0.0",
    "python-dotenv>=1.0.0,<2.0.0",
]
```

---

## 🚀 Quick Start Guide

### 1. Generate Demo Videos

```bash
# Install dependencies
pip install -e ".[launcher]"

# Generate all demo videos
python assets.py --quality high_quality --output-dir ./demo_videos

# Check results
ls -lh demo_videos/
```

### 2. Launch Desktop Application

```bash
# Launch the app
python launcher.py

# The app will open in a Chrome window
# Select an avatar, enter text, and click "Bring to Life"
```

### 3. Deploy to Vercel

```bash
# Generate videos for frontend
python assets.py --output-dir ./frontend/public/videos

# Deploy frontend
cd frontend
vercel deploy
```

---

## 🎯 Use Cases

### For Development

- **Testing**: Generate test videos with different avatars
- **Demo Creation**: Create demo content for presentations
- **Frontend Development**: Generate sample videos for frontend testing

### For Production

- **Content Generation**: Batch-generate videos for multiple personas
- **Desktop Application**: Provide a standalone desktop tool for users
- **Web Deployment**: Serve pre-generated demos on Vercel

---

## 📝 Troubleshooting

### Assets Generator

**Problem**: Image files not found

```bash
# Solution: Check that SadTalker examples exist
ls external_deps/SadTalker/examples/source_image/
```

**Problem**: Models not downloaded

```bash
# Solution: Download models
make download-models
```

### Desktop Launcher

**Problem**: Eel not installed

```bash
# Solution: Install launcher dependencies
pip install eel python-dotenv
```

**Problem**: Chrome not found

```bash
# Solution: Use different browser mode
python launcher.py --mode firefox
```

**Problem**: Port already in use

```bash
# Solution: Use different port
python launcher.py --port 8051
```

---

## 🔄 Workflow Examples

### Generate and Deploy Full Frontend

```bash
# 1. Generate all demo videos
python assets.py --quality high_quality --output-dir ./frontend/public/videos

# 2. Verify videos
ls -lh frontend/public/videos/

# 3. Test frontend locally
cd frontend
npm run dev

# 4. Deploy to Vercel
vercel deploy --prod
```

### Desktop Application Development

```bash
# 1. Install dependencies
pip install -e ".[launcher]"

# 2. Launch app in development mode
python launcher.py --mode chrome --size 1400 1000

# 3. Test video generation with different avatars

# 4. Check generated videos
ls -lh generated_videos/
```

---

## 🎨 Customization

### Adding New Avatar Presets

Edit `assets.py` and add to `AVATAR_CONFIGS`:

```python
AVATAR_CONFIGS = {
    # ... existing avatars
    "my_custom_avatar": {
        "description": "My custom avatar description",
        "image": "path/to/image.png",
        "script": "Custom script text...",
        "output": "my_custom_demo.mp4",
    },
}
```

### Customizing the Launcher UI

Edit the HTML/CSS/JavaScript in `launcher.py`:

- `create_html_interface()` - Modify HTML structure
- `create_css_styles()` - Change styling
- `create_javascript()` - Add new features

---

## 📚 Additional Resources

- **Main README**: [README.md](README.md)
- **Pipeline Documentation**: See `app/pipeline.py`
- **Eel Documentation**: https://github.com/ChrisKnott/Eel
- **Vercel Deployment**: https://vercel.com/docs

---

## 🤝 Contributing

To contribute improvements to the launcher or assets generator:

1. Test your changes thoroughly
2. Update this documentation
3. Submit a pull request

---

## 📄 License

Apache-2.0 License - See [LICENSE](LICENSE) file for details

---

## 👤 Author

**Ruslan Magana Vsevolodovna**

- Website: https://ruslanmv.com
- Email: contact@ruslanmv.com
- GitHub: https://github.com/ruslanmv/avatar-renderer-mcp

---

**Last Updated**: 2025-12-27
