# Avatar Renderer - Modern Web Launcher

A modern, user-friendly web-based UI for the Avatar Renderer MCP, built with Eel (Python + HTML/CSS/JavaScript).

## Overview

The launcher provides a beautiful, modern alternative to the tkinter GUI, featuring:

- 🎨 **Modern Dark Theme** - Sleek, professional interface with gradient accents
- 🌐 **Web-Based** - Runs in your browser with full Python backend access
- 📱 **Responsive Design** - Works on different screen sizes
- ⚡ **Real-time Updates** - Live status indicators and progress tracking
- 🎯 **User-Friendly** - Intuitive step-by-step workflow

## Quick Start

### Using Make (Recommended)

```bash
# Launch the modern web UI (automatically starts backend)
make launch
```

This will:
1. Install launcher dependencies (eel, python-dotenv)
2. Start the backend API server
3. Open the launcher in your default browser
4. Automatically clean up when you close the app

### Manual Launch

```bash
# Install dependencies
uv pip install -e ".[launcher]"

# Start backend (in one terminal)
make server

# Launch UI (in another terminal)
python -m launcher
```

## Features

### Step 1: Text & Voice
- **Multilingual Support** - 20+ languages including English, Spanish, French, German, Arabic, Chinese, etc.
- **Voice Profiles** - Choose from 6 voice personalities (Sophia, Emma, Luna, Marcus, Ethan, Neutral)
- **Sample Texts** - Quick-load common phrases in each language
- **Audio Preview** - Generate and preview audio before creating video

### Step 2: Avatar & Settings
- **Avatar Selection** - Choose any PNG/JPG/JPEG image as your avatar
- **Quality Modes**:
  - 🔄 **Auto** - Automatically selects best quality for your hardware
  - ⚡ **Real-time** - Fast rendering for quick previews
  - ✨ **High Quality** - Best quality (slower, recommended for final output)
- **Output Directory** - View where your videos will be saved

### Step 3: Generate Video
- **Live Logs** - Real-time progress updates and status messages
- **Progress Tracking** - Visual feedback during video generation
- **Output Management** - Quick access to generated videos

## Technical Details

### Architecture

```
launcher/
├── __init__.py          # Package initialization
├── __main__.py          # Entry point
├── app.py               # Python backend (Eel server)
├── README.md            # This file
└── web/                 # Frontend files
    ├── index.html       # Main UI structure
    ├── css/
    │   └── styles.css   # Modern dark theme styling
    └── js/
        └── app.js       # Frontend logic & Eel communication
```

### Communication Flow

```
Browser (JavaScript) ←→ Eel ←→ Python Backend ←→ FastAPI Server
```

1. **Frontend (JS)** - User interactions, UI updates
2. **Eel** - Bridges JavaScript ↔ Python communication
3. **Backend (Python)** - Business logic, API calls
4. **FastAPI Server** - ML pipeline, audio/video generation

### Dependencies

Defined in `pyproject.toml`:
```toml
[project.optional-dependencies]
launcher = [
    "eel>=0.16.0,<1.0.0",
    "python-dotenv>=1.0.0,<2.0.0",
]
```

## Configuration

The launcher reads configuration from environment variables:

```bash
# API endpoint (defaults to http://localhost:8000)
export API_URL=http://localhost:8000

# Model directory
export MODEL_ROOT=/path/to/models

# External dependencies
export EXT_DEPS_DIR=/path/to/external_deps
```

## Comparison with tkinter GUI

| Feature | Launcher (Eel) | GUI (tkinter) |
|---------|----------------|---------------|
| **UI Style** | Modern web-based | Classic desktop |
| **Browser** | Yes (Chrome/Edge) | No |
| **Theming** | Dark theme, gradients | Native OS theme |
| **Responsive** | Yes | Fixed size |
| **Platform** | Cross-platform | Cross-platform |
| **Dependencies** | Eel | tkinter (built-in) |
| **Best For** | Modern UX | Traditional desktop |

## Troubleshooting

### Launcher won't start

**Issue**: `ModuleNotFoundError: No module named 'eel'`

**Solution**:
```bash
uv pip install -e ".[launcher]"
```

### Backend API offline

**Issue**: Red "API Offline" badge

**Solution**:
```bash
# Check if backend is running
curl http://localhost:8000/health/live

# Start backend if not running
make server
```

### Browser doesn't open

**Issue**: Eel starts but no browser window

**Solution**:
- Manually open: http://localhost:8080
- Or set different mode in `launcher/app.py`:
  ```python
  eel.start('index.html', mode='default')  # Use default browser
  ```

### Port already in use

**Issue**: `Address already in use: 8080`

**Solution**:
```bash
# Kill process on port 8080
lsof -ti:8080 | xargs kill -9

# Or change port in launcher/app.py
eel.start('index.html', port=8081)
```

## Development

### Modifying the UI

1. **HTML** - Edit `launcher/web/index.html`
2. **Styles** - Edit `launcher/web/css/styles.css`
3. **JavaScript** - Edit `launcher/web/js/app.js`
4. **Backend** - Edit `launcher/app.py`

Changes to HTML/CSS/JS are reflected on browser refresh. Python changes require restarting the launcher.

### Adding New Features

1. **Backend**: Add exposed function in `app.py`:
   ```python
   @eel.expose
   def my_new_function(param):
       return {"status": "success", "data": param}
   ```

2. **Frontend**: Call from JavaScript:
   ```javascript
   async function callMyFunction() {
       const result = await eel.my_new_function("value")();
       console.log(result);
   }
   ```

## Performance

- **Startup Time**: ~2-3 seconds (includes backend health check)
- **Memory Usage**: ~100-200MB (excluding ML models)
- **Browser**: Chrome/Chromium recommended for best performance

## Future Enhancements

- [ ] Drag & drop for avatar images
- [ ] Video preview player
- [ ] History of generated videos
- [ ] Batch processing
- [ ] Export settings/presets
- [ ] Dark/Light theme toggle

## License

Apache-2.0 - See main project LICENSE file

## Credits

Built with:
- [Eel](https://github.com/python-eel/Eel) - Python web app framework
- Modern CSS3 with gradient themes
- Vanilla JavaScript (no frameworks)

---

**Need Help?** Check the main [README.md](../README.md) or open an issue on GitHub.
