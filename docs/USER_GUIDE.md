# User Guide - Avatar Renderer MCP

A simple step-by-step guide for creating AI-powered talking avatar videos.

Avatar Renderer MCP offers **two interfaces** to create your videos. Choose whichever works best for you:

- **Desktop App** (AvatarStudio) - a native desktop window for local use
- **Web App** (Next.js) - a modern browser-based interface

---

## Option A: Desktop App (AvatarStudio)

### Starting the App

```bash
python launcher/app.py
```

A desktop window will open with a sidebar on the left and three panels on the right.

---

### Step 1: Write Your Script and Generate Audio

1. **Choose a language** from the **Language** dropdown (default: English).
2. **Choose a voice** from the **Voice Persona** dropdown (e.g., Sophia, Marcus).
3. **Type your script** in the text box, or click one of the **sample buttons** (Greeting, Intro, etc.) to load a preset.
4. Click **Generate Audio**.
5. An audio player will appear. Press play to preview your audio and make sure it sounds right.

> **Tip:** You must generate audio before you can render a video.

---

### Step 2: Configure Your Avatar

1. **Select an avatar image:**
   - Click **Browse** next to "Source Image Path".
   - Pick any face photo (PNG or JPG). A clear, front-facing portrait works best.
   - The image will appear in the **Avatar Preview** on the sidebar.

2. **Choose render quality:**
   - **High Quality** - Uses Diff2Lip for the most realistic lip-sync (needs GPU).
   - **Real-time** - Uses Wav2Lip for faster processing.
   - **Auto** - Automatically picks the best option for your hardware.

3. **Select enhancements** (optional but recommended):

   Click the colored chips to toggle each enhancement on or off. The counter shows how many you have selected.

   | Enhancement | What it does | Needs extra models? |
   |---|---|---|
   | **Emotion** | Detects emotion from your audio and moves the face accordingly | No |
   | **Eye Gaze** | Adds natural blinking and subtle eye movements | No |
   | **Gestures** | Adds breathing and upper-body motion | No |
   | **MuseTalk** | Higher FPS lip-sync alternative | Yes |
   | **LivePortrait** | Expression-controllable face motion | Yes |
   | **LatentSync** | Latent-space lip-sync | Yes |
   | **Hallo3** | Cinematic-quality rendering | Yes |
   | **CosyVoice** | Emotionally expressive voice synthesis | Yes |
   | **Viseme** | Phoneme-accurate lip shapes | Yes |
   | **3D Gauss** | 3D Gaussian Splatting avatar | Yes |

   By default, **Emotion**, **Eye Gaze**, and **Gestures** are enabled. These three work out of the box without downloading additional models.

   > **Tip:** The sidebar shows "X/10 ready" to tell you how many enhancement modules are available on your system.

---

### Step 3: Generate Your Video

1. Click the green **Generate Video** button.
2. A progress bar will show the current stage:
   - Initializing AI engine
   - Emotion detection (if enabled)
   - Motion generation (FOMM)
   - Lip-sync processing
   - Applying enhancements
   - Encoding final video
3. When complete, the video will play in the built-in player.

### After Rendering

Three buttons appear after a successful render:

- **Open Player** - Opens the video in your system's default video player.
- **Open Folder** - Opens the output folder to find your MP4 file.
- **Delete / Reset** - Removes the generated video and resets the UI.

All generated videos are saved in the `output/` folder with timestamps (e.g., `avatar_20260307_143025.mp4`).

---

## Option B: Web App (Next.js)

### Starting the Web App

```bash
# Terminal 1: Start the API server
uvicorn app.api:app --host 0.0.0.0 --port 8000

# Terminal 2: Start the frontend
cd frontend
npm install   # first time only
npm run dev
```

Open your browser to `http://localhost:3000`.

---

### Step 1: Choose Your Avatar

Scroll down to the **Interactive Demo** section. You have two options:

- **Pick a preset:** Click one of the six avatar cards (Professional, Creator, Educator, Game NPC, Brand, Custom).
- **Upload your own:** Scroll past the presets and click the upload area. Select a PNG or JPG face photo (up to 10 MB).

---

### Step 2: Add Script and Audio

1. **Type a script** in the text box, or click **Use sample** to load example text.
2. **Upload an audio file** (WAV or MP3) using the audio file picker. This is required for rendering.
3. **Select quality mode:**
   - **Auto** (default) - Best option for your hardware.
   - **Real-time** - Fastest processing.
   - **High quality** - Most realistic output.

4. **Toggle enhancements:**

   Below the quality selector, you'll see 10 enhancement chips. Click each to toggle it on/off:

   - Chips that are **bright and colored** = enabled
   - Chips that are **dim and gray** = disabled

   Three are enabled by default: **Emotion**, **Eye Gaze**, **Gestures**. The counter below shows how many you've selected.

   > **Tip:** Enhancements marked "(requires model download)" need additional model files to function. They will be skipped gracefully if the models aren't installed.

---

### Step 3: Generate Your Avatar

1. Click the large **Bring to Life** button.
2. A circular progress indicator will show the rendering stages:
   - Mapping facial features (25%)
   - Processing voice input (50%)
   - Rendering expressions (75%)
   - Finalizing avatar (100%)
3. When finished, the video appears in the **Your AI Avatar** section.

### After Rendering

- **Download MP4** - Save the video to your computer.
- **Create Another** - Reset everything and start a new avatar.
- The video player supports play/pause, loop, and fullscreen controls.

---

## Quick Reference

### Quality Modes Compared

| Mode | Speed | Visual Quality | Best For |
|---|---|---|---|
| Real-time | Fast (seconds) | Good | Testing, quick previews |
| High Quality | Slower (minutes) | Best | Final output, presentations |
| Auto | Depends on hardware | Best available | General use (recommended) |

### Default Enhancements (No Extra Setup)

These three work immediately after installation:

1. **Emotion** - Reads emotional tone from your audio and adjusts facial expressions.
2. **Eye Gaze** - Adds realistic blinking (3-5 blinks/minute) and micro eye movements.
3. **Gestures** - Adds subtle breathing animation and upper-body motion.

### Tips for Best Results

- Use a **clear, front-facing photo** with good lighting for your avatar image.
- For best lip-sync, use **clean audio** without background music.
- **High Quality** mode requires a CUDA-compatible GPU. Without one, it falls back to Real-time automatically.
- Keep your script under 60 seconds for best performance.
- The **transcript text** (what you typed in the script box) is passed to the AI to improve emotion detection and lip-sync accuracy. Even if you upload audio separately, typing the matching text helps.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| "Audio file missing" error | Make sure you generated audio (Desktop) or uploaded an audio file (Web) before clicking Generate. |
| "Generation in progress" error | Wait for the current render to finish before starting another. |
| Video looks choppy | Try enabling the **Eye Gaze** and **Gestures** enhancements for smoother motion. |
| Lip-sync doesn't match | Switch to **High Quality** mode, or enable **MuseTalk** if the model is installed. |
| Enhancement has no effect | Check the sidebar (Desktop) or tooltip (Web) to see if the model is available. Missing models are skipped silently. |
| API Status shows "Offline" (Desktop) | Make sure the backend server is running. Check the terminal for errors. |
| Web app can't connect to API | Ensure `uvicorn app.api:app` is running on port 8000. |

For more details, see [SETUP_GUIDE.md](./SETUP_GUIDE.md) and [TROUBLESHOOTING.md](./TROUBLESHOOTING.md).
