# 🚀 Google Colab Setup Guide

This guide explains how to use Avatar Renderer MCP in Google Colab.

## 📓 Quick Start

### Option 1: Open Directly in Colab

Click the badge below to open the notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ruslanmv/avatar-renderer-mcp/blob/main/demo_colab.ipynb)

### Option 2: Manual Upload

1. Download `demo_colab.ipynb` from this repository
2. Go to [Google Colab](https://colab.research.google.com/)
3. Click **File → Upload notebook**
4. Select the downloaded `demo_colab.ipynb` file

## ⚙️ Runtime Configuration

### Recommended Settings

For best performance, configure your Colab runtime:

1. **Click**: `Runtime → Change runtime type`
2. **Select**:
   - **Runtime type**: Python 3
   - **Hardware accelerator**: GPU (T4 or better)
   - **GPU type**: High-RAM (if available)

### Runtime Options

| Mode | Hardware | Speed | Quality | Cost |
|------|----------|-------|---------|------|
| **CPU** | No GPU | Slow (60s+) | Good | Free |
| **GPU (Free)** | T4 | Fast (20s) | Great | Free |
| **GPU (Pro)** | V100/A100 | Very Fast (10s) | Excellent | Paid |

## 📦 What Gets Installed

The notebook automatically installs:

1. **System Dependencies**
   - FFmpeg (video encoding)
   - System libraries (libsm6, libxext6, etc.)

2. **Python Dependencies**
   - PyTorch & TorchVision
   - FastAPI & Uvicorn
   - Diffusers & Transformers
   - GFPGAN (face enhancement)
   - Audio processing libraries

3. **External Repositories**
   - SadTalker (real-time avatar generation)
   - First Order Motion Model (FOMM)
   - Wav2Lip (lip synchronization)

4. **Model Checkpoints** (~3-5 GB)
   - SadTalker models
   - Wav2Lip GAN
   - GFPGAN v1.3
   - (Optional) FOMM and Diff2Lip models

## 🎬 Usage Flow

### Step-by-Step Process

1. **Run Section 1-4**: Install dependencies and download models
   - ⏱️ Time: 5-10 minutes (first time)
   - 💾 Storage: ~10 GB

2. **Run Section 5-6**: Verify installation and setup MCP server
   - ⏱️ Time: 1 minute
   - ✅ Checks all components are working

3. **Run Section 7**: Generate "Hello World" avatar
   - ⏱️ Time: 20-60 seconds
   - 🎥 Output: `hello_world.mp4`

4. **Run Section 8**: Test quality modes
   - ⏱️ Time: 1-5 minutes per mode
   - 🎥 Outputs: `realtime_demo.mp4`, `highquality_demo.mp4`

5. **Run Section 9**: Create custom avatars
   - 📤 Upload your own images and audio
   - 🎨 Generate personalized avatars

6. **Run Section 10**: Download videos
   - 📥 Download to your local machine

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 1. Models Not Downloading

**Problem**: Model download fails or times out

**Solutions**:
```python
# Option A: Re-run the download cell
# Option B: Download models manually and upload to Colab

# Option C: Use Google Drive for persistent storage
from google.colab import drive
drive.mount('/content/drive')
MODELS_DIR = Path("/content/drive/MyDrive/avatar-renderer-models")
```

#### 2. Out of Memory (OOM)

**Problem**: `CUDA out of memory` or `RuntimeError: Out of memory`

**Solutions**:
- Use `real_time` mode instead of `high_quality`
- Restart runtime: `Runtime → Restart runtime`
- Upgrade to Colab Pro for more RAM
- Reduce image resolution before processing

```python
# Use real-time mode
result = render_pipeline(
    face_image=str(image),
    audio=str(audio),
    out_path=str(output),
    quality_mode="real_time"  # CPU-friendly mode
)
```

#### 3. Import Errors

**Problem**: `ModuleNotFoundError` or import failures

**Solutions**:
- Re-run all installation cells in order
- Check that external dependencies were cloned
- Verify Python path includes external_deps

```python
# Manually add to Python path
import sys
sys.path.insert(0, '/content/avatar-renderer-mcp/external_deps')
```

#### 4. GPU Not Available

**Problem**: `cuda_available = False`

**Solutions**:
- Enable GPU: `Runtime → Change runtime type → GPU`
- Check GPU quota (free tier has limits)
- Use `real_time` mode which works on CPU

#### 5. Session Timeout

**Problem**: Colab disconnects after 12 hours or 90 minutes idle

**Solutions**:
- Keep browser tab active
- Use Google Drive to persist models
- Upgrade to Colab Pro for longer sessions
- Save important outputs to Drive regularly

```python
# Auto-save to Google Drive
from google.colab import drive
drive.mount('/content/drive')

output_dir = Path("/content/drive/MyDrive/avatar_outputs")
output_dir.mkdir(exist_ok=True)
```

#### 6. FFmpeg Encoding Errors

**Problem**: Video encoding fails

**Solutions**:
```bash
# Re-install FFmpeg
!apt-get update && apt-get install -y ffmpeg

# Check FFmpeg installation
!ffmpeg -version
```

#### 7. Audio Processing Errors

**Problem**: `librosa` or audio file errors

**Solutions**:
```bash
# Re-install audio libraries
!pip install --upgrade librosa soundfile

# Convert audio to correct format
!ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## 💡 Performance Tips

### 1. Use Google Drive for Models

Save models to Google Drive to avoid re-downloading each session:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Set models directory to Drive
MODELS_DIR = Path("/content/drive/MyDrive/avatar-renderer-models")
MODELS_DIR.mkdir(exist_ok=True, parents=True)
os.environ['MODEL_ROOT'] = str(MODELS_DIR)
```

**Pros**: Models persist across sessions
**Cons**: Slower access than local Colab storage

### 2. Enable GPU Runtime

Always use GPU for faster processing:

1. `Runtime → Change runtime type`
2. Select `GPU` hardware accelerator
3. Verify with: `torch.cuda.is_available()`

**Speed improvement**: 5-10x faster than CPU

### 3. Use Real-Time Mode on CPU

If GPU is not available, use real-time mode:

```python
result = render_pipeline(
    face_image=str(image),
    audio=str(audio),
    out_path=str(output),
    quality_mode="real_time"  # CPU-friendly
)
```

### 4. Batch Process Multiple Videos

Generate multiple videos in one session:

```python
demos = [
    {'image': 'image1.png', 'audio': 'audio1.wav', 'output': 'video1.mp4'},
    {'image': 'image2.png', 'audio': 'audio2.wav', 'output': 'video2.mp4'},
]

for demo in demos:
    render_pipeline(
        face_image=demo['image'],
        audio=demo['audio'],
        out_path=demo['output'],
        quality_mode="auto"
    )
```

### 5. Pre-process Images

Optimize images before processing:

```python
from PIL import Image

# Resize to optimal size
img = Image.open('large_image.jpg')
img = img.resize((1024, 1024))
img.save('optimized_image.png')
```

## 📊 Expected Processing Times

| Mode | Hardware | Resolution | Duration | Processing Time |
|------|----------|------------|----------|-----------------|
| Real-time | CPU | 512x512 | 5s | ~60s |
| Real-time | GPU (T4) | 512x512 | 5s | ~20s |
| High-quality | GPU (T4) | 1024x1024 | 5s | ~120s |
| High-quality | GPU (V100) | 1024x1024 | 5s | ~60s |

## 🔒 Security & Privacy

### Data Privacy

- **Colab Storage**: Temporary, deleted after session
- **Google Drive**: Persistent, under your Google account
- **Uploaded Files**: Stored in Colab, not shared with others
- **Generated Videos**: Only accessible to you

### Best Practices

1. **Don't upload sensitive/private images**
2. **Delete outputs after downloading**: `!rm -rf /content/avatar_outputs/*`
3. **Clear Drive storage regularly**: Check Drive quota
4. **Use private notebooks**: Don't share notebooks with sensitive data

## 📚 Additional Resources

### Documentation

- **Main README**: [README.md](README.md)
- **Local Demo**: [demo.ipynb](demo.ipynb)
- **Quality Modes**: [docs/QUALITY_MODES.md](docs/QUALITY_MODES.md)
- **API Reference**: See notebook Section 12

### Links

- **Website**: [https://avatar-renderer-mcp.vercel.app/](https://avatar-renderer-mcp.vercel.app/)
- **Repository**: [https://github.com/ruslanmv/avatar-renderer-mcp](https://github.com/ruslanmv/avatar-renderer-mcp)
- **Issues**: [GitHub Issues](https://github.com/ruslanmv/avatar-renderer-mcp/issues)
- **Author**: [Ruslan Magana Vsevolodovna](https://ruslanmv.com)

## 🆘 Getting Help

### If Something Goes Wrong

1. **Check error messages**: Read the full error trace
2. **Re-run installation**: Re-run Sections 1-4
3. **Check GPU**: Verify GPU is enabled and available
4. **Try real-time mode**: Use `quality_mode="real_time"`
5. **Restart runtime**: `Runtime → Restart runtime`
6. **Check disk space**: `!df -h /content`
7. **Check memory**: `!free -h`

### Report Issues

If you encounter persistent issues:

1. Copy the full error message
2. Note your Colab runtime configuration (CPU/GPU, RAM)
3. List steps to reproduce
4. Open an issue: [GitHub Issues](https://github.com/ruslanmv/avatar-renderer-mcp/issues)

Include:
- Colab notebook version
- Runtime type (CPU/GPU)
- Error message
- Cell that failed

## 🎓 Learning Resources

### Understanding the Pipeline

The avatar generation pipeline has two main modes:

#### Real-Time Mode (Fast)
```
Input Image + Audio
    ↓
SadTalker (head motion)
    ↓
Wav2Lip (lip sync)
    ↓
FFmpeg (encode)
    ↓
Output Video
```

#### High-Quality Mode (Slow, Better)
```
Input Image + Audio
    ↓
FOMM (head motion)
    ↓
Diff2Lip (diffusion lip sync)
    ↓
GFPGAN (face enhancement)
    ↓
FFmpeg (encode)
    ↓
Output Video (High Quality)
```

### Model Details

- **SadTalker**: 3DMM-based head motion
- **FOMM**: First-order motion model
- **Wav2Lip**: GAN-based lip synchronization
- **Diff2Lip**: Diffusion-based lip synthesis
- **GFPGAN**: Face restoration & enhancement

## 🚀 Next Steps

After successfully running the Colab notebook:

1. **Deploy to Production**: Use Docker/Kubernetes for production
2. **Integrate with Apps**: Use the REST API or MCP protocol
3. **Fine-tune Models**: Train on your own data
4. **Create More Content**: Generate avatars for your use case

---

**Happy Avatar Generation! 🎬✨**

*Questions? Contact: contact@ruslanmv.com*
