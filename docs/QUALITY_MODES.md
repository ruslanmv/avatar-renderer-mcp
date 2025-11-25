# Avatar Renderer Quality Modes

## Overview

Avatar Renderer MCP supports two distinct quality modes optimized for different use cases. Choose the mode that best fits your application requirements.

---

## Quality Modes Comparison

| Feature | Real-Time Mode | High-Quality Mode |
|---------|---------------|-------------------|
| **Processing Speed** | ⚡ Fast (<3s for 512x512) | 🎨 Slower (~10-30s) |
| **Use Cases** | Streaming, News, Chatbots | YouTube, Marketing, Demos |
| **Models Used** | SadTalker + Wav2Lip | FOMM + Diff2Lip + GFPGAN |
| **GPU Required** | Optional (CPU fallback) | Required |
| **Output Quality** | Good | Excellent |
| **Bitrate** | 2 Mbps | 6 Mbps |
| **Enhancement** | None (for speed) | GFPGAN face enhancement |
| **Encoding Preset** | Ultrafast/p1 | Slow/p7 |
| **Best For** | Live content, real-time | Pre-recorded content |

---

## 1. Real-Time Mode (`real_time`)

### Description
Optimized for **low-latency** applications where speed is critical. Perfect for live streaming, news broadcasts, and interactive AI chatbots.

### Technical Details
- **Pipeline**: SadTalker → Wav2Lip
- **Target Latency**: <3 seconds for 512x512 @ 25fps
- **Video Encoding**:
  - GPU: H.264 NVENC preset `p1` (fastest)
  - CPU: libx264 preset `ultrafast`
- **Bitrate**: 2 Mbps
- **Face Enhancement**: Disabled (to minimize latency)
- **GPU Requirement**: Optional (works on CPU)

### Use Cases
- ✅ **Live News Broadcasts**: Real-time virtual news anchors
- ✅ **AI Chatbots**: Interactive virtual assistants
- ✅ **Live Streaming**: Twitch/YouTube live streaming avatars
- ✅ **Video Conferencing**: Virtual presence in meetings
- ✅ **Real-Time Translation**: Live dubbing with lip-sync

### Example Request (REST API)
```json
POST /render
{
  "avatarPath": "/path/to/avatar.png",
  "audioPath": "/path/to/speech.wav",
  "qualityMode": "real_time"
}
```

### Example Request (MCP STDIO)
```json
{
  "tool": "render_avatar",
  "params": {
    "avatar_path": "/path/to/avatar.png",
    "audio_path": "/path/to/speech.wav",
    "quality_mode": "real_time"
  }
}
```

### Example CLI
```bash
python -m app.pipeline \
  --face avatar.png \
  --audio speech.wav \
  --out output.mp4 \
  --mode real_time
```

---

## 2. High-Quality Mode (`high_quality`)

### Description
Produces **maximum quality** output for pre-recorded content. Ideal for YouTube videos, marketing materials, and high-fidelity demonstrations.

### Technical Details
- **Pipeline**: FOMM → Diff2Lip → GFPGAN Enhancement
- **Processing Time**: ~10-30 seconds (depending on GPU)
- **Video Encoding**:
  - GPU: H.264 NVENC preset `p7` (best quality)
  - CPU: libx264 preset `slow`, CRF 18
- **Bitrate**: 6 Mbps
- **Face Enhancement**: GFPGAN with RealESRGAN upscaling
- **GPU Requirement**: Required (V100 or better recommended)

### Use Cases
- ✅ **YouTube Content**: High-quality video tutorials and presentations
- ✅ **Marketing Videos**: Product demos, explainer videos
- ✅ **Educational Content**: Online courses, lectures
- ✅ **Virtual Influencers**: Social media content creation
- ✅ **Film/TV Production**: Virtual extras, dubbing

### Example Request (REST API)
```json
POST /render
{
  "avatarPath": "/path/to/avatar.png",
  "audioPath": "/path/to/speech.wav",
  "qualityMode": "high_quality"
}
```

### Example Request (MCP STDIO)
```json
{
  "tool": "render_avatar",
  "params": {
    "avatar_path": "/path/to/avatar.png",
    "audio_path": "/path/to/speech.wav",
    "quality_mode": "high_quality"
  }
}
```

### Example CLI
```bash
python -m app.pipeline \
  --face avatar.png \
  --audio speech.wav \
  --out output.mp4 \
  --mode high_quality
```

---

## 3. Auto Mode (`auto`) - Default

### Description
Automatically selects the best quality mode based on:
- GPU availability
- Model checkpoint availability
- System resources

### Selection Logic
```python
if GPU_available and FOMM_available and Diff2Lip_available:
    use high_quality mode
elif SadTalker_available and Wav2Lip_available:
    use real_time mode
else:
    raise error (no suitable models)
```

### When to Use
- Development and testing
- Unknown deployment environments
- Graceful degradation scenarios

---

## Model Requirements

### Real-Time Mode Models
| Model | Size | Purpose |
|-------|------|---------|
| SadTalker | ~500MB | Head pose & expression |
| Wav2Lip | ~150MB | Lip synchronization |
| GFPGAN (optional) | ~350MB | Face enhancement |

### High-Quality Mode Models
| Model | Size | Purpose |
|-------|------|---------|
| FOMM | ~200MB | First-order motion model |
| Diff2Lip | ~1.2GB | Diffusion-based lip-sync |
| GFPGAN | ~350MB | Face enhancement (required) |

---

## Performance Benchmarks

### Real-Time Mode
| Hardware | Resolution | FPS | Latency |
|----------|-----------|-----|---------|
| V100 GPU | 512x512 | 200+ | ~2.5s |
| T4 GPU | 512x512 | 150+ | ~3.5s |
| CPU (16 cores) | 512x512 | 30+ | ~8s |

### High-Quality Mode
| Hardware | Resolution | FPS | Processing Time |
|----------|-----------|-----|----------------|
| V100 GPU | 512x512 | 50+ | ~12s |
| A100 GPU | 1024x1024 | 80+ | ~15s |
| T4 GPU | 512x512 | 30+ | ~25s |

---

## Deployment Recommendations

### For Live Streaming (News TV, Chatbots)
```yaml
quality_mode: real_time
gpu: Optional (T4 or better recommended)
replicas: 3-5
autoscaling: CPU/Memory based
target_latency: <3s
```

### For Content Creation (YouTube, Marketing)
```yaml
quality_mode: high_quality
gpu: Required (V100 or better)
replicas: 1-2
autoscaling: Queue depth based
target_quality: Maximum
```

---

## Kubernetes Configuration Example

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: avatar-renderer-realtime
spec:
  replicas: 5
  template:
    spec:
      containers:
      - name: avatar-renderer
        env:
        - name: DEFAULT_QUALITY_MODE
          value: "real_time"
        resources:
          limits:
            nvidia.com/gpu: 1  # Optional for real-time
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: avatar-renderer-hq
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: avatar-renderer
        env:
        - name: DEFAULT_QUALITY_MODE
          value: "high_quality"
        resources:
          limits:
            nvidia.com/gpu: 1  # Required for high-quality
```

---

## Troubleshooting

### Real-Time Mode Issues
| Issue | Solution |
|-------|----------|
| Still too slow | Reduce resolution, disable enhancement |
| Quality not acceptable | Switch to high_quality mode |
| Models not found | Run `make download-models` |

### High-Quality Mode Issues
| Issue | Solution |
|-------|----------|
| CUDA OOM | Reduce batch size, use lower resolution |
| No GPU available | Use real_time mode or add GPU |
| GFPGAN fails | Check GFPGAN checkpoint path |

---

## Environment Variables

```bash
# Set default quality mode
export DEFAULT_QUALITY_MODE=real_time

# Model paths (automatically detected)
export MODEL_ROOT=/models
export FOMM_CKPT_DIR=/models/fomm
export DIFF2LIP_CKPT_DIR=/models/diff2lip
export SADTALKER_CKPT_DIR=/models/sadtalker
export WAV2LIP_CKPT=/models/wav2lip/wav2lip_gan.pth
export GFPGAN_CKPT=/models/gfpgan/GFPGANv1.4.pth
```

---

## API Response

Both modes return the same response format:

```json
{
  "jobId": "550e8400-e29b-41d4-a716-446655440000",
  "output": "/tmp/550e8400-e29b-41d4-a716-446655440000.mp4",
  "qualityMode": "real_time",
  "statusUrl": "/status/550e8400-e29b-41d4-a716-446655440000"
}
```

---

## Summary

- **Use `real_time`** for: Live streaming, chatbots, news broadcasts, real-time applications
- **Use `high_quality`** for: YouTube videos, marketing content, high-quality demos
- **Use `auto`** for: Development, testing, or unknown environments

Choose the mode that best fits your **latency requirements** and **quality expectations**.

---

## Additional Resources

- [Model Download Guide](./MODELS.md)
- [API Documentation](./API.md)
- [Deployment Guide](./DEPLOYMENT.md)
- [Performance Tuning](./PERFORMANCE.md)

---

**Author**: Ruslan Magana Vsevolodovna
**Website**: https://ruslanmv.com
**License**: Apache-2.0
