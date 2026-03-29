---
title: Avatar Renderer MCP
emoji: "\U0001F3AD"
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: true
license: apache-2.0
short_description: Production AI Avatar Video Generator with Lip-Sync
---

# Avatar Renderer MCP — Hugging Face Spaces

**Enterprise-grade AI avatar video generator** that creates photorealistic talking-head videos from a single portrait image and audio input.

## Features

- **Multiple Rendering Pipelines**: FOMM + Diff2Lip (high quality), SadTalker + Wav2Lip (real-time), Hallo3 (cinematic), 3D Gaussian Splatting
- **Face Enhancement**: GFPGAN-powered upscaling and restoration
- **10 Enhancement Modules**: Emotion expressions, eye gaze, gestures, MuseTalk, LivePortrait, and more
- **REST API**: Full OpenAPI-documented endpoints for programmatic access
- **MCP Integration**: AI-agent compatible via Model Context Protocol
- **Next.js React UI**: Enterprise-grade interactive demo with glassmorphism design

## How It Works

1. **Upload** a portrait image (PNG/JPG)
2. **Upload** or record audio (WAV/MP3)
3. **Select** rendering quality mode
4. **Generate** — the pipeline produces a lip-synced talking avatar video

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Next.js React enterprise demo UI |
| `/render-upload` | POST | Upload avatar + audio, start render |
| `/render` | POST | Render with server-side file paths |
| `/status/{job_id}` | GET | Check render status or download MP4 |
| `/avatars` | GET | List models and system capabilities |
| `/health/live` | GET | Liveness probe |
| `/health/ready` | GET | Readiness probe |
| `/docs` | GET | Interactive Swagger API docs |

## Quick Test

```bash
# Health check
curl https://ruslanmv-avatar-renderer.hf.space/health/live

# List system capabilities
curl https://ruslanmv-avatar-renderer.hf.space/avatars

# Render (upload files)
curl -X POST https://ruslanmv-avatar-renderer.hf.space/render-upload \
  -F "avatar=@portrait.png" \
  -F "audio=@speech.wav" \
  -F "qualityMode=auto"
```

## Quality Modes

| Mode | Pipeline | Speed | GPU Required |
|---|---|---|---|
| `real_time` | SadTalker + Wav2Lip | ~3s | No |
| `high_quality` | FOMM + Diff2Lip + GFPGAN | ~10-30s | Yes |
| `cinematic` | Hallo3 DiT | ~30s+ | Yes |
| `auto` | Auto-selects based on hardware | — | — |

## Architecture

```
Portrait Image ──┐
                  ├─→ Motion Model (FOMM/SadTalker) ─→ Lip-Sync (Diff2Lip/Wav2Lip)
Audio File ───────┘                                           │
                                                              ▼
                                              Face Enhancement (GFPGAN)
                                                              │
                                                              ▼
                                              Enhancements (emotion, gaze, gestures)
                                                              │
                                                              ▼
                                                    Output MP4 Video
```

## Links

- **GitHub**: [ruslanmv/avatar-renderer-mcp](https://github.com/ruslanmv/avatar-renderer-mcp)
- **Author**: [Ruslan Magana Vsevolodovna](https://ruslanmv.com)
- **License**: Apache-2.0
