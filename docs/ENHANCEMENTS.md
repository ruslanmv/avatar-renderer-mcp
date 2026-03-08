# Enhancement Modules

The enhancement system provides 10 plug-and-play modules that improve avatar realism. Each module operates at a specific pipeline stage and can be enabled/disabled per request.

## Quick Start

Three enhancements work **out of the box** with zero model downloads:

```bash
# Via environment variable
export DEFAULT_ENHANCEMENTS="emotion_expressions,eye_gaze_blink,gesture_animation"

# Via REST API
curl -X POST http://localhost:8080/render \
  -H 'Content-Type: application/json' \
  -d '{
    "avatarPath": "/path/to/avatar.png",
    "audioPath": "/path/to/speech.wav",
    "enhancements": ["emotion_expressions", "eye_gaze_blink", "gesture_animation"]
  }'
```

## Pipeline Stages

Enhancements run in a fixed stage order:

```
1. tts           -> CosyVoice TTS (emotional prosody)
2. pre_motion    -> Emotion Detection (sets ctx.detected_emotion)
3. motion_driver -> LivePortrait / Hallo3 / Gaussian Splatting
4. lip_sync      -> MuseTalk / LatentSync
5. post_process  -> Eye Gaze, Viseme Guided, Gesture Animation
```

Within each stage, enhancements run by priority (lower number = runs first).

## Module Reference

### 1. Emotion Expressions

| Property | Value |
|----------|-------|
| **Module** | `app/enhancements/emotion_expressions.py` |
| **Stage** | `pre_motion` (priority 10) |
| **Models needed** | None (prosody fallback) |
| **Always available** | Yes |

Detects emotion from audio using a 3-tier fallback:
1. **NVIDIA Audio2Emotion** (ONNX model, best quality)
2. **Transformers sentiment** (text-based, auto-downloaded)
3. **Audio prosody heuristics** (pitch, energy, tempo — always works)

Sets `ctx.detected_emotion` and `ctx.expression_params` for downstream enhancements.

**Supported emotions:** happy, sad, angry, surprised, fearful, disgusted, neutral

---

### 2. MuseTalk Lip-Sync

| Property | Value |
|----------|-------|
| **Module** | `app/enhancements/musetalk_lipsync.py` |
| **Stage** | `lip_sync` (priority 10) |
| **Models needed** | MuseTalk checkpoints in `models/musetalk/` |
| **Always available** | No |

Replaces Wav2Lip with MuseTalk v1.5 for real-time lip-sync at 30+ FPS. Significantly better visual quality with fewer artifacts.

```bash
# Download MuseTalk models
./scripts/download_enhancements.sh --musetalk
```

---

### 3. Eye Gaze + Blink

| Property | Value |
|----------|-------|
| **Module** | `app/enhancements/eye_gaze_blink.py` |
| **Stage** | `post_process` (priority 10) |
| **Models needed** | None |
| **Always available** | Yes (needs OpenCV) |

Adds natural blink patterns (15-20 blinks/minute) and micro-saccade gaze movements. Uses OpenCV cascade classifiers for eye detection, with a fallback to approximate eye-band processing.

**Key functions:**
- `generate_blink_schedule(duration, fps)` — Poisson-process blink timing
- `generate_gaze_schedule(duration, fps)` — Gaussian micro-saccade offsets
- `apply_blink_to_frame(frame, progress)` — Smooth eyelid darkening
- `apply_gaze_shift(frame, dx, dy)` — Subtle iris translation

---

### 4. LivePortrait Driver

| Property | Value |
|----------|-------|
| **Module** | `app/enhancements/liveportrait_driver.py` |
| **Stage** | `motion_driver` (priority 10) |
| **Models needed** | LivePortrait in `external_deps/LivePortrait/` |
| **Always available** | No |

Expression-controllable portrait animation using implicit keypoints and retargeting. Reads `ctx.expression_params` from the emotion enhancement to drive contextually appropriate expressions.

---

### 5. LatentSync Lip-Sync

| Property | Value |
|----------|-------|
| **Module** | `app/enhancements/latentsync_lipsync.py` |
| **Stage** | `lip_sync` (priority 20) |
| **Models needed** | LatentSync in `models/latentsync/` |
| **Always available** | No |

ByteDance's latent-space lip-sync. Operates entirely in diffusion latent space, producing sharper results than pixel-space methods.

---

### 6. Hallo3 Cinematic

| Property | Value |
|----------|-------|
| **Module** | `app/enhancements/hallo3_cinematic.py` |
| **Stage** | `motion_driver` (priority 20) |
| **Models needed** | Hallo3 in `external_deps/hallo3/` |
| **Always available** | No |

Diffusion Transformer (DiT) based cinematic avatar generation from Fudan University (CVPR 2025, MIT license). Best quality, longer processing time.

---

### 7. CosyVoice TTS

| Property | Value |
|----------|-------|
| **Module** | `app/enhancements/cosyvoice_tts.py` |
| **Stage** | `tts` (priority 10) |
| **Models needed** | CosyVoice server at `COSYVOICE_URL` |
| **Always available** | No (needs running server) |

Emotion-conditioned text-to-speech from Alibaba. Maps detected emotion to prosody style prompts (e.g., "happy" -> "cheerful and warm").

---

### 8. Viseme Guided

| Property | Value |
|----------|-------|
| **Module** | `app/enhancements/viseme_guided.py` |
| **Stage** | `post_process` (priority 20) |
| **Models needed** | None (needs viseme JSON data) |
| **Always available** | Conditional (needs `ctx.viseme_json`) |

Constrains lip shapes using Montreal Forced Aligner viseme data. Ensures phonetically accurate mouth shapes — bilabial closure on B/M/P, proper openness on vowels.

**Viseme classes:** A, E, O, BMP, FV, L, D, C, G, S, TH, rest

---

### 9. Gesture Animation

| Property | Value |
|----------|-------|
| **Module** | `app/enhancements/gesture_animation.py` |
| **Stage** | `post_process` (priority 30) |
| **Models needed** | None (procedural fallback) |
| **Always available** | Yes |

Co-speech upper-body animation. Uses GestureLSM (ICCV 2025) when available, falls back to procedural idle animation:
- **Breathing**: ~12-20 breaths/min vertical oscillation
- **Shoulder sway**: Slow lateral movement
- Affects only the lower 2/3 of the frame (body area)

---

### 10. 3D Gaussian Splatting

| Property | Value |
|----------|-------|
| **Module** | `app/enhancements/gaussian_splatting.py` |
| **Stage** | `motion_driver` (priority 30) |
| **Models needed** | InsTaG in `external_deps/InsTaG/` |
| **Always available** | No |

Next-generation 3D Gaussian Splatting backend using InsTaG (CVPR 2025). Real-time, high-fidelity talking head synthesis with 3D consistency.

---

## Python API

```python
from app.enhancements import registry, EnhancementContext

# List all registered enhancements
for enh in registry.list_all():
    print(f"{enh.name}: stage={enh.stage}, available={enh.is_available()}")

# Apply specific enhancements
ctx = EnhancementContext(
    face_image="avatar.png",
    audio_path="speech.wav",
    quality_mode="auto",
)
ctx = registry.apply_all(ctx, enabled={"emotion_expressions", "eye_gaze_blink"})
print(f"Emotion: {ctx.detected_emotion}")
print(f"Applied: {ctx.applied_enhancements}")
```

## Writing Custom Enhancements

```python
from app.enhancements import Enhancement, EnhancementContext, registry

class MyEnhancement(Enhancement):
    @property
    def name(self) -> str:
        return "my_enhancement"

    @property
    def stage(self) -> str:
        return "post_process"  # tts | pre_motion | motion_driver | lip_sync | post_process

    @property
    def priority(self) -> int:
        return 50  # lower = runs first within stage

    def is_available(self) -> bool:
        return True  # check for model files, dependencies, etc.

    def apply(self, ctx: EnhancementContext) -> EnhancementContext:
        # Read from ctx, modify frames/video, write back to ctx
        return ctx

# Register (usually at module level for auto-discovery)
registry.register(MyEnhancement())
```

## Testing

All enhancements have mock-based unit tests that run without GPU or models:

```bash
python -m pytest tests/test_enhancements.py -v
# 55 passed in ~0.3s
```

## Architecture Diagram

<img src="architecture.svg" alt="Enhanced Pipeline Architecture" width="900"/>
