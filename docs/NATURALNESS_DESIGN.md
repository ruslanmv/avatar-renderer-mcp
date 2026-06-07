# Naturalness Design — Making the Avatar Less Robotic

A complete review of the program, the techniques in use, and the recommended
"best solution" architecture for natural, lifelike talking avatars — plus how the
optional add-ons are exposed in the settings menu so you can compare them.

---

## 1. Current architecture (what runs today)

```
 Browser (Vercel / Space UI)
        │  image + audio + quality + add-ons
        ▼
 Hugging Face ZeroGPU Space  (Gradio, space_app.py /predict, @spaces.GPU)
        │
        ▼  app.render.render()  — RENDER_BACKEND=auto
        ├─ 1) Wav2Lip (in-process, GPU)        → mouth/lip motion
        │      • OpenCV Haar face box (no s3fd)
        │      • silence-trim + per-frame silence gating (mouth rests when quiet)
        │      • channel order [masked, reference]  (correct lip inpainting)
        │      • mouth-only feather blend          (keep original sharp face)
        │      • GFPGAN restoration                (sharpen the generated face)
        │      • OPTIONAL add-ons (post-process)   ← compare here
        ├─ 2) full FOMM/Diff2Lip pipeline         (only if repos+weights present)
        └─ 3) ffmpeg "demo" renderer              (always-works fallback)
```

Models pulled at runtime from the `ruslanmv/avatar-renderer` model repo:
`wav2lip_gan.pth`, `GFPGANv1.3.pth` (+ facexlib detectors).

---

## 2. Why a talking avatar looks "robotic", and the technique for each

| Robotic symptom | Cause | Technique to fix | Status |
|---|---|---|---|
| Frozen head (statue) | Wav2Lip only moves the mouth | Subtle head/breathing motion; or a motion driver (SadTalker/LivePortrait) | ✅ add-on (gesture) / ⏳ driver |
| Never blinks | Static eyes | Periodic blink synthesis (eye landmarks) | ✅ add-on (eye_gaze_blink) |
| Dead, fixed stare | No gaze shifts | Micro-saccades / gaze drift | ✅ add-on (eye_gaze_blink) |
| Flat affect | Mouth ignores tone | Emotion from prosody → expression/brow | ◐ analysis only without a driver |
| Mouth keeps moving in silence | Wav2Lip animates all audio | Silence trim + per-frame gating | ✅ done |
| Blurry mouth/face | Wav2Lip 96px upscaled | Mouth-only blend + GFPGAN restore | ✅ done |
| Mouth-only, no jaw/cheeks | Wav2Lip scope | Full-face driver (SadTalker) | ⏳ driver |

---

## 3. The optional add-ons (in Settings → compare the extras)

Exposed both in the Space UI (a checkbox group) and the Vercel frontend
(the "Enhancements" toggles), passed through `/predict` → `render()` → applied as
frame post-processing on the Wav2Lip output:

| Setting label | id | What it does | Model-free? |
|---|---|---|---|
| Head motion (breathing/sway) | `gesture_animation` | Gentle periodic translate/scale of the frame — adds "alive" micro-motion | ✅ OpenCV only |
| Eye blink & gaze | `eye_gaze_blink` | Detects eyes (Haar) and adds periodic blinks + small gaze shifts | ✅ OpenCV only |
| Emotion analysis | `emotion_expressions` | Reads pitch/energy → emotion label (drives expression when a motion driver is present) | ✅ librosa |

Turn them on/off and re-render to compare against the plain (robotic) baseline.
The first two produce a visible difference on the Wav2Lip output today; emotion
becomes visible once a motion driver (below) is enabled.

> These are additive and safe: if an add-on fails, rendering continues with the
> base frames (it never breaks a render).

---

## 4. Recommended "best solution" (layered, quality-ordered)

The most natural result comes from a **motion driver** (head + expression) *under*
the lip-sync, then restoration — not lip-sync alone. Target pipeline:

```
audio ─┬─► emotion/prosody  ─────────────┐
       │                                  ▼
image ─┼─► motion driver (SadTalker)  → head pose + blinks + expression
       │        (or LivePortrait w/ a driving clip)
       └─► lip-sync (Wav2Lip / LatentSync) → precise mouth on driven frames
                                  │
                                  ▼
                       GFPGAN / CodeFormer restore  → crisp face
                                  │
                                  ▼
                     light add-ons (micro head sway, gaze)  → final polish
```

Recommended choices, by goal:

- **Most natural single-image talking head:** **SadTalker** — produces head
  motion, blinks and expressions from one image + audio (weights already in the
  model repo: `sadtalker.pth`, `SadTalker_V0.0.2_256.safetensors`, `epoch_20.pth`).
  Heavier; the right next step for "less robotic".
- **Sharper lips than Wav2Lip:** **LatentSync** or **MuseTalk** (higher-res mouth).
- **Expression control / re-enactment:** **LivePortrait** (needs a driving video).
- **Always-on polish:** the model-free add-ons in §3 + GFPGAN (already wired).

Quality tiers to expose later:
- **Fast** = Wav2Lip + add-ons (current, seconds).
- **Natural** = SadTalker + GFPGAN (head motion + expression).
- **HD** = SadTalker + LatentSync + GFPGAN.

---

## 5. How to compare (A/B method)

1. Generate once with **no** add-ons (baseline).
2. Generate again with **Head motion + Eye blink** enabled.
3. Compare side by side — the second should show subtle head movement and
   blinking (less statue-like). We measure this objectively with per-frame pixel
   motion in the head/eye regions during verification.

---

## 6. Roadmap to fully natural

1. ✅ Correct, sharp, silence-aware Wav2Lip (done).
2. ✅ Optional model-free naturalness add-ons, toggleable to compare (done).
3. ⏳ Integrate **SadTalker** as a "Natural" quality tier (head pose + expression).
4. ⏳ Emotion → expression wired through the driver (makes the emotion add-on visible).
5. ⏳ LatentSync/MuseTalk for HD mouth; per-user ZeroGPU quota for scale.
