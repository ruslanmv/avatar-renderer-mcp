# Multi-Engine Lip-Sync Platform — Feasibility Study & Design

**Verdict: feasible** — the plugin/registry/orchestrator design is sound and matches
how production platforms (fal.ai, Replicate, ComfyUI) compose these exact models.
But four research findings materially change the design, and one (lip-sync engines
need *video*, not a still) is a hard architectural dependency. Details below.

---

## 1. Research findings (verified online, June 2026)

| Engine | Input | License → commercial? | GPU / VRAM | Role |
|---|---|---|---|---|
| **SadTalker** | **single image + audio → video** | Apache-2.0 → ✅ yes | A100 ~24s; 256/512px | **Motion driver** (the missing piece) |
| **MuseTalk** | **video** (256×256 face) + audio | MIT → ✅ yes | V100 30fps; modern NVIDIA | Standard / premium-fast lip refine |
| **LatentSync** | **video** (25fps) + audio (16k) | OpenRAIL++ → ✅ yes (attrib.) | **8GB (v1.5) / 18GB (v1.6)**; ~2:30min max on 4090 | Premium lip refine (diffusion) |
| **Diff2Lip** | video + audio | (in-repo) → maybe | GPU | High-quality fallback |
| **Wav2Lip** | video + audio | **research-only (LRS2)** → ❌ no | CPU/GPU | Preview / fallback only |
| **LivePortrait** | image + driving video/audio | MIT code but **InsightFace buffalo_l** → ❌ commercial restricted | GPU | Expression driver (non-commercial) |
| **GFPGAN** | frames | Apache-2.0 → ✅ yes | GPU | Face restore |
| **CodeFormer** | frames | **NTU S-Lab 1.0** → ❌ non-commercial | GPU | Face restore (non-commercial) |

Sources at the bottom.

### Finding A — the critical one: lip-sync engines need a VIDEO, not a photo
MuseTalk and LatentSync (and Diff2Lip/Wav2Lip) **edit the mouth region of an
existing video**. They do **not** animate a still image. So a single-photo avatar
**cannot** go straight into MuseTalk/LatentSync. You first need an audio-driven
**motion stage** to turn the photo into a talking-head video:

```
photo + audio → [SadTalker]  (audio → talking-head VIDEO)
              → [MuseTalk | LatentSync | Diff2Lip]  (refine the lips on that video)
              → mouth cleanup → GFPGAN → encode
```

This is the single biggest correction to the proposed plan: **SadTalker (or
MuseV/Hallo3) is mandatory as the motion driver** — it's the audio-driven
equivalent of the old FOMM stage, and it's Apache-licensed (commercial-safe).
(If the user uploads an existing video — "repair" mode — the motion stage is
skipped and the engines run directly.)

### Finding B — license guard is well-founded, with specifics
Commercial-safe set: **SadTalker, MuseTalk, LatentSync, GFPGAN** (+ MediaPipe).
Must be **excluded from commercial/premium**: **Wav2Lip** (research-only),
**CodeFormer** (NTU non-commercial), **LivePortrait** (InsightFace buffalo_l).
The proposed `license_guard` is correct — these specifics make it actionable.

### Finding C — metrics: SyncNet ✅, VMAF ✅(limited)
- **SyncNet** (LSE-C confidence, LSE-D distance, offset) is **self-referential**
  → perfect for an objective **lip-sync quality gate** on generated output.
- **VMAF is full-reference** — it needs a ground-truth reference video. For
  *generation* there is none, so VMAF **cannot** score talking-head quality. Use
  it only for **encode regression** (pre- vs post-encode), not the premium gate.

### Finding D — ZeroGPU is incompatible; needs a persistent GPU
All these engines run as heavy processes (often subprocess) and need the GPU for
the whole run. ZeroGPU only grants a GPU inside a decorated function and not to
subprocesses → these engines can't run there. The platform must be a
**persistent GPU Space/host** (T4 works for SadTalker/MuseTalk; **A10G/A100 (24GB+)
recommended** for LatentSync v1.6 and 1080p).

---

## 2. Is the proposed architecture feasible? Yes — with these adjustments

The proposed structure (`backends/` plugins + `registry` + `pipeline/orchestrator`
+ `postprocess/` + `quality/` + `compliance/`) is **the right design** and is
directly feasible. Adjusted blueprint:

```
request (image|video + audio + mode + engine + commercial)
  → input_validation (face present, size, codec)
  → MOTION stage (only if input is an image):
        SadTalker (audio→video)            [commercial ✅]
        | LivePortrait (non-commercial)    [video-driven]
  → LIP-SYNC stage (engine selected by mode/license/availability):
        LatentSync | MuseTalk | Diff2Lip   [commercial ✅]
        | Wav2Lip (preview/non-commercial only)
  → mouth_artifact_cleanup (MediaPipe FaceMesh, engine-independent)
  → face_restore: GFPGAN [commercial ✅] | CodeFormer [non-commercial]
  → (optional) upscale / RIFE interpolation
  → encode (tier crf/fps/res)
  → quality gate: SyncNet LSE-C/offset + face-visibility + mouth-artifact + audio
  → return MP4 + quality_report.json (provenance, engine, fallback_used, scores)
```

Strict rule (sound and feasible): **preview may fall back; premium must not**
silently fall back — it retries only among approved premium engines, else errors.

**What already exists in this repo** that maps cleanly:
- `app/enhancements/` already has `musetalk_lipsync`, `latentsync_lipsync`,
  `liveportrait_driver`, `hallo3_cinematic`, `cosyvoice_tts` (subprocess plugins).
- `app/render.py` has `render_workflow()` + engine catalog and `app/modes.py` has
  tiers with strict no-fallback. The proposed `backends/`+`orchestrator` is a
  cleaner refactor of these — **incremental, not a rewrite**.
- SadTalker repo is already cloned by `hf/start.sh`, but **not wired** as a motion
  backend → that's the main new integration.

---

## 3. Effort & risk per component (feasibility detail)

| Component | Feasible? | Effort | Risk |
|---|---|---|---|
| Backend interface + registry | ✅ | Low | Low (clean refactor of existing registry) |
| Quality modes + strict fallback | ✅ done | Low | Low (already implemented in `modes.py`) |
| SadTalker motion backend | ✅ | **High** | Med-High (heavy deps; many sub-models; weights) |
| MuseTalk backend | ✅ | Med | Med (repo deps; needs video input) |
| LatentSync backend | ✅ | Med-High | High (diffusion; 8–18GB VRAM; slow; deps) |
| MediaPipe mouth cleanup | ✅ | Med | Med (landmark-based is far better than color heuristics) |
| GFPGAN after cleanup | ✅ done | Low | Low |
| SyncNet quality gate | ✅ | Med | Med (extra model; calibrate LSE-C threshold) |
| License guard | ✅ | Low | Low |
| Persistent GPU deployment | ✅ | Med | **Cost** (paid GPU) + build/dep shakeout |

**Overall:** architecturally low-risk; the cost is **integration & maintenance** —
each engine is a finicky third-party repo with its own deps/weights, and the GPU
host bills hourly. This is exactly why fal/Replicate exist (they host these as
managed endpoints).

---

## 4. Recommendation (phased, feasible path)

1. **Foundation (low risk, mostly done):** keep the `modes.py` strict tiers +
   `render.py` engine catalog; add the `backends/` interface + `registry` +
   `license_guard` (commercial matrix above). Wire `quality_report.json` (done).
2. **Motion driver:** integrate **SadTalker** as the audio→video stage (Apache,
   commercial-safe) — this unlocks single-photo → high-quality talking head.
3. **Premium lip refine:** add **MuseTalk** (MIT, fast) then **LatentSync**
   (OpenRAIL++, best) backends that refine SadTalker's output video.
4. **Cleanup & gate:** MediaPipe FaceMesh mouth cleanup (engine-independent,
   before GFPGAN) + SyncNet LSE-C gate for premium.
5. **Deploy:** persistent **A10G** GPU Docker Space (per the GPU deploy doc).
   Keep the free ZeroGPU Space as the "preview/fast" tier.

### Shortcut to consider (also feasible, lower ops cost)
Instead of self-hosting every engine, the orchestrator's backends can call
**managed APIs** (fal.ai / Replicate host MuseTalk, LatentSync, SadTalker). Same
plugin interface; no GPU to operate; pay-per-call. Best for getting premium
quality live quickly, then optionally self-host later for cost.

---

## 5. Honest caveats
- VMAF won't gate generation quality (no reference) — use SyncNet/LSE-C.
- Wav2Lip / CodeFormer / LivePortrait are **not commercial-safe** as-is.
- LatentSync is VRAM-heavy and slow; budget A10G/A100 and clip-length limits.
- Self-hosting all engines = ongoing dependency maintenance; managed APIs trade
  that for per-call cost.

## Sources
- MuseTalk (MIT, real-time, video input): https://github.com/TMElyralab/MuseTalk · https://huggingface.co/TMElyralab/MuseTalk · https://arxiv.org/abs/2410.10122
- LatentSync (OpenRAIL++, 8–18GB VRAM, 25fps video): https://github.com/bytedance/LatentSync · https://huggingface.co/ByteDance/LatentSync-1.5 · https://arxiv.org/abs/2412.09262
- Wav2Lip (research-only, LRS2): https://github.com/Rudrabha/Wav2Lip/issues/104 · https://github.com/Rudrabha/Wav2Lip/issues/623
- SadTalker (Apache-2.0, single image + audio): https://github.com/OpenTalker/SadTalker · https://arxiv.org/abs/2211.12194
- LivePortrait (InsightFace commercial restriction): https://github.com/KwaiVGI/LivePortrait · https://arxiv.org/abs/2407.03168
- CodeFormer (NTU S-Lab non-commercial): https://github.com/sczhou/CodeFormer/blob/master/LICENSE
- GFPGAN (Apache-2.0): https://arxiv.org/abs/2101.04061
- SyncNet (LSE-C/LSE-D/offset): https://towardsdatascience.com/syncnet-paper-easily-explained/ · VideoReTalking: https://arxiv.org/abs/2211.14758
- VMAF (full-reference): https://en.wikipedia.org/wiki/Video_Multimethod_Assessment_Fusion
- MediaPipe (Apache, face mesh): https://en.wikipedia.org/wiki/MediaPipe
