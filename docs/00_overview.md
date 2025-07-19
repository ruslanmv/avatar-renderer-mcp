# Avatar Renderer Pod Overview

This document gives a high‑level introduction to the **Avatar Renderer Pod**, a self‑contained microservice that transforms a single still image + audio clip into a high‑fidelity talking‑head video. The service exposes both:

- **REST API** (FastAPI) for synchronous “demo” usage  
- **MCP STDIO** interface for integration into any MCP Gateway  

It combines state‑of‑the‑art open‑source models:

- **First‑Order Motion Model (FOMM)** for natural head & body motion  
- **Diff2Lip** (Diffusion‑based mouth generation) for photorealistic lip sync  
- **GFPGAN** for optional face enhancement  
- **Wav2Lip** fallback / fine‑tuning for GAN‑based lip sync  

---

## Goals & Use Cases

- **Production‑grade avatar video** for presentations, e‑learning, marketing  
- **On‑demand rendering** in GPU clusters (OpenShift, Kubernetes)  
- **Real‑time streaming** via MCP & WebRTC integration (future)  
- **Plug‑and‑play** in larger orchestrations (VideoGenie, conversational agents)  

---

## Core Components

| Component                | Responsibility                                        |
| ------------------------ | ----------------------------------------------------- |
| `app/api.py`             | FastAPI service exposing `/render` and `/status`     |
| `app/mcp_server.py`      | STDIO‑based MCP server implementing `render_avatar`   |
| `app/pipeline.py`        | Orchestrates FOMM → Diff2Lip → GFPGAN → FFmpeg steps  |
| `app/worker.py`          | Celery worker entrypoint for background jobs         |
| `app/viseme_align.py`    | (Optional) phoneme→viseme alignment via MFA          |
| `app/settings.py`        | Pydantic config (env vars, model paths, resource caps)|

---

## Runtime Model Directory

> **Not included** in Git. Must be populated at runtime (e.g. via `scripts/download_models.sh` or mounted PVC).

```text
models/
├── fomm/         # FOMM pre-trained checkpoints
├── diff2lip/     # Diff2Lip Stable Diffusion weights
├── gfpgan/       # GFPGAN face-enhancer weights
└── wav2lip/      # wav2lip_gan.pth checkpoint
````



## Workflow

1. **Request**

   * **REST**: Client POSTs JSON `{ "avatar": "alice.png", "audio": "https://...", "driver_video": null }` to `/render`.
   * **MCP**: Gateway streams `{ tool: "render_avatar", params: {...} }` over STDIO.

2. **Job Scheduling**

   * REST handler enqueues background Celery task (returns `jobId`).
   * MCP server calls `render_pipeline(...)` synchronously.

3. **Rendering Pipeline**

   * **FOMM** generates coarse motion from reference (audio‑driven or driver video).
   * **Diff2Lip** refines lip region using diffusion-based model.
   * **GFPGAN** optionally sharpens and enhances face.
   * **FFmpeg** NVENC encodes final MP4 to `/tmp/{jobId}.mp4`.

4. **Result Delivery**

   * **REST**: `/status/{jobId}` returns MP4 when ready (or “processing”).
   * **MCP**: STDIO server replies `{ jobId, output: "/tmp/{jobId}.mp4" }`.

5. **Storage & CDN**

   * (In full VideoGenie) upload to Cloud Object Storage & cache via CIS.

---

## Deployment Targets

* **Local dev**: Docker with `--gpus all` on workstation
* **Kubernetes**: Helm chart requests `nvidia.com/gpu: 1`, tolerations `dedicated=gpu:NoSchedule`
* **IBM Cloud**: OpenShift GPU pool + KEDA autoscaling

---

## Next Steps

1. Populate `models/` (run `scripts/download_models.sh`)
2. `docker build . && docker run --gpus all -e MCP_ENABLE=true avatar-renderer`
3. Register in MCP Gateway (`mcp-tool.json`)
4. Integrate with VideoGenie’s Kafka→Argo pipeline

For more details, see:

* **02\_mcp\_integration.md** — how to hook into MCP Gateway
* **test\_render\_api.py** — REST smoke tests
* **test\_mcp\_stdio.py** — MCP STDIO round‑trip

