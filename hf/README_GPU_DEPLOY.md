# GPU Docker deployment (dev-v0.1.25 architecture)

This restores the **working high-quality infrastructure**: a persistent GPU Space
running the FastAPI backend + `render_pipeline`, with the engine repos cloned and
weights downloaded at startup. Lip-sync engines run **natively** here (unlike the
ZeroGPU Gradio Space, where subprocess engines can't use the GPU).

## What runs
`FOMM (motion) → Diff2Lip / MuseTalk / LatentSync (lip-sync) → Wav2Lip (fallback) → GFPGAN`
- **Diff2Lip** is the primary high-quality engine (weights in `ruslanmv/avatar-renderer`).
- **MuseTalk / LatentSync** repos are cloned; add their weights to enable them
  (their own HF repos) — until then they're skipped and Diff2Lip is used.
- Engine selectable per request via the API `qualityMode` / method, and the
  `render_workflow()` composition (motion driver + lip-sync + extras).

## Files
- `hf/Dockerfile` — CUDA 12.4 base, installs the full ML stack from `pyproject`.
- `hf/start.sh` — clones FOMM/Wav2Lip/SadTalker/Diff2Lip/guided-diffusion/MuseTalk/
  LatentSync, downloads checkpoints, then serves `uvicorn app.api:app` on 7860.
- `hf/README.md` — Space card (`sdk: docker`, `app_port: 7860`).

## Cutover (do this when you've chosen a GPU tier)
1. In the Space **Settings → Hardware**, select a GPU (T4 small is cheapest; A10G
   is faster for MuseTalk/LatentSync/1080p).
2. Deploy the Docker build to `ruslanmv/avatar-renderer` (replaces the Gradio
   ZeroGPU app): push `hf/README.md`, `hf/Dockerfile`, `hf/start.sh`, `app/`,
   `hf/static/`, `scripts/download_models.sh`, `pyproject.toml`, `LICENSE`.
3. First boot is slow: it installs the stack, clones repos, and downloads several
   GB of weights (HEALTHCHECK start-period is 20 min).

## Notes / honest caveats
- Build is large (CUDA + torch + opencv + gfpgan + …). Expect a few build/startup
  iterations to settle dependency pins — same shakeout we did for the Gradio app,
  but heavier. The build logs are the source of truth (`/logs/build`, `/logs/run`).
- On CPU (no GPU attached) it builds and serves the API/UI but the engines are
  slow/limited — attach the GPU for real results.
- Strict tiers (high_quality/premium/cinematic) never deliver a degraded fallback;
  if an engine fails they raise with a clear error + write `quality_report.json`.
