# Testing the engines on a Colab GPU (temporary job server)

This repo can be tested on a **real GPU** from your local Claude Code session by
turning a Colab GPU runtime into a small, **token-authenticated, allowlisted**
HTTP job server reachable over a Cloudflare quick tunnel. No SSH, no arbitrary
shell — Colab is used as intended (notebook-driven compute), and Claude Code
calls the API with `curl`.

> Why not run the premium engines on the HF ZeroGPU Space? ZeroGPU only
> GPU-accelerates in-process code inside `@spaces.GPU`. MuseTalk/LatentSync/Diff2Lip
> run as subprocesses needing their own repos + multi-GB weights, so they can't use
> that allocation. A persistent Colab GPU runs them natively.

## Architecture

```
Local machine                         Google Colab GPU runtime
┌────────────────────┐                ┌───────────────────────────────────────┐
│ Claude Code        │                │ FastAPI job server (colab/colab_gpu_   │
│  • edits repo      │   HTTPS+token  │ server.py), allowlisted endpoints:     │
│  • commits/pushes  │ ─────────────► │   /git/pull  /setup  /gpu/check        │
│  • curl client     │                │   /engines   /render/sample            │
│ scripts/claude_    │ ◄───────────── │   /jobs/{id} /artifact/{name} /logs    │
│  colab_client.sh   │   JSON / mp4   │                                        │
└────────────────────┘                │ render_one.py → app.render.orchestrate │
        ▲                             │ (same engine registry as production)   │
        │ public URL + token          │ scripts/colab_gpu_setup.sh installs     │
        │                             │ engine repos + weights                 │
   Cloudflare Quick Tunnel  ◄─────────┤ uvicorn :8080                          │
   (trycloudflare.com)                └───────────────────────────────────────┘
```

## Components (in this repo)

| File | Role |
|---|---|
| `colab/Avatar_Renderer_GPU_Server.ipynb` | Launcher: GPU check, clone, install, **token**, start server + tunnel, print URL |
| `colab/colab_gpu_server.py` | FastAPI server — auth, background jobs, allowlisted endpoints, artifact download |
| `colab/render_one.py` | Fixed-argv render CLI the server calls → `orchestrate()` (no shell) |
| `scripts/colab_gpu_setup.sh` | Clones engine repos + downloads weights (reuses `download_enhancements.sh`) |
| `scripts/claude_colab_client.sh` | Local `curl` client (`GET`/`POST`/`DOWNLOAD`/`wait`) |

## Endpoints (all require `X-Colab-Token`, except `/ping`)

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/ping` | — | liveness (no auth) |
| GET | `/health` | — | GPU/VRAM/torch/repo summary |
| GET | `/engines` | — | registry: availability + commercial license |
| POST | `/git/pull` | `{branch}` | `{job_id}` |
| POST | `/gpu/check` | — | `{job_id}` (nvidia-smi + torch) |
| POST | `/setup` | `{engines:[…], core_weights}` | `{job_id}` |
| POST | `/render/sample` | `{engine,quality_mode,text\|audio_path,use_sample,…}` | `{job_id, expected_artifact}` |
| POST | `/upload` (multipart) | `file=@…` | `{name}` (saved to INPUT_DIR) |
| GET | `/jobs/{id}` | — | status, returncode, **log tail**, artifact url |
| GET | `/artifact/{name}` | — | the rendered `.mp4` |
| GET | `/logs/{name}` | — | tail of a log file |

Background-job model: long actions (`/setup`, `/render/sample`) return a `job_id`
immediately; poll `/jobs/{id}` (or use the client's `wait`) so the tunnel never
times out.

## Workflow loop

1. Claude Code edits code locally, commits, pushes the branch.
2. You open `colab/Avatar_Renderer_GPU_Server.ipynb` in Colab (GPU runtime).
3. The notebook prints `PUBLIC_URL` + `TOKEN` and the two `export` lines.
4. Paste the exports into the terminal where Claude Code runs.
5. Claude Code drives the server:
   ```bash
   scripts/claude_colab_client.sh GET  /health
   scripts/claude_colab_client.sh POST /git/pull '{"branch":"claude/project-review-9nn71"}'
   scripts/claude_colab_client.sh POST /setup    '{"engines":["musetalk","diff2lip"]}'
   scripts/claude_colab_client.sh GET  /engines              # confirm available:true
   JID=$(scripts/claude_colab_client.sh POST /render/sample \
     '{"engine":"musetalk","quality_mode":"high_quality","text":"Hello from Colab GPU"}' \
     | python3 -c 'import sys,json;print(json.load(sys.stdin)["job_id"])')
   scripts/claude_colab_client.sh wait $JID
   scripts/claude_colab_client.sh DOWNLOAD /artifact/<name> /tmp/out.mp4
   ```
6. Claude Code reads `/jobs/{id}` logs, fixes the code, repeats.
7. Stop the notebook's last cell (or disconnect the runtime) when done.

## Security model

1. **No SSH. No arbitrary-shell endpoint.** Only the allowlisted actions above.
2. Every request requires `X-Colab-Token` (constant-time compared).
3. The token is **regenerated every session** (`secrets.token_urlsafe(32)`).
4. `engine` / `quality_mode` are validated against allowlists; the render argv is
   built by the server (never from caller-supplied shell).
5. File access is restricted to `INPUT_DIR` / `ARTIFACT_DIR` with name sanitization
   (no path traversal).
6. Keep the tunnel URL private; quick tunnels are for short testing only.
7. Stop the tunnel + server (or disconnect) after the session.

## VRAM guide

| GPU | Engines |
|---|---|
| **T4** (free, 16 GB) | `simple`, `wav2lip_fast`, `wav2lip_raw`, `wav2lip_band`, `fullface`, `diff2lip`, **`musetalk`** |
| **A100 / L4** (Pro) | all of the above **+ `latentsync`** |
