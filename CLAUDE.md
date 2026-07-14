# Avatar Renderer MCP — project instructions

Photo + audio → talking-avatar video. Multi-engine lip-sync with a strict
production contract.

## Architecture (engine selection)

- `app/engines.py` — engine registry (`registry`) with availability + capability probes.
- `app/compliance.py` — per-engine license guard (`assert_engine_allowed`, `is_commercial_safe`).
- `app/render.py` — `orchestrate(face_image, audio, out_path, quality_mode, engine, commercial)`
  is the production entry point: `select_engine` → run → soft fallback (non-strict) → quality gate.
  `render_method(method, …)` runs a specific method (bake-off / UI selector).
- `app/lipsync.py` — in-process Wav2Lip. `full_face=True` (default) is the faithful
  dev-v0.1.25 path (whole-crop paste + per-frame GFPGAN); `full_face=False` is the
  mouth-band/static-base anti-flicker variant.
- `app/modes.py` — quality tiers. **Strict tiers** (`high_quality`/`premium`/`cinematic`)
  must NEVER silently downgrade — they raise instead.

In-process engines (run on ZeroGPU/CPU): `simple`, `wav2lip_fast`, `wav2lip_raw`,
`wav2lip_band`, `fullface`. Pipeline engines (need repos + weights + GPU):
`diff2lip`, `musetalk`, `latentsync`, `wav2lip`.

## Rules

- Premium/strict tiers must not fall back to `simple`/Wav2Lip silently. Keep the
  honest-error behavior.
- Don't commit secrets/tokens. Don't push to branches other than the one assigned.
- Run tests with coverage opts off if needed: `python -m pytest tests/ -q -o addopts=""`.

## GPU testing via the temporary Colab server

Local CUDA is usually unavailable here. To validate engines on a real GPU, use the
**temporary Colab GPU job server** (see `docs/COLAB_GPU_TESTING.md`). When these env
vars are present, prefer it over assuming local GPU:

- `COLAB_GPU_URL` — the trycloudflare/ngrok URL printed by the Colab notebook
- `COLAB_GPU_TOKEN` — the per-session secret

Allowlisted client commands (never request arbitrary shell from the server):

```bash
scripts/claude_colab_client.sh GET  /health
scripts/claude_colab_client.sh GET  /engines
scripts/claude_colab_client.sh POST /git/pull      '{"branch":"<branch>"}'
scripts/claude_colab_client.sh POST /setup         '{"engines":["musetalk","diff2lip"]}'
scripts/claude_colab_client.sh POST /render/sample '{"engine":"musetalk","quality_mode":"high_quality","text":"Hi"}'
scripts/claude_colab_client.sh wait <job_id>
scripts/claude_colab_client.sh DOWNLOAD /artifact/<name> /tmp/out.mp4
```

Typical loop: edit → commit/push branch → `/git/pull` → `/setup` (once) →
`/render/sample` → `wait` → `/artifact` → inspect → fix. If an engine errors,
read `/jobs/<id>` log and run `/setup` for it; do not downgrade the tier to hide it.
