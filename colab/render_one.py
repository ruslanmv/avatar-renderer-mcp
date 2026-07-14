#!/usr/bin/env python3
"""
colab/render_one.py — single-render CLI used by the Colab GPU job server.

The server NEVER runs arbitrary shell. It validates parameters, then invokes
this script with a fixed argv. This script sets up the engine search paths and
calls the production multi-engine orchestrator (app.render.orchestrate), so what
runs here is exactly what ships in the API.

Usage:
    python colab/render_one.py --image P --audio A --out O \
        --engine musetalk --quality high_quality [--commercial]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _setup_engine_paths() -> None:
    """Set the engine env and make the repo root win on sys.path.

    Do NOT pre-seed the engine repos onto sys.path. The pipeline loads each engine
    repo itself (app.pipeline.load_module_from_path inserts the engine dir at the
    FRONT when it imports e.g. FOMM's fomm_wrapper.py) — but it SKIPS that
    promotion if the dir is already on sys.path. If we pre-added the dirs, the
    repo's own top-level modules (`demo.py`, the `app` package) would shadow the
    engines' identically-named modules (FOMM's `demo.py` → ImportError on
    `load_checkpoints`; MuseTalk's `app.py`). The premium subprocesses (Diff2Lip,
    MuseTalk) build their own PYTHONPATH, and app.lipsync adds the Wav2Lip dir
    on demand — so the repo root on sys.path is all we need here.
    """
    repo = Path(os.environ.get("REPO_DIR", Path(__file__).resolve().parents[1]))
    os.environ.setdefault("MODEL_ROOT", str(repo / "models"))
    os.environ.setdefault("EXT_DEPS_DIR", str(repo / "external_deps"))
    os.environ.setdefault("LIPSYNC_HF_REPO", "ruslanmv/avatar-renderer")
    if str(repo) in sys.path:
        sys.path.remove(str(repo))
    sys.path.insert(0, str(repo))         # the repo's own `app` package wins


def main() -> int:
    ap = argparse.ArgumentParser(description="Single avatar render via orchestrate().")
    ap.add_argument("--image", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", default="auto")
    ap.add_argument("--quality", default="standard")
    ap.add_argument("--commercial", action="store_true")
    args = ap.parse_args()

    _setup_engine_paths()

    from app.render import orchestrate  # heavy import after paths are set

    # Normalize audio to PCM WAV. Pipeline engines (FOMM's wrapper) read the audio
    # with wave.open(), which rejects mp3 ("file does not start with RIFF id").
    import subprocess
    audio = args.audio
    if not audio.lower().endswith(".wav"):
        wav = str(Path(args.out).with_suffix("")) + ".in.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", audio,
                        "-ar", "16000", "-ac", "1", wav], check=True)
        audio = wav

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out = orchestrate(
        face_image=args.image, audio=audio, out_path=args.out,
        quality_mode=args.quality, engine=args.engine, commercial=args.commercial,
    )
    size = Path(out).stat().st_size if Path(out).exists() else 0

    # The sidecar JSON is read back by the server to report provenance.
    result = {"ok": Path(out).exists(), "output": out, "bytes": size,
              "engine_requested": args.engine, "quality": args.quality}
    sidecar = Path(args.out).with_suffix(".result.json")
    sidecar.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
