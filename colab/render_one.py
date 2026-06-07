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
    """Point MODEL_ROOT / EXT_DEPS_DIR / sys.path at the cloned engine repos so
    the engine-registry availability probes succeed."""
    repo = Path(os.environ.get("REPO_DIR", Path(__file__).resolve().parents[1]))
    model_root = os.environ.setdefault("MODEL_ROOT", str(repo / "models"))
    ext = os.environ.setdefault("EXT_DEPS_DIR", str(repo / "external_deps"))
    subs = ["first-order-model", "Wav2Lip", "SadTalker", "Diff2Lip",
            "guided-diffusion", "MuseTalk", "LatentSync"]
    paths = [str(repo)] + [f"{ext}/{s}" for s in subs]
    for p in paths:
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ["PYTHONPATH"] = ":".join(paths)
    os.environ.setdefault("LIPSYNC_HF_REPO", "ruslanmv/avatar-renderer")
    _ = model_root  # documented side effect


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

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out = orchestrate(
        face_image=args.image, audio=args.audio, out_path=args.out,
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
