#!/usr/bin/env python3
"""Download Avatar Renderer model checkpoints from Hugging Face Hub.

This Python entry point is used by `make download-models` so model downloads do
not depend on executing a shell script. That avoids CRLF failures on WSL or
Windows-mounted checkouts where `scripts/download_models.sh` may have been
converted to DOS line endings.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

REQUIRED_CHECKPOINTS = (
    "fomm/vox-cpk.pth",
    "diff2lip/Diff2Lip.pth",
    "sadtalker/SadTalker_V0.0.2_256.safetensors",
    "sadtalker/epoch_20.pth",
    "sadtalker/sadtalker.pth",
    "wav2lip/wav2lip_gan.pth",
    "gfpgan/GFPGANv1.3.pth",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dest_dir",
        nargs="?",
        default="models",
        help="Directory where model checkpoints should be stored (default: models).",
    )
    return parser.parse_args()


def missing_huggingface_hub_message() -> str:
    return (
        "huggingface_hub is not installed in the selected Python environment. "
        "Run `make install-internal` first, activate `.venv`, or set "
        "PYTHON=/path/to/.venv/bin/python before running this script."
    )


def all_checkpoints_exist(dest_dir: Path) -> bool:
    return all((dest_dir / rel_path).is_file() for rel_path in REQUIRED_CHECKPOINTS)


def main() -> int:
    args = parse_args()
    dest_dir = Path(args.dest_dir).resolve()

    print(f"🔽  Ensuring model directory '{dest_dir}' exists and is writable...")
    dest_dir.mkdir(parents=True, exist_ok=True)

    if all_checkpoints_exist(dest_dir):
        print("✔  All checkpoints already present — nothing to do.")
        return 0

    if importlib.util.find_spec("huggingface_hub") is None:
        raise SystemExit(missing_huggingface_hub_message())

    from huggingface_hub import hf_hub_download

    repo_id = os.environ.get("MODEL_REPO_ID", "ruslanmv/avatar-renderer")
    repo_type = os.environ.get("MODEL_REPO_TYPE", "model")
    token = os.environ.get("HF_TOKEN") or None

    print(f"🔽  Downloading missing checkpoints from {repo_id} with huggingface_hub...")
    for rel_path in REQUIRED_CHECKPOINTS:
        dst = dest_dir / rel_path
        if dst.is_file():
            print(f"   • {rel_path} already present")
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                filename=rel_path,
                local_dir=str(dest_dir),
                token=token,
            )
        except Exception as exc:  # noqa: BLE001 - CLI should show the failing file.
            raise SystemExit(f"Failed to download {rel_path} from {repo_id}: {exc}") from exc
        print(f"   • {rel_path}")

    print(f"✅ All model checkpoints are now available in '{dest_dir}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
