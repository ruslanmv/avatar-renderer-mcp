#!/usr/bin/env bash
#
# Single‑source checkpoint fetcher for Avatar‑Renderer‑Pod via Hugging Face Hub
# Usage:  bash scripts/download_models.sh [dest_dir]
#
set -euo pipefail

# Target directory for models
DEST_DIR="${1:-models}"

echo "🔽  Ensuring model directory '$DEST_DIR' exists and is writable..."
mkdir -p "$DEST_DIR"

# List of required checkpoint files (relative to DEST_DIR)
required=(
  "fomm/vox-cpk.pth"
  "diff2lip/Diff2Lip.pth"
  "sadtalker/SadTalker_V0.0.2_256.safetensors"
  "sadtalker/epoch_20.pth"
  "sadtalker/sadtalker.pth"
  "wav2lip/wav2lip_gan.pth"
  "gfpgan/GFPGANv1.3.pth"
)

# Check if all files already exist
all_exist=true
for path in "${required[@]}"; do
  if [[ ! -f "$DEST_DIR/$path" ]]; then
    all_exist=false
    break
  fi
done

if $all_exist; then
  echo "✔  All checkpoints already present — nothing to do."
  exit 0
fi

# Download missing files via the Hugging Face Hub Python client. This avoids a
# full `git clone` and does not require `git-lfs` to be installed locally.
PYTHON_BIN="${PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

"$PYTHON_BIN" - "$DEST_DIR" "${required[@]}" <<'PYCODE'
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError as exc:
    raise SystemExit(
        "huggingface_hub is not installed in the selected Python environment. "
        "Run `make install-internal` first, activate `.venv`, or set "
        "PYTHON=/path/to/.venv/bin/python before running this script."
    ) from exc

repo_id = os.environ.get("MODEL_REPO_ID", "ruslanmv/avatar-renderer")
repo_type = os.environ.get("MODEL_REPO_TYPE", "model")
token = os.environ.get("HF_TOKEN") or None
dest_dir = Path(sys.argv[1]).resolve()
required = sys.argv[2:]

print(f"🔽  Downloading missing checkpoints from {repo_id} with huggingface_hub...")
for rel_path in required:
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
PYCODE

# Done
echo "✅ All model checkpoints are now available in '$DEST_DIR'"

