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

# Ensure huggingface_hub library is installed
python3 - <<'PYCODE'
import sys, subprocess
try:
    import huggingface_hub
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "huggingface_hub"])
PYCODE

# Clone repo into a temporary directory
echo "🔽  Cloning repository to temporary location..."
tmpdir=$(mktemp -d)
git clone --depth 1 https://huggingface.co/ruslanmv/avatar-renderer "$tmpdir"

# Copy only missing files into DEST_DIR
echo "📂  Copying required checkpoints into '$DEST_DIR'..."
for path in "${required[@]}"; do
  src="$tmpdir/$path"
  dst="$DEST_DIR/$path"
  if [[ -f "$src" ]]; then
    mkdir -p "$(dirname "$dst")"
    cp -n "$src" "$dst"
    echo "   • $path"
  else
    echo "⚠️  Source missing in repo: $path" >&2
  fi
done

# Clean up
echo "🧹  Cleaning up temporary files..."
rm -rf "$tmpdir"

# Done
echo "✅ All model checkpoints are now available in '$DEST_DIR'"

