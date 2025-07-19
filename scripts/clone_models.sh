#!/usr/bin/env bash
#
# Single‑source checkpoint fetcher for Avatar‑Renderer‑Pod via Git clone from Hugging Face Hub
# Usage:  bash scripts/download_models.sh [dest_dir]
#
set -euo pipefail

DEST_DIR="${1:-models}"

echo "🔽  Ensuring model directory '$DEST_DIR'..."
mkdir -p "$DEST_DIR"

# List of required checkpoint files
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

if \$all_exist; then
  echo "✔  All checkpoints already present in '$DEST_DIR' — nothing to do."
  exit 0
fi

# Clone or update the repo
if [[ -d "$DEST_DIR/.git" ]]; then
  echo "🔄  Updating existing repository in '$DEST_DIR'..."
  git -C "$DEST_DIR" pull --depth 1
else
  echo "🔽  Cloning ruslanmv/avatar-renderer into '$DEST_DIR'..."
  rm -rf "$DEST_DIR"
  git clone --depth 1 https://huggingface.co/ruslanmv/avatar-renderer "$DEST_DIR"
fi

echo "✅  Model checkpoints are now available in '$DEST_DIR':"
# List downloaded files
for path in "${required[@]}"; do
  echo "   • $path"
done
