#!/usr/bin/env bash
#
# Download all model checkpoints for Avatar‑Renderer‑Pod:
#   • FOMM (vox‑cpk.pth)
#   • Diff2Lip (Diff2Lip.pth)
#   • SadTalker (sadtalker.pth)
#   • Wav2Lip (wav2lip_gan.pth)
#   • GFPGAN (GFPGANv1.3.pth)
#
# Usage:
#   bash scripts/download_models.sh [<dest_dir>]
#

set -euo pipefail

DEST_DIR=${1:-models}

echo "🔽 Creating folders under $DEST_DIR..."
mkdir -p \
  "$DEST_DIR/fomm" \
  "$DEST_DIR/diff2lip" \
  "$DEST_DIR/sadtalker" \
  "$DEST_DIR/wav2lip" \
  "$DEST_DIR/gfpgan"

download() {
  local url=$1 dest=$2
  if [ -f "$dest" ]; then
    echo "✔  $dest already exists, skipping"
  else
    echo "⬇️  Downloading $(basename $dest)..."
    wget -q --show-progress "$url" -O "$dest"
  fi
}

# 1) FOMM checkpoint
download \
  https://github.com/AliaksandrSiarohin/first-order-model/releases/download/v0.1.0/vox-cpk.pth \
  "$DEST_DIR/fomm/vox-cpk.pth"

# 2) Diff2Lip weights
download \
  https://github.com/YuanGary/DiffusionLi/releases/download/v1.0/Diff2Lip.pth \
  "$DEST_DIR/diff2lip/Diff2Lip.pth"

# 3) SadTalker checkpoint
download \
  https://github.com/OpenTalker/SadTalker/releases/download/v0.1.0/sadtalker.pth \
  "$DEST_DIR/sadtalker/sadtalker.pth"

# 4) Wav2Lip GAN model
download \
  https://github.com/Rudrabha/Wav2Lip/releases/download/v0.1/wav2lip_gan.pth \
  "$DEST_DIR/wav2lip/wav2lip_gan.pth"

# 5) GFPGAN face‑enhancer
download \
  https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth \
  "$DEST_DIR/gfpgan/GFPGANv1.3.pth"

echo "✅ All model checkpoints are now available in '$DEST_DIR'."
