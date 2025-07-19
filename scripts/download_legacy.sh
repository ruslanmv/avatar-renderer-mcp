#!/usr/bin/env bash
#
# Robust, mirror‑aware checkpoint fetcher for Avatar‑Renderer‑Pod
# Usage:  bash scripts/download_legacy.sh [dest_dir]
#
set -euo pipefail

DEST_DIR="${1:-models}"

echo "🔽  Creating folders under $DEST_DIR..."
mkdir -p \
  "$DEST_DIR/fomm" \
  "$DEST_DIR/diff2lip" \
  "$DEST_DIR/sadtalker" \
  "$DEST_DIR/wav2lip" \
  "$DEST_DIR/gfpgan"

###############################################################################
# Helper: try a list of mirrors until one succeeds
###############################################################################
download () {
  local dest="$1"; shift
  local urls=("$@")

  if [[ -s "$dest" ]]; then
    echo "✔  $(basename "$dest") already exists — skipping"
    return 0
  fi

  for url in "${urls[@]}"; do
    echo "⬇️  Attempting $(basename "$dest")  ⇢  $url"
    if wget -q --show-progress --retry-connrefused --timeout=30 --tries=3 \
            --continue "$url" -O "$dest"; then
      echo "✅  Downloaded $(basename "$dest")"
      return 0
    fi
    echo "⚠️  Mirror failed — trying next one..."
  done
  echo "❌  All mirrors failed for $(basename "$dest")" >&2
  return 1
}

###############################################################################
# 1) FOMM  – vox‑cpk.pth
###############################################################################
download "$DEST_DIR/fomm/vox-cpk.pth" \
  https://huggingface.co/gqy2468/first-order-model/resolve/main/vox-cpk.pth.tar :contentReference[oaicite:0]{index=0} \
  https://archive.org/download/vox-adv-cpk.pth_202103/vox-cpk.pth.tar :contentReference[oaicite:1]{index=1}

###############################################################################
# 2) Diff2Lip – HF mirrors first, fallback to Google Drive + gdown
###############################################################################
download "$DEST_DIR/diff2lip/Diff2Lip.pth" \
  https://huggingface.co/ameerazam08/diff2lip/resolve/main/checkpoints/checkpoint.pt :contentReference[oaicite:2]{index=2} \
  https://huggingface.co/ameerazam08/diff2lip/resolve/main/checkpoints/e7.15_model210000_notUsedInPaper.pt :contentReference[oaicite:3]{index=3}

if [[ ! -s "$DEST_DIR/diff2lip/Diff2Lip.pth" ]]; then
  if ! command -v gdown &>/dev/null; then
    python3 -m pip install --quiet gdown  # ubuntu has python3 but not 'python' by default :contentReference[oaicite:4]{index=4}
  fi
  gdown --fuzzy 1_psqO8LGwx1lJ-wa2xayDPa2LKmxq3ok \
        -O "$DEST_DIR/diff2lip/Diff2Lip.pth" || \
  echo "⚠️  Google Drive quota hit – continuing without Diff2Lip backup" :contentReference[oaicite:5]{index=5}
fi

###############################################################################
# 3) SadTalker – BOTH .pth *and* .safetensors
###############################################################################
# (a) epoch_20.pth — canonical training‑time checkpoint
download "$DEST_DIR/sadtalker/epoch_20.pth" \
  https://huggingface.co/vinthony/SadTalker/resolve/main/epoch_20.pth  \
  https://huggingface.co/camenduru/SadTalker/resolve/main/epoch_20.pth :contentReference[oaicite:7]{index=7}

# (b) packaged safetensors bundle (256‑px renderer)
download "$DEST_DIR/sadtalker/SadTalker_V0.0.2_256.safetensors" \
  https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors :contentReference[oaicite:8]{index=8} \
  https://huggingface.co/TonyD2046/sadtalker-01/resolve/main/SadTalker_V0.0.2_256.safetensors :contentReference[oaicite:9]{index=9}

###############################################################################
# 4) Wav2Lip  – GAN model
###############################################################################
download "$DEST_DIR/wav2lip/wav2lip_gan.pth" \
  https://huggingface.co/numz/wav2lip_studio/resolve/main/Wav2lip/wav2lip_gan.pth :contentReference[oaicite:10]{index=10} \
  https://github.com/Rudrabha/Wav2Lip/releases/download/v0.1/wav2lip_gan.pth :contentReference[oaicite:11]{index=11}

###############################################################################
# 5) GFPGAN 1.3
###############################################################################
download "$DEST_DIR/gfpgan/GFPGANv1.3.pth" \
  https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth :contentReference[oaicite:12]{index=12}

echo "🏁  All model checkpoints (including .safetensors) are now in '$DEST_DIR'."
