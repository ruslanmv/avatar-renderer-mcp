#!/usr/bin/env bash
# =============================================================================
# colab_gpu_setup.sh — install engine repos + weights for the Colab GPU runner.
#
# Called by colab/colab_gpu_server.py (POST /setup). Reuses the repo's existing
# installers so there is a single source of truth.
#
# Usage:  bash scripts/colab_gpu_setup.sh <core_weights:0|1> [engine ...]
#   engines: musetalk diff2lip latentsync liveportrait emotion eye_gaze ...
# =============================================================================
set -uo pipefail

CORE_WEIGHTS="${1:-1}"; shift || true
ENGINES=("$@")

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_ROOT="${MODEL_ROOT:-$REPO_DIR/models}"
export EXT_DEPS_DIR="${EXT_DEPS_DIR:-$REPO_DIR/external_deps}"
mkdir -p "$MODEL_ROOT" "$EXT_DEPS_DIR"
cd "$REPO_DIR"

echo "== Colab GPU setup =="
echo "   MODEL_ROOT=$MODEL_ROOT"
echo "   EXT_DEPS_DIR=$EXT_DEPS_DIR"
echo "   core_weights=$CORE_WEIGHTS  engines=${ENGINES[*]:-<none>}"

clone() {  # name url dir
    if [ -d "$3/.git" ]; then echo "   [OK] $1"; else
        echo "   [DL] $1"; git clone --depth 1 "$2" "$3" 2>&1 | tail -2 || echo "   [SKIP] $1"; fi
}

# FOMM is the motion stage every pipeline engine needs.
clone "First Order Motion Model" "https://github.com/AliaksandrSiarohin/first-order-model.git" "$EXT_DEPS_DIR/first-order-model"
clone "Wav2Lip"          "https://github.com/Rudrabha/Wav2Lip.git"             "$EXT_DEPS_DIR/Wav2Lip"
clone "Diff2Lip"         "https://github.com/soumik-kanad/diff2lip.git"        "$EXT_DEPS_DIR/Diff2Lip"
clone "guided-diffusion" "https://github.com/openai/guided-diffusion.git"      "$EXT_DEPS_DIR/guided-diffusion"

# Per-engine repos + weights via the project's enhancement installer.
if [ "${#ENGINES[@]}" -gt 0 ]; then
    echo "== Installing engines: ${ENGINES[*]} =="
    bash scripts/download_enhancements.sh "${ENGINES[@]}" 2>&1 | tail -40 || true
fi

# Core checkpoints from the project's model repo.
if [ "$CORE_WEIGHTS" = "1" ]; then
    echo "== Downloading core weights =="
    python3 - <<'PY'
import os, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download
root = Path(os.environ["MODEL_ROOT"])
pairs = [
    ("fomm/vox-cpk.pth", "fomm/vox-cpk.pth"),
    ("wav2lip/wav2lip_gan.pth", "wav2lip/wav2lip_gan.pth"),
    ("diff2lip/Diff2Lip.pth", "diff2lip/Diff2Lip.pth"),
    ("gfpgan/GFPGANv1.3.pth", "gfpgan/GFPGANv1.3.pth"),
]
for remote, local in pairs:
    dst = root / local
    if dst.exists():
        print("  [OK]", local); continue
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        p = hf_hub_download(repo_id="ruslanmv/avatar-renderer", filename=remote, repo_type="model")
        shutil.copy2(p, dst)
        print("  [DL]", local, f"{dst.stat().st_size/1e6:.0f} MB")
    except Exception as e:
        print("  [SKIP]", local, str(e)[:120])
PY
fi

# MuseTalk weights (if requested) into models/musetalk/.
if printf '%s\n' "${ENGINES[@]:-}" | grep -q musetalk; then
    echo "== MuseTalk weights =="
    MUSE="$MODEL_ROOT/musetalk"; mkdir -p "$MUSE"
    if ! ls "$MUSE"/*.bin "$MUSE"/*.safetensors >/dev/null 2>&1; then
        huggingface-cli download TMElyralab/MuseTalk --local-dir "$MUSE" 2>&1 | tail -8 || true
    fi
    DW="$EXT_DEPS_DIR/MuseTalk/download_weights.sh"
    [ -f "$DW" ] && (cd "$EXT_DEPS_DIR/MuseTalk" && bash download_weights.sh 2>&1 | tail -8) || true
fi

# LatentSync weights (if requested).
if printf '%s\n' "${ENGINES[@]:-}" | grep -q latentsync; then
    echo "== LatentSync weights =="
    LS="$MODEL_ROOT/latentsync"; mkdir -p "$LS"
    huggingface-cli download chunyu-li/LatentSync --local-dir "$LS" 2>&1 | tail -8 || true
fi

echo "== Setup done =="
