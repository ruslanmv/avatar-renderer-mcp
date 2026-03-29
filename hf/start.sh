#!/bin/bash
# =============================================================================
# Avatar Renderer MCP — HF Spaces Startup Script
# =============================================================================
# Follows ollabridge pattern: heavy assets downloaded at startup, not build.
# 1) Clone external ML repos (if not present)
# 2) Download model checkpoints from HF Hub
# 3) Start FastAPI server on port 7860
# =============================================================================

set -e

echo "=============================================="
echo "  Avatar Renderer MCP — Hugging Face Spaces"
echo "=============================================="
echo ""

# -- Environment ---------------------------------------------------------------
export HOME=/tmp
export MODEL_ROOT="${MODEL_ROOT:-/app/models}"
export EXT_DEPS_DIR="${EXT_DEPS_DIR:-/app/external_deps}"
export PYTHONUNBUFFERED=1
mkdir -p /tmp/avatar-jobs /tmp/avatar-cache "$EXT_DEPS_DIR"

# -- [1/3] Clone external ML repos -------------------------------------------
echo "[1/3] Setting up external ML dependencies..."

clone_repo() {
    local name="$1" url="$2" dir="$3"
    if [ -d "$dir" ] && [ "$(ls -A "$dir" 2>/dev/null)" ]; then
        echo "       [OK] $name"
    else
        echo "       [DL] $name..."
        git clone --depth=1 "$url" "$dir" 2>/dev/null && echo "       [OK] $name" \
            || echo "       [SKIP] $name (clone failed)"
    fi
}

clone_repo "First Order Motion Model" \
    "https://github.com/AliaksandrSiarohin/first-order-model.git" \
    "$EXT_DEPS_DIR/first-order-model"

clone_repo "Wav2Lip" \
    "https://github.com/Rudrabha/Wav2Lip.git" \
    "$EXT_DEPS_DIR/Wav2Lip"

clone_repo "SadTalker" \
    "https://github.com/OpenTalker/SadTalker.git" \
    "$EXT_DEPS_DIR/SadTalker"

clone_repo "Diff2Lip" \
    "https://github.com/soumik-kanad/diff2lip.git" \
    "$EXT_DEPS_DIR/Diff2Lip"

clone_repo "guided-diffusion" \
    "https://github.com/openai/guided-diffusion.git" \
    "$EXT_DEPS_DIR/guided-diffusion"

# Set PYTHONPATH to include all external deps
export PYTHONPATH="/app:${EXT_DEPS_DIR}/first-order-model:${EXT_DEPS_DIR}/Wav2Lip:${EXT_DEPS_DIR}/SadTalker:${EXT_DEPS_DIR}/Diff2Lip:${EXT_DEPS_DIR}/guided-diffusion"

# -- [2/3] Download model checkpoints ----------------------------------------
echo "[2/3] Checking model checkpoints..."
python3 -c "
import os, shutil
from huggingface_hub import hf_hub_download

repo = 'ruslanmv/avatar-renderer'
root = os.environ.get('MODEL_ROOT', '/app/models')

pairs = [
    ('fomm/vox-cpk.pth', f'{root}/fomm/vox-cpk.pth'),
    ('wav2lip/wav2lip_gan.pth', f'{root}/wav2lip/wav2lip_gan.pth'),
    ('diff2lip/Diff2Lip.pth', f'{root}/diff2lip/Diff2Lip.pth'),
    ('gfpgan/GFPGANv1.3.pth', f'{root}/gfpgan/GFPGANv1.3.pth'),
    ('sadtalker/sadtalker.pth', f'{root}/sadtalker/sadtalker.pth'),
    ('sadtalker/SadTalker_V0.0.2_256.safetensors', f'{root}/sadtalker/SadTalker_V0.0.2_256.safetensors'),
    ('sadtalker/epoch_20.pth', f'{root}/sadtalker/epoch_20.pth'),
]

for hf_path, local_path in pairs:
    if os.path.isfile(local_path):
        sz = os.path.getsize(local_path) / 1024 / 1024
        print(f'  [OK] {os.path.basename(hf_path)} ({sz:.0f} MB)')
        continue
    try:
        print(f'  [DL] {hf_path}...')
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        p = hf_hub_download(repo_id=repo, filename=hf_path, repo_type='model', local_dir='/tmp/hf_dl')
        shutil.copy2(p, local_path)
        sz = os.path.getsize(local_path) / 1024 / 1024
        print(f'       OK ({sz:.0f} MB)')
    except Exception as e:
        print(f'  [SKIP] {hf_path}: {e}')

shutil.rmtree('/tmp/hf_dl', ignore_errors=True)
" 2>&1

# -- [3/3] Start FastAPI -------------------------------------------------------
echo "[3/3] Starting server on port ${PORT:-7860}..."
echo ""
echo "=============================================="
echo "  Ready!"
echo "  - UI:       /"
echo "  - API Docs: /docs"
echo "  - Health:   /health/live"
echo "  - Render:   /render-upload"
echo "=============================================="
echo ""

exec python -m uvicorn app.api:app \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-7860}" \
    --workers 1 \
    --timeout-keep-alive 300 \
    --log-level info
