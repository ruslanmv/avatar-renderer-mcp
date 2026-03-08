#!/usr/bin/env bash
#
# download_enhancements.sh – Clone repos and download models for enhancement modules
#
# Usage:
#   ./scripts/download_enhancements.sh [--all | --top3 | ENHANCEMENT_NAME ...]
#
# Examples:
#   ./scripts/download_enhancements.sh --top3              # Top 3: emotion, musetalk, eye_gaze
#   ./scripts/download_enhancements.sh --all               # All 10 enhancements
#   ./scripts/download_enhancements.sh musetalk liveportrait  # Specific ones
#
# This script is ADDITIVE — it never removes existing models or repos.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_ROOT="${MODEL_ROOT:-$PROJECT_ROOT/models}"
EXT_DEPS="${EXT_DEPS_DIR:-$PROJECT_ROOT/external_deps}"

mkdir -p "$MODEL_ROOT" "$EXT_DEPS"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

info()  { echo "  [INFO] $*"; }
warn()  { echo "  [WARN] $*" >&2; }
ok()    { echo "  [OK]   $*"; }
skip()  { echo "  [SKIP] $* (already exists)"; }

clone_if_missing() {
    local repo_url="$1"
    local target_dir="$2"
    local branch="${3:-}"

    if [ -d "$target_dir/.git" ]; then
        skip "$target_dir"
        return
    fi

    info "Cloning $repo_url -> $target_dir"
    if [ -n "$branch" ]; then
        git clone --depth 1 --branch "$branch" "$repo_url" "$target_dir" 2>/dev/null || \
            git clone --depth 1 "$repo_url" "$target_dir"
    else
        git clone --depth 1 "$repo_url" "$target_dir"
    fi
    ok "Cloned $target_dir"
}

download_hf_model() {
    local repo_id="$1"
    local target_dir="$2"
    local filename="${3:-}"

    if [ -d "$target_dir" ] && [ "$(ls -A "$target_dir" 2>/dev/null)" ]; then
        skip "$target_dir"
        return
    fi

    mkdir -p "$target_dir"
    info "Downloading from HuggingFace: $repo_id"

    if command -v huggingface-cli &>/dev/null; then
        if [ -n "$filename" ]; then
            huggingface-cli download "$repo_id" "$filename" --local-dir "$target_dir" || warn "Download failed: $repo_id/$filename"
        else
            huggingface-cli download "$repo_id" --local-dir "$target_dir" || warn "Download failed: $repo_id"
        fi
    else
        warn "huggingface-cli not found. Install with: pip install huggingface-hub[cli]"
        warn "Manual download: https://huggingface.co/$repo_id"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Enhancement Setup Functions
# ─────────────────────────────────────────────────────────────────────────────

setup_musetalk() {
    echo ""
    echo "═══ Enhancement #2: MuseTalk v1.5 (Real-Time Lip-Sync) ═══"

    clone_if_missing \
        "https://github.com/TMElyralab/MuseTalk.git" \
        "$EXT_DEPS/MuseTalk"

    mkdir -p "$MODEL_ROOT/musetalk"
    info "MuseTalk models are auto-downloaded on first run via HuggingFace"
    info "Or manually: huggingface-cli download TMElyralab/MuseTalk --local-dir $MODEL_ROOT/musetalk"
    ok "MuseTalk setup complete"
}

setup_emotion() {
    echo ""
    echo "═══ Enhancement #1: Emotion-Aware Expressions ═══"

    mkdir -p "$MODEL_ROOT/audio2emotion"

    # NVIDIA Audio2Emotion (optional, best quality)
    info "NVIDIA Audio2Emotion is optional. If available, download from:"
    info "  https://huggingface.co/nvidia/Audio2Face-3D"
    info "  Place audio2emotion.onnx in $MODEL_ROOT/audio2emotion/"

    # Fallback: text-based emotion detection auto-downloads on first use
    info "Text-based emotion fallback (j-hartmann/emotion-english-distilroberta-base)"
    info "  auto-downloads via HuggingFace transformers on first use"

    # Audio prosody fallback needs no models
    ok "Emotion detection setup complete (prosody fallback always available)"
}

setup_eye_gaze() {
    echo ""
    echo "═══ Enhancement #3: Eye Gaze + Blink Modeling ═══"

    # No models needed — uses OpenCV built-in cascades and procedural generation
    info "No additional models needed"
    info "Uses OpenCV Haar cascades (bundled) + procedural blink/saccade generation"
    ok "Eye gaze/blink setup complete"
}

setup_liveportrait() {
    echo ""
    echo "═══ Enhancement #4: LivePortrait ═══"

    clone_if_missing \
        "https://github.com/KwaiVGI/LivePortrait.git" \
        "$EXT_DEPS/LivePortrait"

    mkdir -p "$MODEL_ROOT/liveportrait"
    info "LivePortrait models auto-download on first run"
    info "Or manually download pretrained weights to $MODEL_ROOT/liveportrait/"
    ok "LivePortrait setup complete"
}

setup_latentsync() {
    echo ""
    echo "═══ Enhancement #5: LatentSync ═══"

    clone_if_missing \
        "https://github.com/bytedance/LatentSync.git" \
        "$EXT_DEPS/LatentSync"

    mkdir -p "$MODEL_ROOT/latentsync"
    info "LatentSync checkpoints: download from HuggingFace"
    info "  huggingface-cli download chunyu-li/LatentSync --local-dir $MODEL_ROOT/latentsync"
    ok "LatentSync setup complete"
}

setup_hallo3() {
    echo ""
    echo "═══ Enhancement #6: Hallo3 (Cinematic) ═══"

    clone_if_missing \
        "https://github.com/fudan-generative-vision/hallo3.git" \
        "$EXT_DEPS/hallo3"

    mkdir -p "$MODEL_ROOT/hallo3"
    info "Hallo3 models: download pretrained_models from the repo README"
    ok "Hallo3 setup complete"
}

setup_cosyvoice() {
    echo ""
    echo "═══ Enhancement #7: CosyVoice TTS ═══"

    clone_if_missing \
        "https://github.com/FunAudioLLM/CosyVoice.git" \
        "$EXT_DEPS/CosyVoice"

    info "CosyVoice can run as a server or locally"
    info "See $EXT_DEPS/CosyVoice/README.md for model download instructions"
    ok "CosyVoice setup complete"
}

setup_viseme() {
    echo ""
    echo "═══ Enhancement #8: Viseme-Guided Rendering ═══"

    # Uses existing MFA data — no additional setup needed
    info "Uses existing viseme_align.py + MFA data (if available)"
    info "No additional models needed"
    ok "Viseme-guided setup complete"
}

setup_gesture() {
    echo ""
    echo "═══ Enhancement #9: GestureLSM ═══"

    # GestureLSM repo (when available)
    info "GestureLSM (ICCV 2025) — check for repo availability:"
    info "  https://github.com/andypinxinliu/GestureLSM"
    mkdir -p "$MODEL_ROOT/gesturelsm"
    info "Procedural idle animation fallback always available (no models needed)"
    ok "Gesture animation setup complete"
}

setup_gaussian() {
    echo ""
    echo "═══ Enhancement #10: 3D Gaussian Splatting (InsTaG) ═══"

    info "InsTaG (CVPR 2025) — check for repo availability"
    mkdir -p "$MODEL_ROOT/instag"
    mkdir -p "$MODEL_ROOT/gaussian_cache"
    info "3D Gaussian models are built per-avatar from calibration video"
    ok "Gaussian splatting setup complete"
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Avatar Renderer MCP — Enhancement Model Setup                 ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Project Root:  $PROJECT_ROOT"
echo "  Model Root:    $MODEL_ROOT"
echo "  External Deps: $EXT_DEPS"

# Parse arguments
TARGETS=()
if [ $# -eq 0 ] || [ "$1" = "--top3" ]; then
    TARGETS=(emotion eye_gaze musetalk)
    echo "  Mode: Top 3 (emotion + eye_gaze + musetalk)"
elif [ "$1" = "--all" ]; then
    TARGETS=(emotion musetalk eye_gaze liveportrait latentsync hallo3 cosyvoice viseme gesture gaussian)
    echo "  Mode: All enhancements"
else
    TARGETS=("$@")
    echo "  Mode: Custom (${TARGETS[*]})"
fi

echo ""

for target in "${TARGETS[@]}"; do
    case "$target" in
        emotion|emotion_expressions)  setup_emotion ;;
        musetalk|musetalk_lipsync)    setup_musetalk ;;
        eye_gaze|eye_gaze_blink)      setup_eye_gaze ;;
        liveportrait|liveportrait_driver) setup_liveportrait ;;
        latentsync|latentsync_lipsync) setup_latentsync ;;
        hallo3|hallo3_cinematic)       setup_hallo3 ;;
        cosyvoice|cosyvoice_tts)       setup_cosyvoice ;;
        viseme|viseme_guided)          setup_viseme ;;
        gesture|gesture_animation)     setup_gesture ;;
        gaussian|gaussian_splatting)   setup_gaussian ;;
        *)
            warn "Unknown enhancement: $target"
            warn "Available: emotion, musetalk, eye_gaze, liveportrait, latentsync, hallo3, cosyvoice, viseme, gesture, gaussian"
            ;;
    esac
done

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Setup complete! To verify, run:"
echo "    python -c 'from app.enhancements import registry; print([e.get_info() for e in registry.list_all()])'"
echo "═══════════════════════════════════════════════════════════════════"
