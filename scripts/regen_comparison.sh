#!/usr/bin/env bash
# =============================================================================
# regen_comparison.sh — reproduce + update the README "Generation Methods —
# Quality Comparison" section, locally, on a machine WITH a GPU.
#
# Renders every available engine from the SAME portrait + SAME text, measures
# metrics, builds GIFs, (optionally) uploads samples to the HF dataset, and
# rewrites the README block between <!-- COMPARISON:START/END -->.
#
# Usage:
#   scripts/regen_comparison.sh                      # render + metrics + README
#   UPLOAD=1 HF_TOKEN=hf_xxx scripts/regen_comparison.sh   # also push to dataset
#
# Knobs (env vars, all optional):
#   PORTRAIT   portrait image            (default: zerogpu/assets/alice.png)
#   TEXT       narration text (TTS)      (default: a short welcome line)
#   VOICE      edge-tts voice            (default: en-US-AriaNeural)
#   AUDIO      use this audio instead of TTS (skips edge-tts)
#   QUALITY    tier for lip-sync engines (default: high_quality)
#   ENGINES    space list of LABELS to limit to (default: all in the manifest)
#   OUT_DIR    work/output dir           (default: /tmp/comparison)
#   DATASET    HF dataset repo           (default: ruslanmv/avatar-renderer-samples)
#   UPLOAD     0|1|auto                  (default: auto = upload iff HF_TOKEN set)
#   UPDATE_README 0|1                    (default: 1)
#
# Prereqs: a GPU + the engines installed (run scripts/colab_gpu_setup.sh once, or
# scripts/download_enhancements.sh + download_models.sh). Unavailable engines are
# skipped automatically, so it also works with just the in-process Wav2Lip set.
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

export MODEL_ROOT="${MODEL_ROOT:-$REPO_DIR/models}"
export EXT_DEPS_DIR="${EXT_DEPS_DIR:-$REPO_DIR/external_deps}"
export PYTHONPATH="$REPO_DIR:$EXT_DEPS_DIR/first-order-model:$EXT_DEPS_DIR/Wav2Lip:$EXT_DEPS_DIR/SadTalker:$EXT_DEPS_DIR/Diff2Lip:$EXT_DEPS_DIR/guided-diffusion:$EXT_DEPS_DIR/MuseTalk:$EXT_DEPS_DIR/LatentSync:${PYTHONPATH:-}"
export LIPSYNC_HF_REPO="${LIPSYNC_HF_REPO:-ruslanmv/avatar-renderer}"

PORTRAIT="${PORTRAIT:-zerogpu/assets/alice.png}"
TEXT="${TEXT:-Hello! Welcome to the avatar renderer. Watch how natural this looks now.}"
VOICE="${VOICE:-en-US-AriaNeural}"
QUALITY="${QUALITY:-high_quality}"
OUT_DIR="${OUT_DIR:-/tmp/comparison}"
DATASET="${DATASET:-ruslanmv/avatar-renderer-samples}"
UPDATE_README="${UPDATE_README:-1}"
UPLOAD="${UPLOAD:-auto}"
AUDIO="${AUDIO:-}"

[ -f "$PORTRAIT" ] || { echo "portrait not found: $PORTRAIT (set PORTRAIT=...)"; exit 1; }
mkdir -p "$OUT_DIR"

# ── 1) Audio: explicit AUDIO wins, else synthesize TEXT with edge-tts ─────────
if [ -z "$AUDIO" ]; then
    AUDIO="$OUT_DIR/audio.wav"
    echo "[tts] synthesizing narration -> $AUDIO"
    python - "$TEXT" "$VOICE" "$AUDIO" <<'PY'
import sys, asyncio, edge_tts
text, voice, out = sys.argv[1:4]
asyncio.run(edge_tts.Communicate(text, voice).save(out))
PY
fi
echo "[inputs] portrait=$PORTRAIT  audio=$AUDIO"

# ── 2) Variant manifest: "label|engine|quality|star|caption" ─────────────────
# star=1 marks the recommended column. Premium engines are included and skipped
# automatically if not installed/available.
VARIANTS=(
    "simple|simple|preview|0|No lip-sync (static)"
    "wav2lip|wav2lip_raw|standard|0|Full-face, no restore"
    "wav2lip_gfpgan|wav2lip_fast|$QUALITY|1|Full-face + GFPGAN"
    "wav2lip_band|wav2lip_band|$QUALITY|0|Mouth-band, static base"
    "fullface|fullface|$QUALITY|0|+ head motion, static bg"
    "diff2lip|diff2lip|$QUALITY|0|Diffusion lip-sync"
    "musetalk|musetalk|$QUALITY|0|Real-time latent lip-sync"
    "latentsync|latentsync|$QUALITY|0|Diffusion (premium)"
)

# Engine availability snapshot from the registry (JSON: {name: bool}).
AVAIL_JSON="$(python - <<'PY'
import json, os, sys
sys.path.insert(0, os.getcwd())
from app.engines import registry
print(json.dumps({e["name"]: bool(e["available"]) for e in registry.info_all()}))
PY
)"

# ── 3) Render loop ───────────────────────────────────────────────────────────
TSV="$OUT_DIR/variants.tsv"; : > "$TSV"
for v in "${VARIANTS[@]}"; do
    IFS='|' read -r label engine quality star caption <<<"$v"
    # Optional ENGINES filter (by label).
    if [ -n "${ENGINES:-}" ] && ! grep -qw "$label" <<<"$ENGINES"; then continue; fi
    # Availability: simple always works; otherwise consult the registry.
    avail="$(python - "$engine" "$AVAIL_JSON" <<'PY'
import json, sys
eng, avail = sys.argv[1], json.loads(sys.argv[2])
print("1" if eng == "simple" or avail.get(eng, False) else "0")
PY
)"
    if [ "$avail" != "1" ]; then echo "[skip] $label ($engine not available here)"; continue; fi

    out="$OUT_DIR/$label.mp4"
    echo "[render] $label  <-  engine=$engine quality=$quality"
    if python colab/render_one.py --image "$PORTRAIT" --audio "$AUDIO" \
            --out "$out" --engine "$engine" --quality "$quality"; then
        printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$engine" "$quality" "$star" "$caption" >> "$TSV"
    else
        echo "[fail]  $label excluded (render error)"
    fi
done

[ -s "$TSV" ] || { echo "no variants rendered — check engines/weights"; exit 1; }

# ── 4) Decide upload, then metrics + GIFs + (upload) + README rewrite ────────
do_upload=""
case "$UPLOAD" in
    auto) [ -n "${HF_TOKEN:-}" ] && do_upload="--upload" ;;
    1)    do_upload="--upload" ;;
esac
update_flag=""; [ "$UPDATE_README" = "1" ] && update_flag="--update-readme"

python scripts/comparison_report.py --dir "$OUT_DIR" --variants "$TSV" \
    --dataset "$DATASET" --readme README.md $update_flag $do_upload

echo
echo "Done. Artifacts in $OUT_DIR (mp4 + gif + metrics.json + comparison_block.md)."
[ -n "$do_upload" ] && echo "Samples uploaded to https://huggingface.co/datasets/$DATASET"
[ "$UPDATE_README" = "1" ] && echo "README 'Quality Comparison' block updated — review the diff before committing."
