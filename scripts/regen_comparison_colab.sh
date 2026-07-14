#!/usr/bin/env bash
# =============================================================================
# regen_comparison_colab.sh — same as regen_comparison.sh, but the renders run on
# the temporary **Colab GPU job server** (so every engine, incl. premium, runs on
# a real GPU). Metrics, GIFs, dataset upload and the README rewrite happen LOCALLY.
#
# Prereqs: the Colab notebook (colab/Avatar_Renderer_GPU_Server.ipynb) is running
# and you've exported the two values it printed:
#   export COLAB_GPU_URL='https://....trycloudflare.com'
#   export COLAB_GPU_TOKEN='...'
#
# Usage:
#   scripts/regen_comparison_colab.sh                       # render + metrics + README
#   UPLOAD=1 HF_TOKEN=hf_xxx scripts/regen_comparison_colab.sh   # also push to dataset
#
# Knobs (env, optional): TEXT VOICE QUALITY OUT_DIR DATASET ENGINES UPLOAD
#   UPDATE_README BRANCH WAIT_TRIES PULL(0|1)
# =============================================================================
set -euo pipefail

: "${COLAB_GPU_URL:?export COLAB_GPU_URL from the Colab notebook}"
: "${COLAB_GPU_TOKEN:?export COLAB_GPU_TOKEN from the Colab notebook}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
CLIENT="scripts/claude_colab_client.sh"

TEXT="${TEXT:-Hello! Welcome to the avatar renderer. Watch how natural this looks now.}"
VOICE="${VOICE:-en-US-AriaNeural}"
QUALITY="${QUALITY:-high_quality}"
OUT_DIR="${OUT_DIR:-/tmp/comparison_colab}"
DATASET="${DATASET:-ruslanmv/avatar-renderer-samples}"
UPDATE_README="${UPDATE_README:-1}"
UPLOAD="${UPLOAD:-auto}"
BRANCH="${BRANCH:-claude/project-review-9nn71}"
WAIT_TRIES="${WAIT_TRIES:-300}"   # 300 x 5s = 25 min/job (premium first run is slow)
PULL="${PULL:-1}"
mkdir -p "$OUT_DIR"

jqpy() { python3 -c "import sys,json;d=json.load(sys.stdin);print(d$1)" 2>/dev/null || true; }

# ── 0) Preflight: the tunnel + server must be reachable and return JSON ───────
ping="$(curl -sS --max-time 20 "$COLAB_GPU_URL/ping" 2>/dev/null || true)"
if ! printf '%s' "$ping" | grep -q avatar-renderer-colab-gpu; then
    echo "ERROR: Colab server not reachable at $COLAB_GPU_URL"
    echo "  The Cloudflare tunnel likely dropped (Error 1033) or the runtime disconnected."
    echo "  Re-run the notebook's tunnel cell, then re-export COLAB_GPU_URL/COLAB_GPU_TOKEN."
    exit 2
fi

# ── 1) Make the Colab checkout current, so all engines/render_one exist ───────
if [ "$PULL" = "1" ]; then
    echo "[colab] git pull $BRANCH"
    jid="$($CLIENT POST /git/pull "{\"branch\":\"$BRANCH\"}" | jqpy "['job_id']")"
    [ -n "$jid" ] && $CLIENT wait "$jid" 60 >/dev/null || true
fi

echo "[colab] /health"; $CLIENT GET /health; echo
AVAIL_JSON="$($CLIENT GET /engines)"
if ! printf '%s' "$AVAIL_JSON" | python3 -c 'import sys,json;json.load(sys.stdin)' 2>/dev/null; then
    echo "ERROR: /engines did not return JSON (tunnel/runtime issue). Aborting."; exit 2
fi
echo "[colab] available engines: $(printf '%s' "$AVAIL_JSON" | jqpy "['available']")"

# ── 1) Variant manifest: "label|engine|quality|star|caption" (same as local) ──
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

is_available() {  # engine -> 0 if available
    printf '%s' "$AVAIL_JSON" | python3 -c "
import sys,json
eng=sys.argv[1]; d=json.load(sys.stdin)
ok = eng=='simple' or eng in d.get('available',[])
sys.exit(0 if ok else 1)" "$1"
}

# ── 2) Render each variant on the Colab GPU; download the mp4 ─────────────────
TSV="$OUT_DIR/variants.tsv"; : > "$TSV"
for v in "${VARIANTS[@]}"; do
    IFS='|' read -r label engine quality star caption <<<"$v"
    if [ -n "${ENGINES:-}" ] && ! grep -qw "$label" <<<"$ENGINES"; then continue; fi
    if ! is_available "$engine"; then echo "[skip] $label ($engine not available on Colab)"; continue; fi

    payload="$(python3 -c "import json,sys;print(json.dumps({
        'engine':sys.argv[1],'quality_mode':sys.argv[2],'text':sys.argv[3],
        'voice':sys.argv[4],'use_sample':True,'timeout_sec':1500}))" \
        "$engine" "$quality" "$TEXT" "$VOICE")"

    echo "[render] $label  <-  engine=$engine quality=$quality (on Colab GPU)"
    resp="$($CLIENT POST /render/sample "$payload")"
    jid="$(printf '%s' "$resp" | jqpy "['job_id']")"
    art="$(printf '%s' "$resp" | jqpy "['expected_artifact']")"
    if [ -z "$jid" ]; then echo "[fail] $label submit failed: $resp"; continue; fi

    final="$($CLIENT wait "$jid" "$WAIT_TRIES")"
    status="$(printf '%s' "$final" | jqpy "['status']")"
    if [ "$status" = "done" ]; then
        $CLIENT DOWNLOAD "/artifact/$art" "$OUT_DIR/$label.mp4"
        printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$engine" "$quality" "$star" "$caption" >> "$TSV"
    else
        echo "[fail] $label status=$status — log tail:"
        printf '%s' "$final" | jqpy "['log_tail'][-800:]"
    fi
done

[ -s "$TSV" ] || { echo "no variants rendered on Colab — check /engines + /jobs logs"; exit 1; }

# ── 3) Metrics + GIFs + (upload) + README rewrite — all LOCAL ─────────────────
do_upload=""
case "$UPLOAD" in
    auto) [ -n "${HF_TOKEN:-}" ] && do_upload="--upload" ;;
    1)    do_upload="--upload" ;;
esac
update_flag=""; [ "$UPDATE_README" = "1" ] && update_flag="--update-readme"

python scripts/comparison_report.py --dir "$OUT_DIR" --variants "$TSV" \
    --dataset "$DATASET" --readme README.md $update_flag $do_upload

echo
echo "Done. GPU renders downloaded to $OUT_DIR (mp4 + gif + metrics.json)."
[ -n "$do_upload" ] && echo "Uploaded to https://huggingface.co/datasets/$DATASET"
[ "$UPDATE_README" = "1" ] && echo "README 'Quality Comparison' block updated — review the diff."
