#!/usr/bin/env bash
# =============================================================================
# claude_colab_client.sh — local client for the temporary Colab GPU server.
#
# Requires two env vars (printed by the Colab notebook):
#   COLAB_GPU_URL    e.g. https://something.trycloudflare.com
#   COLAB_GPU_TOKEN  the per-session secret
#
# Usage:
#   scripts/claude_colab_client.sh GET  /health
#   scripts/claude_colab_client.sh GET  /engines
#   scripts/claude_colab_client.sh POST /git/pull        '{"branch":"claude/project-review-9nn71"}'
#   scripts/claude_colab_client.sh POST /gpu/check
#   scripts/claude_colab_client.sh POST /setup           '{"engines":["musetalk","diff2lip"]}'
#   scripts/claude_colab_client.sh POST /render/sample   '{"engine":"musetalk","quality_mode":"high_quality","text":"Hi"}'
#   scripts/claude_colab_client.sh GET  /jobs/<job_id>
#   scripts/claude_colab_client.sh DOWNLOAD /artifact/<name> out.mp4
#
# Convenience: `wait <job_id>` polls until the job leaves "running".
# =============================================================================
set -euo pipefail

: "${COLAB_GPU_URL:?Set COLAB_GPU_URL to the trycloudflare/ngrok URL}"
: "${COLAB_GPU_TOKEN:?Set COLAB_GPU_TOKEN to the per-session token}"

H_AUTH=(-H "X-Colab-Token: ${COLAB_GPU_TOKEN}")
H_JSON=(-H "Content-Type: application/json")

method="${1:?GET|POST|DOWNLOAD|wait}"; shift

case "$method" in
  GET)
    curl -sS "${H_AUTH[@]}" "${COLAB_GPU_URL}${1}" ;;
  POST)
    endpoint="$1"; payload="${2:-{}}"
    curl -sS -X POST "${COLAB_GPU_URL}${endpoint}" "${H_AUTH[@]}" "${H_JSON[@]}" -d "${payload}" ;;
  DOWNLOAD)
    endpoint="$1"; outfile="${2:?provide an output file path}"
    curl -sS "${H_AUTH[@]}" "${COLAB_GPU_URL}${endpoint}" -o "${outfile}"
    echo "saved ${outfile}" ;;
  wait)
    jid="${1:?provide a job_id}"; tries="${2:-120}"
    for ((i=0; i<tries; i++)); do
      resp="$(curl -sS "${H_AUTH[@]}" "${COLAB_GPU_URL}/jobs/${jid}")"
      status="$(printf '%s' "$resp" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("status",""))' 2>/dev/null || echo "")"
      if [ "$status" != "running" ]; then printf '%s\n' "$resp"; exit 0; fi
      sleep 5
    done
    echo "timed out waiting for job ${jid}" >&2; exit 1 ;;
  *)
    echo "unknown method: $method (use GET|POST|DOWNLOAD|wait)" >&2; exit 2 ;;
esac
