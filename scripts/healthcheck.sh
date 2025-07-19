#!/usr/bin/env bash
#
# scripts/healthcheck.sh
#
# Simple liveness/readiness probe for the Avatar Renderer Pod.
# Returns 0 if the service is healthy, non-zero otherwise.

HOST="${HOST:-localhost}"
PORT="${PORT:-8080}"
ENDPOINT="${ENDPOINT:-/avatars}"

URL="http://${HOST}:${PORT}${ENDPOINT}"

# Try up to 5 times with a 2s interval
for i in $(seq 1 5); do
  if curl -sf "$URL" >/dev/null; then
    echo "✅ Service healthy at $URL"
    exit 0
  else
    echo "⚠️  Attempt $i: service not responding yet, retrying..."
    sleep 2
  fi
done

echo "❌ Service failed health check at $URL"
exit 1
