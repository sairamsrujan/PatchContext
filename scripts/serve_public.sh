#!/usr/bin/env bash
# Bring PatchContext up on a free public URL via a Cloudflare quick tunnel.
#
# Runs the Docker image locally and exposes it at a https://*.trycloudflare.com
# URL — free, no account, no card. The URL is printed once the tunnel is up and
# is reachable while this script (and your machine) stays running.
#
# Prereqs: Docker Desktop running, cloudflared installed (`brew install cloudflared`),
# a .env with LLM_API_KEY / LLM_FALLBACK_API_KEY, and the image built once:
#     docker build -t patchcontext:local .
#
# Usage:  ./scripts/serve_public.sh     (Ctrl-C to stop; container is cleaned up)

set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE=patchcontext:local
NAME=pc_live
PORT=8501

cleanup() { echo; echo "stopping…"; docker rm -f "$NAME" >/dev/null 2>&1 || true; kill "${TUNNEL_PID:-0}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "starting container ($IMAGE) on :$PORT …"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --env-file .env -e MODEL_DEVICE=cpu -p "$PORT:8501" "$IMAGE" >/dev/null

# wait for the app's health endpoint
for _ in $(seq 1 30); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/_stcore/health" 2>/dev/null)" = "200" ] && break
  sleep 2
done
echo "app healthy. opening public tunnel …"

cloudflared tunnel --url "http://localhost:$PORT" 2>&1 | while read -r line; do
  echo "$line"
  case "$line" in
    *trycloudflare.com*) echo ">>> Public URL is above (https://…trycloudflare.com) <<<" ;;
  esac
done
