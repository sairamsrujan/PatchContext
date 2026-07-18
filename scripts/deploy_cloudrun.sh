#!/usr/bin/env bash
# Deploy PatchContext to Google Cloud Run (always-on-capable, scales to zero
# when idle, free within the trial/free-tier quota for light traffic).
#
# One-time prerequisites (only you can do these — they need your Google
# account / browser / billing card):
#   1. Create a GCP project at https://console.cloud.google.com/projectcreate
#   2. Enable billing on it (Free Trial: $300 / 90 days) at
#      https://console.cloud.google.com/billing
#   3. Run:  gcloud auth login        (opens a browser to sign in)
#   4. Run:  gcloud config set project <YOUR_PROJECT_ID>
#
# Then just run this script:  ./scripts/deploy_cloudrun.sh
#
# Cost controls baked in: --min-instances 0 (no charge while idle, the
# container spins down and cold-starts on the next request) and
# --max-instances 1 (caps how much a traffic spike could ever cost).

set -euo pipefail
cd "$(dirname "$0")/.."

SERVICE=patchcontext
REGION=us-central1

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud CLI not found. Install: brew install --cask google-cloud-sdk" >&2
  exit 1
fi

PROJECT=$(gcloud config get-value project 2>/dev/null || true)
if [ -z "$PROJECT" ] || [ "$PROJECT" = "(unset)" ]; then
  echo "No gcloud project configured. Run:" >&2
  echo "  gcloud auth login" >&2
  echo "  gcloud config set project <YOUR_PROJECT_ID>" >&2
  exit 1
fi
echo "deploying to project: $PROJECT"

echo "enabling required APIs (safe to re-run) …"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

# Build an env-vars file from .env (never printed, never committed) so keys
# don't need to be typed on the command line.
ENV_YAML=$(mktemp)
trap 'rm -f "$ENV_YAML"' EXIT
{
  echo "MODEL_DEVICE: cpu"
  for var in LLM_API_KEY LLM_FALLBACK_API_KEY LLM_BASE_URL LLM_FALLBACK_BASE_URL LLM_MODEL LLM_FALLBACK_MODEL RAGAS_JUDGE_API_KEY; do
    val=$(grep -E "^${var}=" .env 2>/dev/null | head -1 | cut -d= -f2-)
    [ -n "$val" ] && echo "${var}: \"${val}\""
  done
} > "$ENV_YAML"

echo "building + deploying (Cloud Build; first deploy takes ~10-15 min) …"
gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --memory 8Gi \
  --cpu 2 \
  --cpu-boost \
  --timeout 600 \
  --min-instances 0 \
  --max-instances 1 \
  --allow-unauthenticated \
  --env-vars-file "$ENV_YAML" \
  --quiet

URL=$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')
echo ""
echo "=== LIVE URL (stable, always the same): $URL ==="
