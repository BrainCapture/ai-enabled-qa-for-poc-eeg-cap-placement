#!/usr/bin/env bash

set -euo pipefail

if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  source .env
fi

KEY_FILE_PATH="${GCP_SA_KEY:-${GOOGLE_APPLICATION_CREDENTIALS:-google-service-account-key.json}}"

if [[ ! -f "${KEY_FILE_PATH}" ]]; then
  echo "Service account key file not found at ${KEY_FILE_PATH}" >&2
  exit 1
fi

gcloud auth activate-service-account --key-file="${KEY_FILE_PATH}"

ACCESS_TOKEN="$(gcloud auth print-access-token)"
cat > ~/.netrc <<EOF
machine europe-west6-python.pkg.dev
  login oauth2accesstoken
  password ${ACCESS_TOKEN}
EOF
chmod 600 ~/.netrc
