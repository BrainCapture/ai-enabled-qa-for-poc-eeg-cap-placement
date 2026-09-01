#!/usr/bin/env bash
set -euo pipefail

# Determine project root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  source .env
fi
KEY_FILE="${GCP_SA_KEY:-$ROOT_DIR/google-service-account-key.json}"

# Check for Google service account key
if [ ! -f "$KEY_FILE" ]; then
  echo "Error: '$KEY_FILE' not found."
  echo "Please add your service account JSON (something-key.json) to the project root."
  exit 1
fi
echo "Found service account key."

# Ensure Taskfile is installed
if ! command -v task >/dev/null 2>&1; then
  echo "Taskfile CLI not found. Installing..."
  curl -sSL https://taskfile.dev/install.sh | sh
  echo "Taskfile installed."
else
  echo "Taskfile is already installed."
fi

# Run an initial “help” task
echo "Running initial 'help' check..."
task help

echo "Setup complete! You can now run 'task sync'."