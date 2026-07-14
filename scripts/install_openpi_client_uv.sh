#!/usr/bin/env bash
set -euo pipefail

PIPER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$PIPER_ROOT/.." && pwd)"
OPENPI_CLIENT_DIR="${OPENPI_CLIENT_DIR:-$REPO_ROOT/docker_vla_share_clean/workspace/openpi_vla_proj/packages/openpi-client}"

cd "$PIPER_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 127
fi

if [ ! -d "$OPENPI_CLIENT_DIR" ]; then
  echo "openpi-client directory not found: $OPENPI_CLIENT_DIR" >&2
  exit 1
fi

uv pip install -e "$OPENPI_CLIENT_DIR"
