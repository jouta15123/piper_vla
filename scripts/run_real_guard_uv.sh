#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 127
fi

CONFIG="${PIPER_VLA_CONFIG:-configs/safety.example.yaml}"
HOST="${PIPER_VLA_HOST:-127.0.0.1}"
PORT="${PIPER_VLA_PORT:-7860}"
CAN="${PIPER_VLA_CAN:-can0}"
ROBOT_CAMERA_SOURCE="${PIPER_VLA_ROBOT_CAMERA_SOURCE:-}"
OVERHEAD_CAMERA_SOURCE="${PIPER_VLA_OVERHEAD_CAMERA_SOURCE:-}"

ARGS=(--config "$CONFIG" --server-name "$HOST" --server-port "$PORT" --can "$CAN" --real-hardware --dry-run)
if [[ -n "$ROBOT_CAMERA_SOURCE" ]]; then
  ARGS+=(--robot-camera-source "$ROBOT_CAMERA_SOURCE")
fi
if [[ -n "$OVERHEAD_CAMERA_SOURCE" ]]; then
  ARGS+=(--overhead-camera-source "$OVERHEAD_CAMERA_SOURCE")
fi

UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" uv run --extra piper --extra vision --extra openpi piper-vla-guard-ui "${ARGS[@]}"
