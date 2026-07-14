#!/usr/bin/env bash
set -euo pipefail

CONFIG="${PIPER_VLA_CONFIG:-configs/safety.example.yaml}"
ROBOT_CAMERA_SOURCE="${PIPER_VLA_ROBOT_CAMERA_SOURCE:-}"
OVERHEAD_CAMERA_SOURCE="${PIPER_VLA_OVERHEAD_CAMERA_SOURCE:-}"

ARGS=(--config "$CONFIG" --dry-run)
if [[ -n "$ROBOT_CAMERA_SOURCE" ]]; then
	ARGS+=(--robot-camera-source "$ROBOT_CAMERA_SOURCE")
fi
if [[ -n "$OVERHEAD_CAMERA_SOURCE" ]]; then
	ARGS+=(--overhead-camera-source "$OVERHEAD_CAMERA_SOURCE")
fi

python -m piper_vla_guard.ui_app "${ARGS[@]}"
