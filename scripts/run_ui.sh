#!/usr/bin/env bash
set -euo pipefail
python -m piper_vla_guard.ui_app --config configs/safety.example.yaml --dry-run
