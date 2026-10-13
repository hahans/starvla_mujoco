#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="${1:-$(dirname "$ROOT")/starvla_mujoco_code.tar.gz}"
# Includes modified upstream files, task code, tests, and portable MuJoCo assets.
# Datasets, environments, caches and model weights are transferred separately.
tar -czf "$OUTPUT" --exclude='.git' --exclude='.venv*' --exclude='__pycache__' \
  --exclude='.pytest_cache' --exclude='.ruff_cache' --exclude='data' --exclude='outputs' \
  --exclude='logs' --exclude='wandb' \
  --exclude='playground' --exclude='*.egg-info' \
  -C "$(dirname "$ROOT")" "$(basename "$ROOT")"
printf '%s\n' "$OUTPUT"
