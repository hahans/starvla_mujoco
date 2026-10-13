#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON_BIN="${PYTHON_BIN:-/home/czj/anaconda3/envs/hdp/bin/python}"
SOURCE="${1:-../pocketvla_mujoco/data/block_transfer2}"
OUTPUT="${2:-data/block_lerobot}"
exec "$PYTHON_BIN" -m mujoco_task.convert --source "$SOURCE" --output "$OUTPUT"
