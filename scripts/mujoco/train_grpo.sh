#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export USE_TF=0 USE_FLAX=0 STARVLA_DISABLE_DEEPSPEED=1
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export TOKENIZERS_PARALLELISM=false
exec "${PYTHON:-python}" -m mujoco_task.grpo "$@"
