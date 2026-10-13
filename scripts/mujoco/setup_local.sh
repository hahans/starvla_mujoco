#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
BASE_PYTHON="${BASE_PYTHON:-/home/czj/anaconda3/envs/hdp/bin/python}"
# Isolate Transformers changes; leave the user's collection environment intact.
"$BASE_PYTHON" -m venv --system-site-packages .venv-mujoco
.venv-mujoco/bin/python -m pip install 'transformers==4.57.0' 'qwen-vl-utils==0.0.14' \
  'accelerate==1.5.2' 'omegaconf==2.3.0' 'av==12.3.0' 'rich>=13' 'pyarrow>=14,<22' \
  'h5py>=3.10' 'scipy>=1.11' 'pytest>=8'
printf '%s\n' 'Activate with: source .venv-mujoco/bin/activate'
