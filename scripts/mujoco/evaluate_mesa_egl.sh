#!/usr/bin/env bash
# Evaluate OSMesa-trained weights using Mesa's EGL interface and CPU renderer.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
task_python="${PYTHON_BIN:-${PYTHON:-python}}"
task_vendor_json="${MESA_EGL_VENDOR_JSON:-/usr/share/glvnd/egl_vendor.d/50_mesa.json}"
if [[ ! -f "$task_vendor_json" ]]; then
    echo "Mesa EGL vendor file not found: $task_vendor_json" >&2
    echo "Set MESA_EGL_VENDOR_JSON to an existing Mesa EGL vendor JSON." >&2
    exit 1
fi

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES="$task_vendor_json"
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
# Device indices from NVIDIA EGL are not valid in the Mesa-only device list.
unset MUJOCO_EGL_DEVICE_ID
export USE_TF=0 USE_FLAX=0 STARVLA_DISABLE_DEEPSPEED=1

# Fail before loading the VLA if the requested renderer is unavailable.
"$task_python" - <<'PY'
from mujoco_task.env import TransferEnv
from OpenGL import GL

env = TransferEnv()
try:
    env.reset(1)
    renderer = GL.glGetString(GL.GL_RENDERER).decode()
    version = GL.glGetString(GL.GL_VERSION).decode()
    print(f"Mesa EGL renderer: {renderer}; {version}", flush=True)
    print(f"Actual MSAA samples: {env.renderer._mjr_context.offSamples}", flush=True)
    if "llvmpipe" not in renderer.lower():
        raise RuntimeError(f"Expected llvmpipe, got {renderer}")
finally:
    env.close()
PY

exec "$task_python" -m mujoco_task.evaluate "$@"
