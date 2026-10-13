"""The Mesa launcher must preserve CUDA selection and stop on render failure."""

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/mujoco/evaluate_mesa_egl.sh"


@pytest.fixture
def launcher_env(tmp_path):
    recorder = tmp_path / "python recorder"
    recorder.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "if sys.argv[1:] == ['-']:\n"
        "    sys.stdin.read()\n"
        "    sys.exit(int(os.environ.get('TEST_PREFLIGHT_EXIT', '0')))\n"
        "print(json.dumps({'args': sys.argv[1:], 'env': dict(os.environ)}))\n"
    )
    recorder.chmod(0o755)
    vendor = tmp_path / "mesa vendor.json"
    vendor.write_text("{}")
    return {
        **os.environ,
        "PYTHON_BIN": str(recorder),
        "MESA_EGL_VENDOR_JSON": str(vendor),
        "CUDA_VISIBLE_DEVICES": "5",
        "MUJOCO_EGL_DEVICE_ID": "5",
        "MUJOCO_GL": "osmesa",
        "PYOPENGL_PLATFORM": "osmesa",
    }


def test_preserves_model_options_and_cuda_device(launcher_env):
    args = ["--checkpoint", "/tmp/weights with spaces/model.pt", "--torch-dtype", "float32"]
    result = subprocess.run(["bash", str(SCRIPT), *args], env=launcher_env, capture_output=True, text=True, check=True)
    record = json.loads(result.stdout)
    assert record["args"] == ["-m", "mujoco_task.evaluate", *args]
    env = record["env"]
    assert env["CUDA_VISIBLE_DEVICES"] == "5"
    assert "MUJOCO_EGL_DEVICE_ID" not in env
    assert env["MUJOCO_GL"] == env["PYOPENGL_PLATFORM"] == "egl"
    assert env["__EGL_VENDOR_LIBRARY_FILENAMES"] == launcher_env["MESA_EGL_VENDOR_JSON"]
    assert env["LIBGL_ALWAYS_SOFTWARE"] == "1"
    assert env["GALLIUM_DRIVER"] == "llvmpipe"


def test_failed_renderer_check_prevents_model_launch(launcher_env):
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        env={**launcher_env, "TEST_PREFLIGHT_EXIT": "7"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 7
    assert result.stdout == ""


def test_missing_mesa_does_not_fall_back_to_nvidia(launcher_env):
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        env={**launcher_env, "MESA_EGL_VENDOR_JSON": "/nonexistent/mesa.json"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Mesa EGL vendor file not found" in result.stderr
    assert result.stdout == ""
