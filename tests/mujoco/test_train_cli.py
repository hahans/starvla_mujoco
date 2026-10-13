"""Check native entrypoint/arguments without starting accelerate or training."""

import json
import os
import shutil
import subprocess
from pathlib import Path

from omegaconf import OmegaConf

from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/mujoco/qwenfast.yaml"


def test_bash_calls_official_trainer_and_preserves_cli_overrides(tmp_path):
    root = tmp_path / "repo"
    script = root / "scripts/mujoco/train.sh"
    script.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/mujoco/train.sh", script)
    recorder = tmp_path / "python recorder"
    recorder.write_text(
        "#!/usr/bin/env python3\nimport json,os,sys\nprint(json.dumps({'args':sys.argv[1:],'cuda':os.environ['CUDA_VISIBLE_DEVICES'],'cwd':os.getcwd()}))\n"
    )
    recorder.chmod(0o755)
    env = {**os.environ, "PYTHON_BIN": str(recorder), "NPROC": "2", "CONFIG": str(CONFIG), "CUDA_VISIBLE_DEVICES": "0,1"}
    result = subprocess.run(
        [
            "bash",
            str(script),
            "--trainer.max_train_steps",
            "123",
            "run_id=cli_override",
            "--framework.qwenvl.base_vlm",
            "/tmp/model with spaces",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    record = json.loads(next(line for line in result.stdout.splitlines() if line.startswith('{"args"')))
    arguments = record["args"]
    assert arguments[:2] == ["-m", "accelerate.commands.launch"]
    assert "starVLA/training/train_starvla.py" in arguments
    assert "mujoco_task.train" not in arguments
    assert record["cuda"] == "0,1"
    assert record["cwd"] == str(root)
    overrides = arguments[arguments.index("--config_yaml") + 2 :]
    cfg = OmegaConf.merge(OmegaConf.load(CONFIG), OmegaConf.from_dotlist(normalize_dotlist_args(overrides)))
    assert cfg.trainer.max_train_steps == 123
    assert cfg.trainer.gradient_accumulation_steps == 16
    assert cfg.run_id == "cli_override"
    assert cfg.framework.qwenvl.base_vlm == "/tmp/model with spaces"
    assert cfg.datasets.vla_data.dataset_py == "mujoco_lerobot"
    assert len(list((root / "logs").glob("*.log"))) == 1


def test_bash_propagates_training_failure(tmp_path):
    root = tmp_path / "repo"
    script = root / "scripts/mujoco/train.sh"
    script.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/mujoco/train.sh", script)
    stub = tmp_path / "fail-python"
    stub.write_text("#!/usr/bin/env bash\nexit 7\n")
    stub.chmod(0o755)
    result = subprocess.run(["bash", str(script)], env={**os.environ, "PYTHON_BIN": str(stub)}, capture_output=True)
    assert result.returncode == 7
