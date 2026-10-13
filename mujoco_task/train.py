"""Compatibility notice: the former task-specific training loop is retired."""

if __name__ == "__main__":
    raise SystemExit(
        "The custom trainer has been retired. Use bash scripts/mujoco/train.sh "
        "(StarVLA's native train_starvla.py), or python -m mujoco_task.check_data "
        "to validate data without training. See README_MUJOCO_ZH.md."
    )
