#!/usr/bin/env python3
"""Plot GRPO training and fixed-seed validation success from metrics.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def _load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No metrics in {path}")
    return rows


def _baseline_rate(metrics_path: Path) -> float | None:
    checkpoint = metrics_path.parent / "best_checkpoint.json"
    if not checkpoint.is_file():
        return None
    payload = json.loads(checkpoint.read_text())
    baseline = payload.get("baseline_metrics") or {}
    if "success_rate" in baseline:
        return float(baseline["success_rate"])
    successes = baseline.get("successes")
    episodes = baseline.get("episodes")
    if successes is None or not episodes:
        return None
    return float(successes) / len(episodes)


def _use_cjk_font() -> None:
    candidates = (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"),
    )
    for path in candidates:
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = "Noto Sans CJK JP"
            return


def plot_success(metrics_path: Path, output_path: Path) -> None:
    """Write one figure with rollout success and greedy validation success."""
    rows = _load_rows(metrics_path)
    iterations = [int(row["iteration"]) for row in rows]
    train = [float(row["success_rate"]) for row in rows]
    validation = [
        (int(row["iteration"]), float(row["validation"]["successes"]) / len(row["validation"]["seeds"]))
        for row in rows
        if "validation" in row
    ]
    baseline = _baseline_rate(metrics_path)
    _use_cjk_font()
    plt.rcParams["axes.unicode_minus"] = False

    figure, axis = plt.subplots(figsize=(10, 4.5), dpi=140)
    axis.plot(iterations, train, color="#4C78A8", linewidth=1.2, label="采样成功率")
    if validation:
        val_x, val_y = zip(*validation, strict=True)
        axis.plot(val_x, val_y, color="#E45756", marker="o", linewidth=1.4, label="固定评估成功率")
        best_iteration, best_rate = max(validation, key=lambda item: (item[1], -item[0]))
        axis.scatter([best_iteration], [best_rate], color="#E45756", s=46, zorder=3)
        axis.annotate(
            f"最佳 {best_iteration}: {best_rate:.0%}",
            (best_iteration, best_rate),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
        )
    if baseline is not None:
        axis.axhline(baseline, color="#54A24B", linestyle="--", linewidth=1.1, label=f"SFT 基线 {baseline:.0%}")
    axis.set_xlabel("迭代")
    axis.set_ylabel("成功率")
    axis.set_ylim(-0.02, 1.02)
    axis.set_title(metrics_path.parent.name)
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend(loc="upper left")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metrics",
        nargs="?",
        type=Path,
        default=Path("outputs/mujoco_nb2_lr2e5/metrics.jsonl"),
        help="metrics.jsonl written by mujoco_task.grpo",
    )
    parser.add_argument("--output", type=Path, help="PNG path. Defaults to success.png beside the metrics file.")
    args = parser.parse_args()
    output = args.output or args.metrics.parent / "success.png"
    plot_success(args.metrics, output)
    print(output)


if __name__ == "__main__":
    main()
