"""Download pinned pretrained assets; never starts model training."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from .data import write_json

ASSETS = {
    "fast": ("physical-intelligence/fast", "ec4d7aa71691cac0b8bed6942be45684db2110f4"),
    "Qwen2.5-VL-3B-Instruct-Action": (
        "StarVLA/Qwen2.5-VL-3B-Instruct-Action",
        "97163e6190ca87d6abae7a4dd15840cadde2da1d",
    ),
}


def main() -> None:
    """Fetch exact revisions, optionally only the small action tokenizer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("playground/Pretrained_models"))
    parser.add_argument("--fast-only", action="store_true")
    args = parser.parse_args()
    for name, (repo, revision) in ASSETS.items():
        if args.fast_only and name != "fast":
            continue
        target = args.output / name
        snapshot_download(
            repo,
            revision=revision,
            local_dir=str(target),
            allow_patterns=["*.json", "*.py", "*.txt", "*.model", "*.safetensors", "*.md", "*.jinja"],
        )
        write_json(target / "source_revision.json", {"repo": repo, "revision": revision})


if __name__ == "__main__":
    main()
