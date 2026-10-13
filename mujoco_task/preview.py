"""Export player-compatible previews without modifying lossless training videos."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .data import episode_path, read_json


def make_preview(source: Path, destination: Path) -> None:
    """Convert RGB H.264 to BT.601 YUV420 H.264 for ordinary video players.

    Preview chroma subsampling is lossy; these files must not replace the
    original training observations. The original video is never modified.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists() or source == destination:
        raise FileExistsError(f"Refusing to overwrite {destination}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("Install ffmpeg to export compatible previews")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(suffix=".mp4", prefix=".preview-", dir=destination.parent)
    os.close(fd)
    try:
        subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-an",
                "-vf",
                "scale=in_range=full:out_range=tv:out_color_matrix=bt601,format=yuv420p",
                "-c:v",
                "libx264",
                "-profile:v",
                "high",
                "-crf",
                "18",
                "-color_range",
                "tv",
                "-colorspace",
                "smpte170m",
                "-color_primaries",
                "smpte170m",
                "-color_trc",
                "smpte170m",
                "-movflags",
                "+faststart",
                temporary,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        # Atomic creation without overwriting an existing destination.
        os.link(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> None:
    """Export one episode or all episodes from one split to a separate directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/block_lerobot"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--all", action="store_true", help="Export all episodes of the selected split")
    parser.add_argument("--output", type=Path, default=Path("outputs/video_preview"))
    args = parser.parse_args()
    root = args.data_root / f"block_{args.split}"
    count = read_json(root / "meta/info.json")["total_episodes"]
    if not 0 <= args.episode < count:
        parser.error(f"episode must be between 0 and {count - 1}")
    for episode in range(count) if args.all else [args.episode]:
        destination = args.output / f"{args.split}_episode_{episode:06d}.mp4"
        make_preview(episode_path(root, episode, video=True), destination)
        print(destination.resolve(), flush=True)


if __name__ == "__main__":
    main()
