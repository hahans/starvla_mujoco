"""Validate preview channel order and preserve lossless training observations."""

import hashlib
import json
import shutil
import subprocess

import numpy as np
import pytest

from mujoco_task.convert import encode_video
from mujoco_task.data import read_video
from mujoco_task.preview import make_preview


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg/ffprobe required")
def test_preview_color_order_metadata_and_source_unchanged(tmp_path):
    frames = np.zeros((4, 64, 96, 3), dtype=np.uint8)
    frames[:, :, :32] = [255, 0, 0]
    frames[:, :, 32:64] = [0, 255, 0]
    frames[:, :, 64:] = [0, 0, 255]
    source, output = tmp_path / "lossless.mp4", tmp_path / "preview.mp4"
    encode_video(source, frames, 20)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    make_preview(source, output)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    np.testing.assert_array_equal(read_video(source), frames)
    decoded = read_video(output)
    assert decoded.shape == frames.shape
    for channel, x in enumerate((16, 48, 80)):
        pixel = decoded[0, 32, x].astype(int)
        assert pixel[channel] > 245
        assert np.delete(pixel, channel).max() < 10
    info = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=pix_fmt,color_space,color_range",
                "-of",
                "json",
                str(output),
            ],
            text=True,
        )
    )["streams"][0]
    assert info == {"pix_fmt": "yuv420p", "color_range": "tv", "color_space": "smpte170m"}
    with pytest.raises(FileExistsError):
        make_preview(source, output)
