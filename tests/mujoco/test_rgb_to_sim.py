"""Check camera channel order, geometry, fixed calibration and source integrity."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mujoco_task.rgb_to_sim import Transform, convert_rgb, fit_reference, load_profile, prepare_rgb, read_rgb

ROOT = Path(__file__).resolve().parents[2]


def test_identity_and_rgb_channel_order():
    rng = np.random.default_rng(4)
    rgb = rng.integers(0, 256, (15, 21, 3), dtype=np.uint8)
    np.testing.assert_array_equal(convert_rgb(rgb, Transform(smooth=0)), rgb)
    colors = np.asarray([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    np.testing.assert_array_equal(convert_rgb(colors, Transform()), colors)


def test_filter_reduces_noise_without_blending_across_strong_edge():
    rng = np.random.default_rng(8)
    clean = np.zeros((48, 64, 3), dtype=float)
    clean[:, :32] = [180, 20, 20]
    clean[:, 32:] = [20, 180, 20]
    noisy = np.clip(clean + rng.normal(0, 4, clean.shape), 0, 255).astype(np.uint8)
    result = convert_rgb(noisy, Transform(smooth=1))
    assert np.mean((result.astype(float) - clean) ** 2) < np.mean((noisy.astype(float) - clean) ** 2) * 0.4
    assert result[:, 31, 0].mean() > 170
    assert result[:, 32, 1].mean() > 170
    assert result[:, 31, 1].mean() < 30


def test_reference_fit_corrects_known_color_offset_and_handles_flat_channels():
    ramp = np.linspace(30, 190, 32 * 32).reshape(32, 32)
    source = np.stack([ramp, np.flipud(ramp), np.full_like(ramp, 100)], axis=-1).astype(np.uint8)
    reference = np.rint(source.astype(float) * [1.1, 0.9, 1] + [6, 10, 8]).astype(np.uint8)
    cfg = fit_reference(source, reference, Transform(smooth=0), strength=1)
    result = convert_rgb(source, cfg)
    assert np.abs(result.astype(float) - reference).mean() < 1
    assert cfg.color_gain[2] == 1
    assert np.isfinite(result).all()


def test_explicit_crop_resize_and_no_silent_stretch():
    rgb = np.zeros((20, 40, 3), dtype=np.uint8)
    rgb[:, :20, 0] = 255
    rgb[:, 20:, 1] = 255
    cropped = convert_rgb(rgb, Transform(crop=(20, 0, 20, 20), size=(10, 10), smooth=0))
    np.testing.assert_array_equal(cropped, np.broadcast_to([0, 255, 0], (10, 10, 3)))
    with pytest.raises(ValueError, match="aspect ratio"):
        prepare_rgb(rgb, Transform(size=(128, 128)))
    with pytest.raises(ValueError, match="exceeds"):
        prepare_rgb(rgb, Transform(crop=(25, 0, 20, 20)))


@pytest.mark.parametrize("kwargs", [{"gamma": 0}, {"smooth": 2}, {"exposure": float("nan")}, {"wb_gains": [1, 0, 1]}])
def test_invalid_transform_rejected(kwargs):
    with pytest.raises(ValueError):
        Transform(**kwargs)


def test_exposure_gamma_and_white_balance_direction():
    gray = np.full((6, 8, 3), 100, dtype=np.uint8)
    assert np.all(convert_rgb(gray, Transform(exposure=1, smooth=0)) > gray)
    assert np.all(convert_rgb(gray, Transform(gamma=1.2, smooth=0)) > gray)
    balanced = convert_rgb(gray, Transform(wb_gains=(1.2, 1, 0.8), smooth=0))
    assert balanced[0, 0, 0] > balanced[0, 0, 1] > balanced[0, 0, 2]


def test_reject_float_and_alpha_instead_of_guessing(tmp_path):
    with pytest.raises(ValueError, match="uint8"):
        convert_rgb(np.zeros((2, 2, 3), dtype=float), Transform())
    path = tmp_path / "alpha.png"
    Image.new("RGBA", (8, 8)).save(path)
    with pytest.raises(ValueError, match="RGB first"):
        read_rgb(path)


def test_cli_batch_freezes_profile_and_preserves_sources(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    Image.new("RGB", (16, 16), (100, 110, 120)).save(source / "frame.png")
    Image.new("RGB", (16, 16), (20, 210, 30)).save(source / "frame.jpg")
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.iterdir()}
    reference = tmp_path / "ref.png"
    Image.new("RGB", (16, 16), (100, 100, 100)).save(reference)
    output = tmp_path / "output"
    command = [sys.executable, "-m", "mujoco_task.rgb_to_sim", "--input", str(source), "--output", str(output)]
    subprocess.run([*command, "--reference", str(reference)], cwd=ROOT, check=True, capture_output=True)
    cfg = load_profile(output / "profile.json")
    records = json.loads((output / "manifest.json").read_text())
    assert len(records) == 2
    assert len({record["output"] for record in records}) == 2
    for record in records:
        path = Path(record["source"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[path]
        np.testing.assert_array_equal(read_rgb(output / record["output"]), convert_rgb(read_rgb(path), cfg))
    replay = tmp_path / "replay"
    subprocess.run(
        [*command[:-1], str(replay), "--profile", str(output / "profile.json")],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    for record in records:
        assert (output / record["output"]).read_bytes() == (replay / record["output"]).read_bytes()
    result = subprocess.run(command, cwd=ROOT, capture_output=True)
    assert result.returncode != 0
    assert b"Output must not exist" in result.stderr
