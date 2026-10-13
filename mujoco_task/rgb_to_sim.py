"""Conservative RGB preprocessing with optional fixed reference color matching.

This approximates appearance, not a renderer or a geometric camera transform.
Only NumPy and Pillow are required; arrays at the public interface are RGB uint8.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Transform:
    """Fixed parameters shared across frames; crop uses x, y, width, height."""

    crop: tuple[int, int, int, int] | None = None
    size: tuple[int, int] | None = None  # width, height; no implicit aspect change
    wb_gains: tuple[float, float, float] = (1.0, 1.0, 1.0)
    exposure: float = 0.0  # EV in linear-light RGB
    gamma: float = 1.0  # output = input ** (1 / gamma)
    saturation: float = 1.0
    smooth: float = 0.25
    color_gain: tuple[float, float, float] = (1.0, 1.0, 1.0)
    color_bias: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        """Reject invalid or extreme values before touching output files."""
        for name, length in (("crop", 4), ("size", 2)):
            value = getattr(self, name)
            if value is not None and (
                len(value) != length or any(isinstance(v, bool) or not isinstance(v, int) for v in value)
            ):
                raise ValueError(f"{name} must contain {length} integers")
        if self.crop and (min(self.crop[:2]) < 0 or min(self.crop[2:]) <= 0):
            raise ValueError("crop requires x,y >= 0 and width,height > 0")
        if self.size and min(self.size) <= 0:
            raise ValueError("size must be positive")
        for name in ("wb_gains", "color_gain", "color_bias"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (3,) or not np.isfinite(values).all():
                raise ValueError(f"{name} must contain three finite values")
            if name != "color_bias" and (np.any(values <= 0) or np.any(values > 4)):
                raise ValueError(f"{name} must be in (0, 4]")
        for name, low, high in (("exposure", -4, 4), ("gamma", 0.1, 4), ("saturation", 0, 2), ("smooth", 0, 1)):
            value = getattr(self, name)
            if not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be finite and in [{low}, {high}]")


def check_rgb(rgb: np.ndarray) -> None:
    """Require explicit RGB bytes rather than guessing dtype or channel order."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError("Expected nonempty H x W x 3 RGB uint8 (not BGR or float)")


def read_rgb(path: Path) -> np.ndarray:
    """Read RGB/sRGB camera imagery, applying EXIF orientation if present.

    Embedded non-sRGB ICC profiles are not converted. Export such inputs to
    sRGB first. Alpha images are rejected instead of silently changing a scene.
    """
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode not in ("RGB", "L"):
            raise ValueError(f"{path}: export as 8-bit RGB first (got {image.mode})")
        return np.asarray(image.convert("RGB")).copy()


def prepare_rgb(rgb: np.ndarray, cfg: Transform) -> np.ndarray:
    """Apply explicit crop and aspect-preserving resize; never infer a viewpoint."""
    check_rgb(rgb)
    if cfg.crop:
        x, y, width, height = cfg.crop
        if x + width > rgb.shape[1] or y + height > rgb.shape[0]:
            raise ValueError(f"crop {cfg.crop} exceeds input width/height {rgb.shape[1::-1]}")
        rgb = rgb[y : y + height, x : x + width]
    if cfg.size:
        width, height = cfg.size
        if width * rgb.shape[0] != height * rgb.shape[1]:
            raise ValueError("Resize would distort aspect ratio; use --crop X Y W H to select the matching view first")
        rgb = np.asarray(Image.fromarray(rgb).resize((width, height), Image.Resampling.LANCZOS))
    return rgb.copy()


def _manual_color(rgb: np.ndarray, cfg: Transform) -> np.ndarray:
    value = rgb.astype(np.float32) / 255.0
    if cfg.exposure != 0 or tuple(cfg.wb_gains) != (1.0, 1.0, 1.0):
        linear = np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)
        linear = np.clip(linear * np.asarray(cfg.wb_gains) * 2.0**cfg.exposure, 0, 1)
        value = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)
    if cfg.gamma != 1:
        value = value ** (1 / cfg.gamma)
    if cfg.saturation != 1:
        luminance = value @ np.asarray([0.2126, 0.7152, 0.0722])
        value = luminance[..., None] + cfg.saturation * (value - luminance[..., None])
    return np.clip(value, 0, 1).astype(np.float32)


def _bilateral(value: np.ndarray) -> np.ndarray:
    """Small RGB bilateral filter; intensity sigma 20/255 and spatial sigma 1.5."""
    height, width = value.shape[:2]
    padded = np.pad(value, ((2, 2), (2, 2), (0, 0)), mode="edge")
    total = np.zeros_like(value)
    weights = np.zeros((height, width), dtype=np.float32)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            neighbor = padded[2 + dy : 2 + dy + height, 2 + dx : 2 + dx + width]
            distance = np.sum((neighbor - value) ** 2, axis=-1)
            weight = np.exp(-distance / (2 * (20 / 255) ** 2) - (dx * dx + dy * dy) / (2 * 1.5**2))
            total += neighbor * weight[..., None]
            weights += weight
    return total / weights[..., None]


def fit_reference(source: np.ndarray, reference: np.ndarray, cfg: Transform, strength: float = 0.35) -> Transform:
    """Estimate bounded, global RGB correction once; no spatial correspondence.

    A reference should show similar objects/background proportions. Quantiles
    cannot separate illuminant changes from changes in the scene composition.
    """
    check_rgb(reference)
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("reference strength must be in [0, 1]")
    source_value = _manual_color(prepare_rgb(source, cfg), cfg).reshape(-1, 3)
    ref_value = reference.reshape(-1, 3).astype(np.float32) / 255.0
    src = np.quantile(source_value, [0.1, 0.5, 0.9], axis=0)
    ref = np.quantile(ref_value, [0.1, 0.5, 0.9], axis=0)
    # Near-flat channels cannot identify a contrast slope: only adjust offset.
    spread = src[2] - src[0]
    gain = np.clip((ref[2] - ref[0]) / np.maximum(spread, 0.04), 0.8, 1.25)
    gain[spread < 0.04] = 1
    bias = np.clip(ref[1] - gain * src[1], -0.06, 0.06)
    return replace(cfg, color_gain=tuple(1 + strength * (gain - 1)), color_bias=tuple(strength * bias))


def convert_rgb(rgb: np.ndarray, cfg: Transform) -> np.ndarray:
    """Convert one RGB camera frame using fixed parameters; output RGB uint8."""
    value = _manual_color(prepare_rgb(rgb, cfg), cfg)
    value = np.clip(value * np.asarray(cfg.color_gain) + np.asarray(cfg.color_bias), 0, 1).astype(np.float32)
    if cfg.smooth:
        value = (1 - cfg.smooth) * value + cfg.smooth * _bilateral(value)
    return np.clip(np.rint(value * 255), 0, 255).astype(np.uint8)


def load_profile(path: Path) -> Transform:
    """Load a profile produced by this CLI for batch or live camera use."""
    payload = json.loads(Path(path).read_text())
    if payload.get("version") != 1:
        raise ValueError("Unsupported RGB conversion profile version")
    return Transform(**payload["transform"])


def main() -> None:
    """Convert an image or recursive image directory, preserving source files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="One image or a directory of images")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    calibration = parser.add_mutually_exclusive_group()
    calibration.add_argument("--reference", type=Path, help="OSMesa reference; fit once on the first input image")
    calibration.add_argument("--profile", type=Path, help="Reuse a previously saved fixed profile")
    parser.add_argument("--strength", type=float, default=0.35, help="Reference correction strength, 0..1")
    parser.add_argument("--crop", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--size", nargs=2, type=int, metavar=("W", "H"), help="Optional resize; same aspect required")
    parser.add_argument("--wb-gains", nargs=3, type=float, metavar=("R", "G", "B"))
    parser.add_argument("--exposure", type=float, help="Exposure adjustment in EV, default 0")
    parser.add_argument("--gamma", type=float, help="Values >1 brighten, default 1")
    parser.add_argument("--saturation", type=float, help="Default 1; 0 is grayscale")
    parser.add_argument("--smooth", type=float, help="Bilateral blend 0..1, default 0.25; 0 disables")
    args = parser.parse_args()
    overrides = {
        name: getattr(args, name)
        for name in ("crop", "size", "wb_gains", "exposure", "gamma", "saturation", "smooth")
        if getattr(args, name) is not None
    }
    if args.profile and overrides:
        parser.error("--profile already defines processing parameters; omit parameter overrides")
    if not args.input.exists():
        parser.error(f"Input does not exist: {args.input}")
    if args.output.exists():
        parser.error(f"Output must not exist: {args.output}")
    if args.input.is_dir() and args.output.resolve().is_relative_to(args.input.resolve()):
        parser.error("Output must be outside the input directory")
    paths = (
        [args.input]
        if args.input.is_file()
        else sorted(path for path in args.input.rglob("*") if path.is_file() and path.suffix.lower() in EXTENSIONS)
    )
    if not paths:
        parser.error("No supported images found")
    cfg = load_profile(args.profile) if args.profile else Transform(**overrides)
    calibration_info = None
    if args.reference:
        cfg = fit_reference(read_rgb(paths[0]), read_rgb(args.reference), cfg, args.strength)
        calibration_info = {
            "source": str(paths[0].resolve()),
            "reference": str(args.reference.resolve()),
            "source_sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest(),
            "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
            "strength": args.strength,
        }
    args.output.mkdir(parents=True)
    profile = {
        "version": 1,
        "transform": asdict(cfg),
        "calibration": calibration_info,
        "inherited_profile": str(args.profile.resolve()) if args.profile else None,
    }
    (args.output / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    records = []
    for path in paths:
        rgb = read_rgb(path)
        prepared = prepare_rgb(rgb, cfg)
        converted = convert_rgb(rgb, cfg)
        relative = Path(path.name) if args.input.is_file() else path.relative_to(args.input)
        # Preserve original suffix to distinguish frame.jpg from frame.png.
        relative = relative.with_name(relative.name + ".png")
        output = args.output / "images" / relative
        comparison = args.output / "comparisons" / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        comparison.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(converted).save(output)
        Image.fromarray(np.concatenate([prepared, converted], axis=1)).save(comparison)
        records.append(
            {
                "source": str(path.resolve()),
                "output": str(output.relative_to(args.output)),
                "comparison": str(comparison.relative_to(args.output)),
                "input_shape": list(rgb.shape),
                "output_shape": list(converted.shape),
                "change_mae_8bit": float(np.abs(prepared.astype(float) - converted).mean()),
            }
        )
        print(output.resolve(), flush=True)
    (args.output / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print("Comparisons: left = input after crop/resize; right = processed. No grasping performance claim.")


if __name__ == "__main__":
    main()
