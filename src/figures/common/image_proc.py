"""
Image processing utilities for figure generation.

Consolidates logic from:
  - latentrender-figures/normalize_latent.py  (latent EXR → PNG)
  - latentrender-figures/paper_figures_v2.py  (gamma/exposure, crops, boxes, error maps)
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyexr
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Low-level loaders
# ---------------------------------------------------------------------------

def open_image(path: Path | str) -> np.ndarray:
    """Load any supported image format → float32 HxWxC array in [0, ∞)."""
    path = str(path)
    ext = path.rsplit(".", 1)[-1].lower()
    if ext == "exr":
        return pyexr.read(path).astype(np.float32)
    if ext == "hdr":
        import cv2
        fp = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        return cv2.cvtColor(fp, cv2.COLOR_BGR2RGB).astype(np.float32)
    if ext in ("png", "jpg", "jpeg"):
        im = Image.open(path).convert("RGB")
        arr = np.array(im, dtype=np.float32) / 255.0
        im.close()
        return arr
    raise ValueError(f"Unsupported format: {path}")


# ---------------------------------------------------------------------------
# Tone mapping / colour transforms
# ---------------------------------------------------------------------------

def apply_gamma_exposure(
    img: np.ndarray,
    gamma: float = 2.2,
    exposure: float = 0.0,
) -> np.ndarray:
    """Apply exposure (EV stops) then gamma, return float32 in [0, 1]."""
    img = img.copy()
    if exposure != 0.0:
        img *= 2.0 ** exposure
    if gamma != 1.0:
        img = np.power(np.clip(img, 0.0, None), 1.0 / gamma)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 255, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Crop / box annotations (ported from paper_figures_v2.py)
# ---------------------------------------------------------------------------

def draw_crop_box(
    img: Image.Image,
    x: int,
    y: int,
    w: int,
    h: int,
    color: tuple[int, int, int] = (238, 53, 56),
    border_px: int = 2,
) -> Image.Image:
    """Draw a coloured rectangle on `img` (in-place copy) marking a crop region."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle(
        [x - border_px, y - border_px, x + w + border_px - 1, y + h + border_px - 1],
        outline=color,
        width=border_px,
    )
    return img


def crop_region(
    img: Image.Image,
    x: int,
    y: int,
    w: int,
    h: int,
    border_px: int = 0,
    color: tuple[int, int, int] = (238, 53, 56),
) -> Image.Image:
    """Crop a region from `img`, optionally adding a coloured border."""
    region = img.crop((x - border_px, y - border_px,
                       x + w + border_px, y + h + border_px))
    if border_px > 0:
        bordered = Image.new("RGB", region.size, color)
        inner = img.crop((x, y, x + w, y + h))
        bordered.paste(inner, (border_px, border_px))
        return bordered
    return region


# ---------------------------------------------------------------------------
# Error / false-colour maps
# ---------------------------------------------------------------------------

def _lum(img: np.ndarray) -> np.ndarray:
    return 0.21268 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


def compute_error_map(
    pred: np.ndarray,
    ref: np.ndarray,
    metric: str = "flip",
    eps: float = 1e-2,
) -> np.ndarray:
    """Return a 2-D float32 error map between pred and ref (HxWx3 arrays).

    metric: 'l1' | 'l2' | 'mrse' | 'mape' | 'smape' | 'flip'
    """
    pred = np.clip(pred, 0.0, None)
    ref  = np.clip(ref,  0.0, None)
    diff = ref - pred
    if metric == "flip":
        import flip_evaluator
        return flip_evaluator.evaluate(ref.copy(), pred.copy(),
                                       dynamicRangeString="HDR")[0]
    p = _lum(pred)
    r = _lum(ref)
    d = _lum(ref) - _lum(pred)
    if metric == "l1":
        return np.abs(d).astype(np.float32)
    if metric == "l2":
        return (d * d).astype(np.float32)
    if metric == "mrse":
        return (d * d / (r * r + eps)).astype(np.float32)
    if metric == "mape":
        return (np.abs(d) / (r + eps)).astype(np.float32)
    if metric == "smape":
        err = 2 * np.abs(d) / (r + p + eps)
        err[r == 0] = 0.0
        return err.astype(np.float32)
    raise ValueError(f"Unknown metric: {metric!r}")


def apply_false_color(
    arr: np.ndarray,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap_name: str = "batlow",
) -> np.ndarray:
    """Map a 2-D float array to an HxWx4 uint8 RGBA image via cmcrameri."""
    from cmcrameri import cm as cmc
    cmap = getattr(cmc, cmap_name)
    arr = np.clip((arr - vmin) / (vmax - vmin + 1e-12), 0, 1)
    return np.clip(cmap(arr) * 255, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Latent EXR → PNG (ported from normalize_latent.py)
# ---------------------------------------------------------------------------

def _to_posneg(
    channel: np.ndarray,
    sign_normalization: bool = True,
    clip_max: float = 5.0,
) -> np.ndarray:
    """Split a single-channel latent map into [neg, pos, 0] RGB."""
    pos = channel.clip(min=0)
    neg = (-channel).clip(min=0)
    if sign_normalization:
        pos = pos.clip(max=clip_max) / clip_max
        neg = neg.clip(max=clip_max) / clip_max
    return np.stack([neg, pos, np.zeros_like(pos)], axis=-1)


def latent_abs_percentile(exr_path: Path | str, q: float = 99.0) -> float:
    """Return the q-th percentile of |value| over every channel of a latent EXR.

    Different VAEs put their latents on wildly different scales (FLUX.2 ≈ ±0.8,
    FLUX.1 ≈ ±3.7), so a shared fixed clip_max renders some of them nearly black.
    Use this as a per-file clip_max to normalise contrast across models while
    preserving each channel's relative amplitude within its own model.
    """
    keys = sorted(pyexr.open(str(exr_path)).channels)
    data = pyexr.read(str(exr_path), channels=keys)
    arr = np.stack([data[k][..., 0] for k in keys], axis=-1)
    return float(np.percentile(np.abs(arr), q))


def latent_channel_png(
    exr_path: Path | str,
    out_path: Path | str,
    channel_idx: int,
    scale: float = 1.0,
    upsample: int = 4,
    gamma: float = 2.2,
    clip_max: float = 5.0,
) -> None:
    """Convert a single EXR channel to a to_posneg PNG visualisation."""
    key = f"ch{str(channel_idx).zfill(2)}"
    data = pyexr.read(str(exr_path), channels=[key])
    ch = data[key][..., 0]  # (H, W)
    vis = _to_posneg(ch, clip_max=clip_max)
    if upsample > 1:
        vis = np.kron(vis, np.ones((upsample, upsample, 1)))
    vis = np.power(np.clip(vis, 0.0, 1.0), 1.0 / gamma) * scale
    img = Image.fromarray(to_uint8(vis), mode="RGB")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path))


def latent_exr_to_png(
    exr_path: Path | str,
    output_path: Path | str,
    channel_indices: Sequence[int] | None = None,
    scale: float = 1.0,
    upsample: int = 4,
    gamma: float = 2.2,
    add_ambient: bool = False,
) -> None:
    """Convert one EXR latent file into per-channel PNG visualisations.

    Each channel is saved as  <output_path_stem>_{i}.png  (one file per channel).
    channel_indices: which latent channels to export (default: all 16).
    """
    keys_all = [f"ch{str(i).zfill(2)}" for i in range(16)]
    if channel_indices is None:
        keys = keys_all
    else:
        keys = [keys_all[i] for i in channel_indices]

    data = pyexr.read(str(exr_path), channels=keys)

    # if add_ambient and "_processsed_" in str(exr_path):
    #     amb_path = str(exr_path).replace("_processsed_", "_ambient_")
    #     amb = pyexr.read(amb_path, channels=keys)
    #     for k in keys:
    #         data[k] = data[k] + amb[k]

    keys_sorted = sorted(data.keys())
    # (K, H, W, 1) → (K, H, W)
    latent = np.stack([data[k][..., 0] for k in keys_sorted], axis=0)

    out_stem = Path(output_path)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    for i, ch in enumerate(latent):
        vis = _to_posneg(ch)               # (H, W, 3)
        if upsample > 1:
            vis = np.kron(vis, np.ones((upsample, upsample, 1)))
        vis = np.power(np.clip(vis, 0.0, 1.0), 1.0 / gamma) * scale
        img = Image.fromarray(to_uint8(vis), mode="RGB")
        img.save(f"{out_stem}_{i}.png")
