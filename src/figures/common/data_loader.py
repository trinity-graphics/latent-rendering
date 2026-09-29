"""
Resolve run directories and image paths given the run output layout:

    <input_dir>/
    └── <run_name>/
        ├── metrics.jsonl
        ├── debug/
        │   └── <variant>_XXXX.{png,exr}
        ├── move_camera/
        │   └── images/
        │       └── <variant>_XXXX.{png,exr}
        ├── move_light/
        │   └── images/
        └── move_object/
            └── images/

Standardised SVG href scheme (used in templates):
  runs/{scene}/{test_type}/{frame}/{variant}.png      → test-view flat image
  debug/{scene}/{frame}/{variant}.png                 → training/debug flat image
  latents/{scene}/{test_type}/{frame}/{ltype}/ch{N}   → test latent channel PNG
  debug_latents/{scene}/{frame}/{ltype}/ch{N}         → debug latent channel PNG
  crops/{scene}/{test_type}/{frame}/{ltype}/ch{N}_b{M}→ latent crop PNG

scene is the lowercase scene label (cbox, lamp, …); the resolver uses
SCENE_TO_RUN to map to the actual run directory name.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Scene label → run directory name
# ---------------------------------------------------------------------------

SCENE_TO_RUN: dict[str, str] = {
    "cbox":       "CBox-1024-Best",
    "lamp":       "Lamp-1024-Best",
    "living":     "Living-1024-Best",
    "dining":     "Dining-1024-Best",
    "veach":      "Veach-1024-Best",
    "lamp_naive": "Lamp-1024-Naive",
    "bedroom":    "Bedroom-1024-Best",
    "living2":    "Living2-1024-Best",
    "classroom":  "Classroom-1024-Best",
    # VAE comparison (outputs/VAE/): same cbox scene + full method, one run
    # per VAE.  Ablation variants (-NoAmb/-NoOcc) are deliberately not mapped.
    "vae_flux2":  "VAE/Flux-Cbox",
    "vae_flux1":  "VAE/Flux1-Cbox",
    "vae_qwen":   "VAE/Qwen-Cbox",
}

# ---------------------------------------------------------------------------
# Variant → filename template for test-view images
# Path: <run_dir>/<test_type>/images/<filename>
# ---------------------------------------------------------------------------
_TEST_VARIANTS: dict[str, str] = {
    "final":             "decoded_final_{frame:04d}.png",
    "post_ambient":      "decoded_post_ambient_{frame:04d}.png",
    "refined":           "decoded_refined_{frame:04d}.png",
    "best_rgb":          "best_rgb_{frame:04d}.png",
    "gt_rgb":            "gt_rgb_{frame:04d}.png",
    # EXR latent sources
    "latent_ref":        "latent_best_latent_{frame:04d}.exr",
    "latent_postambient":"latent_post_ambient_{frame:04d}.exr",
    "refined_exr":       "latent_refined_{frame:04d}.exr",
    "final_latent":      "latent_final_{frame:04d}.exr",
}

# Variant → filename template for training/debug images
# Path: <run_dir>/debug/<filename>
_DEBUG_VARIANTS: dict[str, str] = {
    "img":              "decoded_final_{frame:04d}.png",
    "final":            "decoded_final_{frame:04d}.png",
    "post_ambient":     "decoded_post_ambient_{frame:04d}.png",
    "prerefine":        "decoded_processed_{frame:04d}.png",
    # EXR latent sources
    "latent_postambient": "latent_post_ambient_{frame:04d}.exr",
    "final_latent":     "latent_final_{frame:04d}.exr",
}

# Latent EXR type name (as used in href ltype segment) → filename template
LATENT_TYPE_TO_FILE: dict[str, str] = {
    "latent_ref":        "latent_best_latent_{frame:04d}.exr",
    "latent_postambient":"latent_post_ambient_{frame:04d}.exr",
    "refined":           "latent_refined_{frame:04d}.exr",
    "final_latent":      "latent_final_{frame:04d}.exr",
}

# Reference-view latents, indexed by view rather than by frame.
# Path: <run_dir>/views/<filename>
VIEW_LATENT_TYPE_TO_FILE: dict[str, str] = {
    "latent_sample":     "latent_sample_{index}.exr",
    "latent_processed":  "latent_processed_{index}.exr",
}


# ---------------------------------------------------------------------------
# Low-level path resolvers
# ---------------------------------------------------------------------------

def _run_dir(input_dir: Path, run_or_scene: str) -> Path:
    """Return the run directory for a scene label or a literal run name."""
    name = SCENE_TO_RUN.get(run_or_scene, run_or_scene)
    return input_dir / name


def resolve_test_image(
    input_dir: Path | str,
    run_name: str,
    test_type: str,
    frame_id: int,
    variant: str = "final",
) -> Path:
    """Return the absolute path for a test-view image."""
    tmpl = _TEST_VARIANTS[variant]
    filename = tmpl.format(frame=frame_id)
    return (Path(input_dir) / run_name / test_type / "images" / filename).resolve()


def resolve_debug_image(
    input_dir: Path | str,
    run_name: str,
    variant: str = "final",
    frame_id: int = 0,
) -> Path:
    """Return the absolute path for a training/debug image."""
    tmpl = _DEBUG_VARIANTS[variant]
    filename = tmpl.format(frame=frame_id)
    return (Path(input_dir) / run_name / "debug" / filename).resolve()


def resolve_test_latent_exr(
    input_dir: Path | str,
    run_name: str,
    test_type: str,
    frame_id: int,
    ltype: str,
) -> Path:
    """Return the EXR path for a test-view latent by ltype name."""
    tmpl = LATENT_TYPE_TO_FILE[ltype]
    filename = tmpl.format(frame=frame_id)
    return (Path(input_dir) / run_name / test_type / "images" / filename).resolve()


def resolve_debug_latent_exr(
    input_dir: Path | str,
    run_name: str,
    frame_id: int,
    ltype: str,
) -> Path:
    """Return the EXR path for a debug latent by ltype name."""
    tmpl = LATENT_TYPE_TO_FILE[ltype]
    filename = tmpl.format(frame=frame_id)
    return (Path(input_dir) / run_name / "debug" / filename).resolve()


# ---------------------------------------------------------------------------
# Kept for backward compatibility (no longer required by figure scripts)
# ---------------------------------------------------------------------------

def load_best_csv(input_dir: Path | str) -> pd.DataFrame:
    path = Path(input_dir) / "best.csv"
    df = pd.read_csv(path, skipinitialspace=True)
    df["frame_id"] = df["frame_id"].astype(int)
    return df


def get_best_frame(df: pd.DataFrame, run_name: str, test_type: str) -> int:
    mask = (df["run_name"] == run_name) & (df["test_type"] == test_type)
    rows = df[mask]
    if rows.empty:
        raise KeyError(f"No best.csv entry for run={run_name!r}, test_type={test_type!r}")
    return int(rows.iloc[0]["frame_id"])


def list_runs(input_dir: Path | str) -> list[str]:
    excluded = {"figures", "output", "outputs"}
    return [
        d.name for d in sorted(Path(input_dir).iterdir())
        if d.is_dir() and not d.name.startswith(".") and d.name not in excluded
    ]
