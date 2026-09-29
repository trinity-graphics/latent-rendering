import csv
import re
from copy import deepcopy
from pathlib import Path
from shutil import copy2

import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
import pandas as pd
import torch
from skimage.util import montage
from tqdm.auto import tqdm

mi.set_variant("scalar_rgb")

import utils.img_utils

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times"],
    "font.size": 21,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black"
})

def update_dict_keys(d: dict, prefix: str = "", suffix: str = ""):
    if len(prefix) > 0:
        prefix = prefix + "_"
    if len(suffix) > 0:
        suffix = "_" + suffix
    return {f"{prefix}{key}{suffix}": value for key, value in d.items()}

def generate_latent_metrics(exp_dir):
    TRACKED_METRICS = ["mse", "psnr", "ssim"]
    # latent_best_latent_i = reference latent (best quality, denoised)
    # latent_post_ambient_i = rendered latent (pre-refine)
    # latent_refined_i = refined latent

    with open(exp_dir / f"_{exp_dir.name}_latent.csv", 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["it", *[f"prerefine_{m}" for m in TRACKED_METRICS], *TRACKED_METRICS])

        for i in range(60):
            rendered_path = exp_dir / "images" / f"latent_post_ambient_{i:04d}.exr"
            refined_path = exp_dir / "images" / f"latent_refined_{i:04d}.exr"
            reference_path = exp_dir / "images" / f"latent_best_latent_{i:04d}.exr"

            if not rendered_path.exists() or not reference_path.exists():
                continue
            has_refined = refined_path.exists()

            csv_line = [i]

            lat_rendered = torch.as_tensor(mi.Bitmap(str(rendered_path))).permute(2, 0, 1).unsqueeze(0)
            lat_refined = torch.as_tensor(mi.Bitmap(str(refined_path))).permute(2, 0, 1).unsqueeze(0) if has_refined else None
            lat_reference = torch.as_tensor(mi.Bitmap(str(reference_path))).permute(2, 0, 1).unsqueeze(0)

            prerefine_metrics, _ = utils.img_utils.compute_image_metrics(lat_reference, lat_rendered, is_latent=True)
            csv_line.extend([prerefine_metrics[k] for k in TRACKED_METRICS])

            if has_refined:
                postrefine_metrics, _ = utils.img_utils.compute_image_metrics(lat_reference, lat_refined, is_latent=True)
                csv_line.extend([postrefine_metrics[k] for k in TRACKED_METRICS])
            else:
                csv_line.extend([None] * len(TRACKED_METRICS))

            writer.writerow(csv_line)


def plot_prerefine_vs_refine(csv_path: Path, out_dir: Path, metric: str, dpi: int = 300):
    prerefine_col = f"prerefine_{metric}"
    refine_col = metric

    title = " ".join(csv_path.stem[1:].split("_")).title()
    df = pd.read_csv(csv_path, skip_blank_lines=True)

    # Validate columns
    required_cols = {"it", prerefine_col, refine_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Compute DSSIM if metric is SSIM
    if metric.lower() == "ssim":
        df[prerefine_col] = 1 - df[prerefine_col]
        df[refine_col] = 1 - df[refine_col]
        ylabel = "DSSIM"
    else:
        ylabel = metric.upper()

    if metric == "lpips":
        suff = "+ decoded RGB"
    else:
        suff = "latent"

    # Ensure output directory exists
    out_dir.mkdir(exist_ok=True)

    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(df["it"], df[prerefine_col], label=f"Rendered {suff}", linewidth=2)
    plt.plot(df["it"], df[refine_col], label=f"Refined {suff}", linewidth=2)

    plt.xlabel("Frame", fontsize=21)
    plt.ylabel(ylabel, fontsize=21)
    plt.title(title, fontsize=21)
    plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=1, fontsize=21, frameon=True)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()

    out_path = out_dir / f"{csv_path.stem}_{metric}.pdf"
    plt.savefig(out_path, dpi=dpi, format="pdf")
    plt.close()

    return out_path


def get_exr(src_path, out_dir, target_channels = None):
    try:
        latent = torch.as_tensor(mi.Bitmap(str(src_path))).permute(2, 0, 1).unsqueeze(0)
    except RuntimeError:
        print(f"File at {src_path} not found, you may need to run the test cases again.")
        return

    # Make pos/neg color map
    latent_posneg = utils.img_utils.to_posneg(latent)
    latent_posneg = np.asarray(latent_posneg.squeeze(0).permute(0, 2, 3, 1))

    # Create 16-channel montage
    dst_path = out_dir / Path(src_path.name).with_suffix(".png")
    tmp_path = dst_path.with_name(f"{dst_path.stem}_tmp{dst_path.suffix}")
    latent_mtg = montage(latent_posneg, fill=(0, 0, 0), rescale_intensity=True, channel_axis=-1)
    out_bmp = mi.Bitmap(latent_mtg)
    out_bmp = out_bmp.convert(pixel_format=mi.Bitmap.PixelFormat.RGB, component_format=mi.Struct.Type.UInt8, srgb_gamma=False)
    out_bmp.write(str(tmp_path))
    utils.img_utils.upsample(str(tmp_path), str(dst_path), scale=8)
    tmp_path.unlink()

    if target_channels is not None:
        # Output individual channels if desired
        for target_channel in target_channels:
            dst_path = out_dir / Path(f"{src_path.stem}_ch{target_channel}").with_suffix(".png")
            tmp_path = dst_path.with_name(f"{dst_path.stem}_tmp{dst_path.suffix}")
            lat_ch = latent_posneg[target_channel]
            lat_ch = lat_ch / lat_ch.max()
            out_bmp = mi.Bitmap(lat_ch)
            out_bmp = out_bmp.convert(pixel_format=mi.Bitmap.PixelFormat.RGB, component_format=mi.Struct.Type.UInt8, srgb_gamma=False)
            out_bmp.write(str(tmp_path))
            utils.img_utils.upsample(str(tmp_path), str(dst_path), scale=8)
            tmp_path.unlink()


def _latest_debug_index(debug_dir: Path, prefix: str) -> int | None:
    """Return the highest frame index for PNGs matching <prefix>_XXXX.png in debug_dir."""
    pattern = re.compile(rf'^{re.escape(prefix)}_(\d+)\.png$')
    indices = [int(m.group(1)) for f in debug_dir.iterdir() if (m := pattern.match(f.name))]
    return max(indices) if indices else None


def get_png(src_path, out_dir):
    dst_path = out_dir / src_path.name
    try:
        copy2(src_path, dst_path)
    except Exception:
        return


def get_mp4(src_path, out_dir):
    dst_path = out_dir / src_path.name
    try:
        copy2(src_path, dst_path)
    except Exception:
        return


def get_pngs(out_dir, scene_dir, subdir=None, key_frames=None):
    # Experiment outputs
    if subdir in ["move_camera", "move_light", "move_object"] and key_frames:
        target_pngs = []
        for kf in key_frames:
            for png in [
                Path(f"best_rgb_{kf:04d}.png"),             # ground truth (full spp, denoised)
                Path(f"decoded_post_ambient_{kf:04d}.png"),  # pre-refine output
                Path(f"decoded_final_{kf:04d}.png"),         # post-refine output
            ]:
                target_pngs.append(
                    scene_dir / subdir / "images" / png
                )

    elif subdir == "views":
        target_pngs = [
            scene_dir / subdir / png for png in [
                Path("img_0.png"),
            ]
        ]

    elif subdir == "debug":
        debug_dir = scene_dir / subdir
        if not debug_dir.exists():
            return
        target_pngs = []
        pa_idx  = _latest_debug_index(debug_dir, "decoded_post_ambient")
        fin_idx = _latest_debug_index(debug_dir, "decoded_final")
        if pa_idx is not None:
            target_pngs += [
                debug_dir / f"decoded_post_ambient_{pa_idx:04d}.png",
                debug_dir / f"latent_post_ambient_{pa_idx:04d}.png",
            ]
        if fin_idx is not None:
            target_pngs += [
                debug_dir / f"decoded_final_{fin_idx:04d}.png",
                debug_dir / f"latent_final_{fin_idx:04d}.png",
            ]

    # Training final outputs
    elif subdir is None:
        target_pngs = [
            scene_dir / png for png in [
                Path("final.png"),
                Path("final_latent.png")
            ]
        ]

    for png in target_pngs:
        get_png(png, out_dir)


def get_exrs(out_dir, scene_dir, subdir=None, key_frames=None, target_channels=None):
    # Experiment outputs
    target_exrs = []
    if subdir in ["move_camera", "move_light", "move_object"] and key_frames:
        target_exrs = []
        for kf in key_frames:
            for exr in [
                Path(f"latent_best_latent_{kf:04d}.exr"),   # reference latent (best quality)
                Path(f"latent_post_ambient_{kf:04d}.exr"),  # rendered latent (pre-refine)
                Path(f"latent_refined_{kf:04d}.exr"),       # refined latent
            ]:
                target_exrs.append(
                    scene_dir / subdir / "images" / exr
                )

    # Training references
    elif subdir == "views":
        target_exrs = [
            scene_dir / subdir / exr for exr in [
                Path("latent_sample_0.exr"),
            ]
        ]

    # Debug outputs handled by get_pngs (PNGs only)
    elif subdir == "debug":
        return

    # Training final outputs
    elif subdir is None:
        pass

    for exr in target_exrs:
        get_exr(exr, out_dir, target_channels)


def make_flip_video(images_dir: Path, out_dir: Path) -> None:
    """Generate a flip-error video from per-frame PNGs.

    Uses flip_decoded_refined if available (runs with refiner), falls back to
    flip_decoded_final for ablation runs that have no refiner output.
    """
    for prefix in ("flip_decoded_refined", "flip_decoded_final"):
        first = images_dir / f"{prefix}_0000.png"
        if first.exists():
            pattern = str(images_dir / f"{prefix}_%04d.png")
            out_stem = str(out_dir / prefix)
            utils.img_utils.write_mp4(pattern, out_stem, framerate=60)
            return


def get_mp4s(out_dir, scene_dir, subdir=None):
    if subdir not in ["move_camera", "move_light", "move_object"]:
        return

    target_mp4s = [
        scene_dir / subdir / mp4 for mp4 in [
            Path("decoded_final.mp4"),
            Path("decoded_post_ambient.mp4"),
            Path("decoded_refined.mp4"),
            Path("latent_post_ambient.mp4"),
            Path("latent_best_latent.mp4"),
            Path("best_rgb.mp4"),
            Path("gt_rgb.mp4"),
        ]
    ]
    for mp4 in target_mp4s:
        get_mp4(mp4, out_dir)


def get_paper_data(
    scenes,
    out_dir,
    target_channels,
):
    for scene in tqdm(scenes):
        scene_dir = RESULT_DIR / scene
        scene_out_dir = OUT_DIR / scene
        scene_out_dir.mkdir(parents=True, exist_ok=True)

        for test_case, key_frames in tqdm(scenes[scene].items(), leave=False):
            test_dir = RESULT_DIR / scene / test_case
            test_out_dir: Path = OUT_DIR / scene / test_case
            test_out_dir.mkdir(parents=True, exist_ok=True)

            generate_latent_metrics(test_dir)

            latent_csv = test_dir / f"_{test_case}_latent.csv"
            if latent_csv.exists():
                plot_prerefine_vs_refine(latent_csv, test_out_dir, "ssim")

            get_pngs(test_out_dir, scene_dir, test_case, key_frames)
            get_exrs(test_out_dir, scene_dir, test_case, key_frames, target_channels)
            get_mp4s(test_out_dir, scene_dir, subdir=test_case)
            make_flip_video(test_dir / "images", test_out_dir)


        # Top-level data
        get_exrs(scene_out_dir, scene_dir)
        get_exrs(scene_out_dir, scene_dir, subdir="views", target_channels=target_channels)

        get_pngs(scene_out_dir, scene_dir)
        get_pngs(scene_out_dir, scene_dir, subdir="views")

        # Training debug: latest post_ambient + final frames
        debug_out_dir = scene_out_dir / "debug"
        debug_out_dir.mkdir(parents=True, exist_ok=True)
        get_pngs(debug_out_dir, scene_dir, subdir="debug")


if __name__ == "__main__":
    RESULT_DIR = Path("./outputs")
    OUT_DIR = Path("./paper_supplementals")
    _ABLATION_FRAMES = {
        "move_light": [0, 30, 59],
        "move_camera": [0, 30, 59],
        "move_object": [0, 30, 59],
    }
    EXPERIMENTS = {
        # Best results
        "CBox-1024-Best": {
            "move_light": [0, 30, 59],
            "move_camera": [0, 30, 59],
            "move_object": [0, 30, 59]
        },
        "Lamp-1024-Best": {
            "move_light": [0, 30, 59],
            "move_camera": [0, 30, 59],
            "move_object": [0, 30, 59]
        },
        "Living-1024-Best": {
            "move_light": [0, 20, 39],
            "move_camera": [9, 24, 30, 49],
            "move_object": [0, 18, 55]
        },
        "Dining-1024-Best": {
            "move_light": [0, 30, 59],
            "move_camera": [0, 30, 59],
            "move_object": [0, 30, 59]
        },
        "Veach-1024-Best": {
            "move_light": [0, 20, 39],
            "move_camera": [4, 24, 30, 57],
            "move_object": [13, 19, 32, 57]
        },
        # New scenes (move_camera only)
        "Bedroom-1024-Best":   {"move_camera": [0, 30, 59]},
        "Classroom-1024-Best": {"move_camera": [0, 30, 59]},
        "Living2-1024-Best":   {"move_camera": [0, 30, 59]},
        # Ablations
        "CBox-1024-NoAmb":   deepcopy(_ABLATION_FRAMES),
        "CBox-1024-NoOcc":   deepcopy(_ABLATION_FRAMES),
        "Lamp-1024-NoAmb":   deepcopy(_ABLATION_FRAMES),
        "Lamp-1024-NoOcc":   deepcopy(_ABLATION_FRAMES),
        "Living-1024-NoAmb": deepcopy(_ABLATION_FRAMES),
        "Living-1024-NoOcc": deepcopy(_ABLATION_FRAMES),
    }

    get_paper_data(
        EXPERIMENTS,
        OUT_DIR,
        target_channels=[],
    )
