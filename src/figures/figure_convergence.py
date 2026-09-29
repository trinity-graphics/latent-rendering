"""
Convergence curves figure (pure Python / matplotlib).

Reads convergence_*.jsonl from each run's convergence experiment subdirectory.
Each figure shows two lines per run: pre-refine (decoded_processed) vs
post-refine (decoded_final), matching the style of scripts/extract_convergence.py.
Step 0 is always excluded (corrupted values).

Default layout (3 panels, top to bottom):
  1. Training view convergence charts  (convergence_best)
  2. Novel view convergence charts     (convergence_novel_best)
  3. Training view image grid

The novel-view image grid is present in the code but commented out.
Pass --experiment to show a single experiment in the classic 2-panel layout.

Usage
-----
    # Default: training + novel charts, training image grid:
    python -m src.figures.figure_convergence --input-dir outputs/ --output-dir figures/

    # Single experiment (classic charts + image grid):
    python -m src.figures.figure_convergence --input-dir outputs/ --output-dir figures/ \
        --experiment convergence_novel_best

    # Custom base metrics:
    python -m src.figures.figure_convergence --input-dir outputs/ --output-dir figures/ \
        --metrics lpips ssim
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.lines import Line2D

from src.figures.common import fonts

fonts.apply(**{
    "axes.titlesize":     32,
    "axes.titleweight":   "bold",
    "axes.labelsize":     27,
    "legend.fontsize":    24,
    "xtick.labelsize":    24,
    "ytick.labelsize":    24,
    "text.color":         "black",
    "axes.labelcolor":    "black",
    "xtick.color":        "black",
    "ytick.color":        "black",
})

_RUN_LABELS: dict[str, str] = {
    "CBox-1024-Best":   "Cornell Box",
    "Lamp-1024-Best":   "Lamp",
    "Living-1024-Best": "Living Room",
    "Dining-1024-Best": "Dining Room",
    "Veach-1024-Best":  "Veach Bidir",
}

# Base metric name → y-axis display label
_METRIC_LABELS: dict[str, str] = {
    "lpips": "LPIPS",
    "mse":   "MSE",
    "psnr":  "PSNR",
    "ssim":  "DSSIM",
    "flip":  "FLIP",
}

# Per-metric legend labels (pre-refiner, refined)
_LEGEND_LABELS: dict[str, tuple[str, str]] = {
    "lpips": ("Rendered + Decoded RGB", "Refined + Decoded RGB"),
    "ssim":  ("Rendered Latent",        "Refined Latent"),
}

# Line styles to distinguish runs when there are multiple
_RUN_LINESTYLES = ["-", "--", ":", "-."]

# Default experiments shown in multi-section mode (training first, then novel)
_EXPERIMENTS = [
    ("convergence_best",       "Training View Convergence"),
    ("convergence_novel_best", "Novel View Convergence"),
]


def _fmt_spp(x: float, _pos) -> str:
    """Format spp x-axis values: 1, 1k, 1M, 1G, 1T, 1P …"""
    xi = int(round(x))
    for divisor, suffix in [
        (10**15, "P"), (10**12, "T"), (10**9, "G"),
        (10**6, "M"), (10**3, "k"),
    ]:
        if xi >= divisor:
            v = xi / divisor
            return f"{v:.0f}{suffix}" if v == int(v) else f"{v:.1f}{suffix}"
    return str(xi)


# Fixed tick positions at powers of 10 (spp = step^2, max ≈ 59^2 = 3481)
_SPP_TICKS = [1, 10, 100, 1000, 10000]


def _fmt_y(x: float, _pos) -> str:
    return f"{x:.1f}"


def _flatten_row(row: dict) -> dict:
    flat: dict = {"step": row.get("step")}
    for key, val in row.items():
        if key == "step":
            continue
        if isinstance(val, dict):
            for subkey, subval in val.items():
                flat[f"{key}.{subkey}"] = subval
        elif val is not None:
            flat[key] = val
    return flat


def _load_convergence_jsonl(run_dir: Path, experiment: str | None) -> pd.DataFrame | None:
    if experiment is not None:
        candidates = [run_dir / experiment / f"{experiment}.jsonl"]
    else:
        candidates = sorted(run_dir.glob("convergence_*/*.jsonl"))

    for path in candidates:
        if path.exists():
            rows = [_flatten_row(json.loads(line))
                    for line in path.read_text().splitlines() if line.strip()]
            if rows:
                df = pd.DataFrame(rows)
                # Drop step 0 — values are corrupted at initialisation
                return df[df["step"] > 0].reset_index(drop=True)
    return None


def _setup_xaxis(ax: plt.Axes) -> None:
    """Log10 x-axis (spp = step²); ticks at 1, 10, 100, 1k."""
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(ticker.FixedLocator(_SPP_TICKS))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_spp))
    ax.set_xlabel("SPP", fontsize=26)
    ax.xaxis.set_label_coords(1.0, -0.04)


def _plot_one_metric(
    ax: plt.Axes,
    data: dict[str, pd.DataFrame],
    metric: str,
    smooth: int,
) -> None:
    """Draw pre-refine (blue/C0) and refined (orange/C1) lines for each run."""
    pre_col  = f"decoded_post_ambient.{metric}"
    post_col = f"decoded_final.{metric}"

    for i, (run, df) in enumerate(data.items()):
        ls = _RUN_LINESTYLES[i % len(_RUN_LINESTYLES)]
        spp = (df["step"].values ** 2).astype(float)

        if pre_col in df.columns:
            y = df[pre_col].values.astype(float)
            if metric == "ssim":
                y = 1.0 - y
            if smooth > 1:
                y = np.convolve(y, np.ones(smooth) / smooth, mode="same")
            ax.plot(spp, y, linewidth=2, color="C0", linestyle=ls)

        if post_col in df.columns:
            y = df[post_col].values.astype(float)
            if metric == "ssim":
                y = 1.0 - y
            if smooth > 1:
                y = np.convolve(y, np.ones(smooth) / smooth, mode="same")
            ax.plot(spp, y, linewidth=2, color="C1", linestyle=ls)

    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_y))
    ax.grid(True, which="major", ls="-", alpha=0.2)
    _setup_xaxis(ax)


_PREVIEW_SPPS = [16, 64, 256, 1024]

_IMG_VARIANTS = [
    ("decoded_post_ambient", "Decoded (PBR Latents)"),
    ("decoded_refined",      "Decoded (Refined Latents)"),
]


def _find_experiment_dir(run_dir: Path, experiment: str | None) -> Path | None:
    if experiment is not None:
        d = run_dir / experiment
        return d if d.is_dir() else None
    candidates = sorted(run_dir.glob("convergence_*/"))
    return candidates[0] if candidates else None


def _plot_charts(
    sf,
    data: dict[str, pd.DataFrame],
    metrics: list[str],
    smooth: int,
    title: str,
) -> None:
    """Render convergence chart panels into a figure or subfigure."""
    n_panels = len(metrics)
    sf.suptitle(title, fontsize=36, fontweight="bold", y=0.995)
    axes = sf.subplots(1, n_panels, squeeze=False)[0]
    sf.subplots_adjust(bottom=0.26, top=0.83, left=0.08, right=0.97, wspace=0.35)
    for ax, metric in zip(axes, metrics):
        _plot_one_metric(ax, data, metric, smooth)
        ax.set_title(_METRIC_LABELS.get(metric, metric.upper()))
        pre_label, post_label = _LEGEND_LABELS.get(metric, ("pre-refiner", "refined"))
        ax.legend(
            handles=[
                Line2D([0], [0], color="C0", linewidth=2, label=pre_label),
                Line2D([0], [0], color="C1", linewidth=2, label=post_label),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.10),
            frameon=True,
            ncol=1,
        )


def _plot_image_grid(
    sf,
    exp_dir: Path | None,
    metrics: list[str],
) -> None:
    """Render convergence image-grid panels into a figure or subfigure."""
    n_panels = len(metrics)
    img_subfigs = sf.subfigures(1, n_panels, wspace=0.03)
    if n_panels == 1:
        img_subfigs = [img_subfigs]

    for img_sf, (variant, grid_title) in zip(img_subfigs, _IMG_VARIANTS):
        img_sf.suptitle(grid_title, fontsize=32, fontweight="bold")
        axes_img = img_sf.subplots(2, 2)
        img_sf.subplots_adjust(top=0.97, bottom=0.0, left=0.0, right=1.0,
                               wspace=0.0, hspace=0.03)
        for ax, spp in zip(axes_img.flatten(), _PREVIEW_SPPS):
            step = int(spp ** 0.5)
            img_path = (
                exp_dir / "images" / f"{variant}_{step:04d}.png"
                if exp_dir else None
            )
            if img_path and img_path.exists():
                ax.imshow(plt.imread(str(img_path)))
            else:
                ax.set_facecolor("#ddd")
                label = img_path.name if img_path else "no data"
                ax.text(0.5, 0.5, f"missing\n{label}",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=8, color="#666")
            ax.axis("off")
            ax.text(0.5, -0.02, f"{spp}spp",
                    transform=ax.transAxes,
                    ha="center", va="top",
                    fontsize=30, fontweight="bold",
                    clip_on=False)


def _plot_section(
    sf,
    data: dict[str, pd.DataFrame],
    exp_dir: Path | None,
    metrics: list[str],
    smooth: int,
    title: str,
) -> None:
    """Render one full section (charts + image grid) — used in single-experiment mode."""
    inner = sf.subfigures(2, 1, height_ratios=[1.0, 1.4], hspace=0.02)
    _plot_charts(inner[0], data, metrics, smooth, title)
    _plot_image_grid(inner[1], exp_dir, metrics)


def generate(
    input_dir: Path | str,
    output_dir: Path | str,
    export_pdf_flag: bool = True,
    runs: list[str] | None = None,
    metrics: list[str] | None = None,
    experiment: str | None = None,
    smooth: int = 1,
    figsize: tuple[float, float] = (14, 17),
    dpi: int = 300,
) -> list[Path]:
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if runs is None:
        runs = ["Veach-1024-Best"]
    if metrics is None:
        metrics = ["lpips", "ssim"]

    # Single-experiment mode (CLI override) vs default multi-section mode
    experiments_to_show = (
        [(experiment, experiment.replace("_", " ").title())]
        if experiment is not None
        else _EXPERIMENTS
    )

    # Load data for each experiment section
    sections: list[tuple[str, dict, Path | None]] = []
    for exp_name, title in experiments_to_show:
        data: dict[str, pd.DataFrame] = {}
        exp_dirs: dict[str, Path] = {}
        for run in runs:
            run_dir = input_dir / run
            if not run_dir.is_dir():
                continue
            exp_dir = _find_experiment_dir(run_dir, exp_name)
            df = _load_convergence_jsonl(run_dir, exp_name)
            if df is None:
                print(f"  warning: no JSONL for run={run!r} experiment={exp_name!r}, skipping")
            else:
                data[run] = df
                if exp_dir is not None:
                    exp_dirs[run] = exp_dir
        if data:
            sections.append((title, data, next(iter(exp_dirs.values()), None)))

    if not sections:
        print("No convergence metrics found.")
        return []

    n = len(sections)

    if n == 1:
        # Classic 2-panel layout: charts on top, image grid below
        fig = plt.figure(figsize=figsize)
        _plot_section(fig, sections[0][1], sections[0][2], metrics, smooth, sections[0][0])
    else:
        # Multi-section layout (top to bottom):
        #   [0] training view charts   (sections[0])
        #   [1] novel view charts      (sections[1])
        #   [2] training view img grid (sections[0])
        #   -- novel view img grid is commented out --
        _CHART_R, _GRID_R = 1.0, 1.4
        height_ratios = [_CHART_R] * n + [_GRID_R]
        fig_height = figsize[1] * sum(height_ratios) / (_CHART_R + _GRID_R)
        fig = plt.figure(figsize=(figsize[0], fig_height))
        parts = fig.subfigures(n + 1, 1, height_ratios=height_ratios, hspace=0.04)

        for i, (title, data, _exp_dir) in enumerate(sections):
            _plot_charts(parts[i], data, metrics, smooth, title)

        _plot_image_grid(parts[n], sections[0][2], metrics)         # training view
        # _plot_image_grid(parts[n + 1], sections[1][2], metrics)   # novel view (if re-enabled, add to height_ratios too)

    ext = "pdf" if export_pdf_flag else "png"
    out_path = output_dir / f"convergence.{ext}"
    fonts.savefig(fig, out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"convergence: → {out_path}")
    return [out_path]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir",   required=True,            type=Path)
    p.add_argument("--output-dir",  default=Path("figures/"), type=Path)
    p.add_argument("--runs",        nargs="*", default=None,
                   help="Run subdirectory names (default: Veach-1024-Best)")
    p.add_argument("--metrics",     nargs="*", default=["lpips", "ssim"],
                   help="Base metric names: lpips, mse, ssim, psnr, flip")
    p.add_argument("--experiment",  default=None,
                   help="Single convergence experiment name; default shows "
                        "training + novel charts with training image grid stacked")
    p.add_argument("--smooth",      default=1, type=int,
                   help="Moving-average window size (1 = no smoothing)")
    p.add_argument("--no-pdf",      action="store_true",
                   help="Save PNG instead of PDF")
    args = p.parse_args()
    generate(
        args.input_dir, args.output_dir,
        export_pdf_flag=not args.no_pdf,
        runs=args.runs, metrics=args.metrics,
        experiment=args.experiment, smooth=args.smooth,
    )


if __name__ == "__main__":
    main()
