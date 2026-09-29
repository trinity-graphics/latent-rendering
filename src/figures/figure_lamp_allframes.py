"""
Per-frame metrics figure for the Lamp scene across all three test types.

Generates a 3×2 grid:
  Rows:    move_camera (Novel View Rendering)
           move_light  (Relighting)
           move_object (Object Motion)
  Columns: LPIPS on decoded images | DSSIM on latents

Usage
-----
    python -m src.figures.figure_lamp_allframes --input-dir outputs/ --output-dir figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
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

_DEFAULT_RUN = "Lamp-1024-Best"

_ROWS = [
    ("move_camera", "Novel View Rendering"),
    ("move_light",  "Relighting"),
    ("move_object", "Object Motion"),
]

_METRICS = ["lpips", "ssim"]

_COL_TITLES = [
    ("Decoded (LPIPS)", "left"),
    ("Latent (DSSIM)",  "right"),
]

_LEGEND_LABELS = {
    "lpips": ("Rendered + Decoded RGB", "Refined + Decoded RGB"),
    "ssim":  ("Rendered Latent",        "Refined Latent"),
}


def _flatten_row(row: dict) -> dict:
    flat: dict = {"frame": row.get("step")}
    for key, val in row.items():
        if key == "step":
            continue
        if isinstance(val, dict):
            for subkey, subval in val.items():
                flat[f"{key}.{subkey}"] = subval
        elif val is not None:
            flat[key] = val
    return flat


def _load_metrics(run_dir: Path, test_type: str) -> pd.DataFrame | None:
    path = run_dir / test_type / f"{test_type}.jsonl"
    if not path.exists():
        print(f"  [missing] {path}")
        return None
    rows = [_flatten_row(json.loads(line))
            for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("frame").reset_index(drop=True)


def _fmt_y(x: float, _pos) -> str:
    return f"{x:.2f}"


def _plot_metric(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    show_xaxis: bool,
) -> None:
    pre_col  = f"decoded_post_ambient.{metric}"
    post_col = f"decoded_final.{metric}"
    frames   = df["frame"].values.astype(float)

    if pre_col in df.columns:
        y = df[pre_col].values.astype(float)
        if metric == "ssim":
            y = 1.0 - y
        ax.plot(frames, y, linewidth=2, color="C0")

    if post_col in df.columns:
        y = df[post_col].values.astype(float)
        if metric == "ssim":
            y = 1.0 - y
        ax.plot(frames, y, linewidth=2, color="C1")

    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_y))
    ax.grid(True, which="major", ls="-", alpha=0.2)

    if show_xaxis:
        ax.set_xlabel("Frames", fontsize=27)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    else:
        ax.tick_params(axis="x", labelbottom=False)


def generate(
    input_dir: Path | str,
    output_dir: Path | str,
    export_pdf_flag: bool = True,
    run: str = _DEFAULT_RUN,
    figsize: tuple[float, float] = (14, 16),
    dpi: int = 300,
) -> Path:
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dir = input_dir / run
    data: dict[str, pd.DataFrame | None] = {
        test_type: _load_metrics(run_dir, test_type)
        for test_type, _ in _ROWS
    }

    fig = plt.figure(figsize=figsize)

    # Alternating rows: thin title rows + plot rows
    # height_ratios: [title, plot, title, plot, title, plot]
    gs = GridSpec(
        6, 2, figure=fig,
        height_ratios=[0.11, 1, 0.11, 1, 0.11, 1],
        hspace=0.06, wspace=0.20,
        top=0.97, bottom=0.10, left=0.08, right=0.97,
    )

    plot_axes: list[tuple[plt.Axes, plt.Axes]] = []

    for i, (test_type, row_title) in enumerate(_ROWS):
        # Thin title row spanning both columns
        ax_hdr = fig.add_subplot(gs[i * 2, :])
        ax_hdr.axis("off")
        ax_hdr.text(0.5, 0.5, row_title,
                    transform=ax_hdr.transAxes,
                    ha="center", va="center",
                    fontsize=32, fontweight="bold")

        ax_l = fig.add_subplot(gs[i * 2 + 1, 0])
        ax_r = fig.add_subplot(gs[i * 2 + 1, 1])
        plot_axes.append((ax_l, ax_r))

        is_bottom = (i == len(_ROWS) - 1)
        df = data[test_type]

        for ax, metric in zip((ax_l, ax_r), _METRICS):
            if df is not None:
                _plot_metric(ax, df, metric, show_xaxis=is_bottom)
            else:
                ax.text(0.5, 0.5, "no data",
                        transform=ax.transAxes,
                        ha="center", va="center",
                        fontsize=20, color="red")
                if not is_bottom:
                    ax.tick_params(axis="x", labelbottom=False)

    # Column titles over the top row only
    for ax, (col_title, loc) in zip(plot_axes[0], _COL_TITLES):
        ax.set_title(col_title, loc=loc, fontsize=32, fontweight="bold", pad=10)

    # Per-column legends below the bottom row
    for ax, metric in zip(plot_axes[-1], _METRICS):
        pre_label, post_label = _LEGEND_LABELS.get(metric, ("pre-refiner", "refined"))
        handles = [
            Line2D([0], [0], color="C0", linewidth=2, label=pre_label),
            Line2D([0], [0], color="C1", linewidth=2, label=post_label),
        ]
        ax.legend(handles=handles, loc="upper center",
                  bbox_to_anchor=(0.5, -0.18), frameon=True, ncol=1)

    ext = "pdf" if export_pdf_flag else "png"
    out_path = output_dir / f"lamp_allframes.{ext}"
    fonts.savefig(fig, out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"lamp_allframes: → {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir",  required=True,            type=Path)
    p.add_argument("--output-dir", default=Path("figures/"), type=Path)
    p.add_argument("--run",        default=_DEFAULT_RUN,
                   help="Run directory name (default: Lamp-1024-Best)")
    p.add_argument("--no-pdf",     action="store_true")
    args = p.parse_args()
    generate(args.input_dir, args.output_dir,
             export_pdf_flag=not args.no_pdf,
             run=args.run)


if __name__ == "__main__":
    main()
