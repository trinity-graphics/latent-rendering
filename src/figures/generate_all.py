"""
Run every figure script and the tables script for a given input directory.

PDF export is on by default for every figure. Use --no-pdf to skip.
SVG intermediates are always kept (no image embedding).

Usage
-----
    python -m src.figures.generate_all --input-dir outputs/ --output-dir figures/

    # Skip specific figures:
    python -m src.figures.generate_all --input-dir outputs/ --skip convergence naive

    # Skip PDF export (faster, SVG only):
    python -m src.figures.generate_all --input-dir outputs/ --no-pdf

    # Substitute metric placeholders (requires tables to have run first):
    python -m src.figures.generate_all --input-dir outputs/ --metrics-csv figures/metrics_keyframe.csv
"""

from __future__ import annotations

import argparse
import re
import traceback
from pathlib import Path

from src.figures import (
    figure_bidir,
    figure_convergence,
    figure_lamp_allframes,
    figure_latent,
    figure_move_camera,
    figure_naive,
    figure_results,
    figure_teaser,
    figure_training,
    tables,
)
from src.figures.common.data_loader import SCENE_TO_RUN
from src.figures.common.resolve import (
    CropHref,
    DebugHref,
    DebugLatentHref,
    LatentHref,
    RunsHref,
    ViewLatentHref,
    parse_href,
)

_FIGURES = {
    "results":          figure_results.generate,
    "training":         figure_training.generate,
    "bidir":            figure_bidir.generate,
    "teaser":           figure_teaser.generate,
    "naive":            figure_naive.generate,
    "latent":           figure_latent.generate,
    "latent_vae":       figure_latent.generate_vae,
    "convergence":      figure_convergence.generate,
    "lamp_allframes":   figure_lamp_allframes.generate,
    "move_camera":      figure_move_camera.generate,
}

# Figures that accept a metrics_csv keyword argument
_METRICS_FIGURES = {"results", "training", "bidir", "naive", "move_camera"}

_CONVERGENCE_RUN = "Veach-1024-Best"

# SVG template of each template-based figure, used to find the run directories it reads
_TEMPLATES = {
    "results":      figure_results.TEMPLATE,
    "training":     figure_training.TEMPLATE,
    "bidir":        figure_bidir.TEMPLATE,
    "teaser":       figure_teaser.TEMPLATE,
    "naive":        figure_naive.TEMPLATE,
    "latent":       figure_latent.TEMPLATE,
    "latent_vae":   figure_latent.TEMPLATE_VAE,
    "move_camera":  figure_move_camera.TEMPLATE,
}

# (run, experiment) directories read by the figures without a template
_FIXED_INPUTS = {
    "convergence":    [(_CONVERGENCE_RUN, exp) for exp, _ in figure_convergence._EXPERIMENTS],
    "lamp_allframes": [(figure_lamp_allframes._DEFAULT_RUN, exp) for exp, _ in figure_lamp_allframes._ROWS],
}


def _required_dirs(name: str) -> set[tuple[str, str]]:
    """Returns the (run, experiment) directories that a figure reads from the input directory."""
    if name in _FIXED_INPUTS:
        return set(_FIXED_INPUTS[name])
    dirs = set()
    for href in re.findall(r'href="([^"]+)"', _TEMPLATES[name].read_text()):
        parsed = parse_href(href)
        if parsed is None:
            continue
        run = SCENE_TO_RUN.get(parsed.scene, parsed.scene)
        if isinstance(parsed, (RunsHref, LatentHref, CropHref)):
            dirs.add((run, parsed.test_type))
        elif isinstance(parsed, (DebugHref, DebugLatentHref)):
            dirs.add((run, "debug"))
        elif isinstance(parsed, ViewLatentHref):
            dirs.add((run, "views"))
    return dirs


def generate_all(
    input_dir: Path | str,
    output_dir: Path | str,
    skip: list[str] | None = None,
    export_pdf_flag: bool = True,
    metrics_csv: Path | str | None = None,
) -> None:
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    skip       = set(skip or [])

    # Tables first so metrics CSVs exist before figure scripts run
    if "tables" not in skip:
        print("\n=== tables ===")
        try:
            tables.make_tables(input_dir, output_dir)
        except Exception:
            print("  ERROR in tables:")
            traceback.print_exc()

    # Auto-discover keyframe CSV from output dir if not explicitly provided
    if metrics_csv is None:
        candidate = output_dir / "metrics_keyframe.csv"
        if candidate.exists():
            metrics_csv = candidate
            print(f"\n[metrics] using {metrics_csv}")

    for name, fn in _FIGURES.items():
        if name in skip:
            print(f"  [skip] {name}")
            continue
        missing = sorted(f"{run}/{exp}" for run, exp in _required_dirs(name) if not (input_dir / run / exp).is_dir())
        if missing:
            print(f"  [skip] {name}: missing {', '.join(missing)}")
            continue
        print(f"\n=== {name} ===")
        try:
            kwargs: dict = {"export_pdf_flag": export_pdf_flag}
            if name in _METRICS_FIGURES and metrics_csv is not None:
                kwargs["metrics_csv"] = metrics_csv
            if name == "convergence":
                kwargs["runs"] = [_CONVERGENCE_RUN]
            fn(input_dir, output_dir, **kwargs)
        except Exception:
            print(f"  ERROR in {name}:")
            traceback.print_exc()

    print("\nDone.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir",    required=True,            type=Path)
    p.add_argument("--output-dir",   default=Path("figures/"), type=Path)
    p.add_argument("--skip",         nargs="*", default=None,
                   help="Figure names to skip: " + ", ".join(_FIGURES) + ", tables")
    p.add_argument("--no-pdf",       action="store_true",
                   help="Skip PDF export (keep SVG only)")
    p.add_argument("--metrics-csv",  default=None, type=Path,
                   help="CSV with per-frame metrics for placeholder substitution")
    args = p.parse_args()
    generate_all(
        args.input_dir, args.output_dir,
        skip=args.skip,
        export_pdf_flag=not args.no_pdf,
        metrics_csv=args.metrics_csv,
    )


if __name__ == "__main__":
    main()
