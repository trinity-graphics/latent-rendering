"""
Results matrix figure (template: templates/results_matrix.svg).

All scene/test_type/frame choices are baked into the template hrefs.
This script reads each <image> href, resolves it to an actual file path
using resolve.py, generates latent channel PNGs on demand, and rewrites
the SVG hrefs to point at the generated files.

Usage
-----
    python -m src.figures.figure_results --input-dir outputs/ --output-dir figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.figures.common.image_proc import latent_channel_png
from src.figures.common.resolve import (
    LatentHref,
    RunsHref,
    parse_href,
    resolve_flat,
    resolve_latent_exr,
)
from src.figures.common.svg_utils import (
    export_pdf,
    find_all_images,
    load_svg,
    save_svg,
    set_image_href,
    substitute_metric_placeholders,
)

TEMPLATE = Path(__file__).parent / "templates" / "results_matrix.svg"


def generate(
    input_dir: Path | str,
    output_dir: Path | str,
    export_pdf_flag: bool = True,
    metrics_csv: Path | str | None = None,
) -> Path:
    input_dir  = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    latent_dir = output_dir / "latent_results"
    latent_dir.mkdir(parents=True, exist_ok=True)

    tree    = load_svg(TEMPLATE)
    replaced = 0

    for elem, href in find_all_images(tree):
        parsed = parse_href(href)
        if parsed is None:
            continue

        if isinstance(parsed, RunsHref):
            path = resolve_flat(parsed, input_dir)
            if path.exists():
                set_image_href(elem, path.as_uri())
                replaced += 1
            else:
                print(f"  [missing] {path}")

        elif isinstance(parsed, LatentHref):
            exr, ch = resolve_latent_exr(parsed, input_dir)
            out_png = (
                latent_dir
                / f"{parsed.scene}_{parsed.test_type}_{parsed.frame}_{parsed.ltype}_ch{ch}.png"
            )
            if exr.exists():
                if not out_png.exists():
                    latent_channel_png(exr, out_png, ch)
                set_image_href(elem, out_png.as_uri())
                replaced += 1
            else:
                print(f"  [missing exr] {exr}")

    if metrics_csv is not None:
        n = substitute_metric_placeholders(tree, metrics_csv)
        if n:
            print(f"  substituted {n} metric placeholders")

    svg_dir = output_dir / "svg"
    svg_dir.mkdir(parents=True, exist_ok=True)
    out_svg = svg_dir / "results_matrix.svg"
    save_svg(tree, out_svg)
    print(f"results_matrix: {replaced} hrefs → {out_svg}")

    if export_pdf_flag:
        pdf = output_dir / out_svg.with_suffix(".pdf").name
        try:
            export_pdf(out_svg, pdf)
            print(f"  exported → {pdf}")
        except Exception as e:
            print(f"  PDF export failed: {e}")

    return out_svg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir",   required=True,            type=Path)
    p.add_argument("--output-dir",  default=Path("figures/"), type=Path)
    p.add_argument("--no-pdf",      action="store_true")
    p.add_argument("--metrics-csv", default=None, type=Path)
    args = p.parse_args()
    generate(args.input_dir, args.output_dir,
             export_pdf_flag=not args.no_pdf,
             metrics_csv=args.metrics_csv)


if __name__ == "__main__":
    main()
