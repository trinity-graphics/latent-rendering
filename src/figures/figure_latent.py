"""
Latent figures.  Two outputs, from two templates:

1. latent_analysis.svg — 16-channel analysis of a single latent.
   Template hrefs:
     latents/cbox/move_object/0/latent_ref/ch{N}.png  — all 16 channels (ch0–ch15)
     crops/cbox/move_object/0/latent_ref/ch{N}_b{M}.png — crop variants
   Crop parameters are hardcoded here and must match the template's visual layout.
   To adjust frame, edit the template hrefs (search "latents/cbox/move_object/0").

2. latent_vae_comparison.svg — four latent channels per VAE in a 2x2 block
   (FLUX.1 / FLUX.2 / Qwen), same cbox scene and full method, read from the
   reference view outputs/VAE/*-Cbox/views/latent_sample_0.exr.
   Channel choice is per model (_VAE_CHANNELS) because the latent spaces are
   not aligned; contrast is normalised per model (_VAE_PERCENTILE) because
   their value ranges differ by ~5x.

Usage
-----
    python -m src.figures.figure_latent --input-dir outputs/ --output-dir figures/

    # Contact sheets of every channel per VAE, to choose _VAE_CHANNELS:
    python -m src.figures.figure_latent --input-dir outputs/ --dump-vae-channels
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.figures.common.image_proc import (
    crop_region,
    latent_abs_percentile,
    latent_channel_png,
)
from src.figures.common.resolve import (
    CropHref,
    LatentHref,
    ViewLatentHref,
    parse_href,
    resolve_latent_exr,
)
from src.figures.common.svg_utils import (
    export_pdf,
    find_all_images,
    load_svg,
    save_svg,
    set_image_href,
)

# Crop regions derived from the SVG indicator box positions.
# Coordinates are in the upsampled PNG space (128×128 EXR → 512×512 PNG at upsample=4,
# so 30 px/mm in SVG coords).  _CROPS[i] = crop for b{i+1}.
#   b1 (teal  #18b6c2): ch8, ch11  — upper-left quadrant
#   b2 (purple #d261eb): ch0, ch9  — upper-right quadrant
#   b3 (yellow #e8e81d): ch14      — lower-left region
#   b4 (orange, presentation only, not in the template): ch14, ch3 — upper right
#      wall plus the ceiling/back-wall corner as an anchor; ch14 carries a strong
#      Lambert cosine lobe, ch3 the same falloff at a much lower amplitude
_CROPS = [
    {"x": 189, "y":  17, "w": 138, "h": 147},  # b1
    {"x": 350, "y":  29, "w": 138, "h": 147},  # b2
    {"x": 139, "y": 379, "w": 218, "h": 116},  # b3
    {"x": 361, "y":  90, "w": 138, "h": 147},  # b4 (same size as b1/b2)
]
_BOX_COLORS = [
    (24, 182, 194),   # b1 teal
    (210, 97, 235),   # b2 purple
    (232, 232, 29),   # b3 yellow
    (245, 130, 32),   # b4 orange
]
# Captions under each crop group in the template; also the strip PNG filenames.
_CROP_LABELS = ["Signed emitters", "Geometric edges", "Signed shadows", "Cosine falloff"]
# Crops with no box in latent_analysis.svg: crop_idx -> channels, left to right.
_EXTRA_CROP_CHANNELS = {4: [14, 3]}
_UPSAMPLE  = 4
_GAMMA     = 2.2

TEMPLATE = Path(__file__).parent / "templates" / "latent_analysis.svg"
PRESENTATION_DIR = Path(__file__).parents[2] / "presentation"

# ── VAE comparison figure ───────────────────────────────────────────────────
TEMPLATE_VAE = Path(__file__).parent / "templates" / "latent_vae_comparison.svg"

# Which four latent channels to show per model.  They are sorted ascending and
# then fill the 2x2 block in reading order (top-left, top-right, bottom-left,
# bottom-right), so the order they are typed in here does not matter — drop the
# sorted() in generate_vae to place them by hand instead.  The three VAEs have
# independent latent spaces (FLUX.2 32ch, FLUX.1 16ch, Qwen 16ch), so "channel
# N" is not the same feature in each — these are picked by eye.  Re-pick from
# the contact sheets written by --dump-vae-channels.  Keys are scene labels
# (SCENE_TO_RUN); block order and grid shape live in the template.
_VAE_CHANNELS = {
    "vae_flux1": (2, 3, 9, 13),
    "vae_flux2": (1, 8, 19, 26),
    "vae_qwen":  (1, 5, 7, 10),
}
# Reference view the channels are read from: views/latent_sample_{index}.exr.
_VAE_VIEW_INDEX = 0
_VAE_VIEW_LTYPE = "latent_sample"
# Latent value ranges differ ~5x between these VAEs (FLUX.2 ≈ ±0.8, FLUX.1 ≈
# ±3.7), so a shared fixed clip would render FLUX.2 nearly black.  Clip each
# model at this percentile of |value| over all of its channels instead.
_VAE_PERCENTILE = 99.0

_TSPAN_TAG = "{http://www.w3.org/2000/svg}tspan"


def _channel_label(ch: int) -> str:
    """Corner label for a latent tile: two digits, as in latent_analysis.svg.

    Note latent_analysis numbers its tiles 01–16, i.e. 1-indexed (its ch0 tile
    reads "01").  Here the label is the raw channel index, so it matches
    _VAE_CHANNELS and the --dump-vae-channels contact sheets.  Return
    f"{ch + 1:02d}" instead to follow the 1-indexed convention.
    """
    return f"{ch:02d}"


def _make_channel_png(exr_path: Path, ch: int, out_dir: Path) -> Path:
    out = out_dir / f"ch{ch}.png"
    if not out.exists():
        latent_channel_png(exr_path, out, ch, upsample=_UPSAMPLE, gamma=_GAMMA)
    return out


def _make_crop_png(base_png: Path, crop_idx: int, ch: int, out_dir: Path,
                   c: dict | None = None) -> Path:
    out = out_dir / f"ch{ch}_b{crop_idx}.png"
    c = c or _CROPS[crop_idx - 1]
    color = _BOX_COLORS[crop_idx - 1]
    img = Image.open(str(base_png)).convert("RGB")
    crop = crop_region(img, c["x"], c["y"], c["w"], c["h"],
                       border_px=2, color=color)
    crop_hi = crop.resize((crop.width * _UPSAMPLE, crop.height * _UPSAMPLE),
                           resample=Image.NEAREST)
    out.parent.mkdir(parents=True, exist_ok=True)
    crop_hi.save(str(out))
    return out


def _make_labelled_strip(pngs: list[Path], label: str, out_dir: Path) -> Path:
    """Crops side by side (as in the template) with the label underneath."""
    tiles = [Image.open(str(p)).convert("RGB") for p in pngs]
    w, h = sum(t.width for t in tiles), max(t.height for t in tiles)
    # Label scales with the tiles: strips differ in size (b3 is shorter) but are
    # shown at the same size, as in the template.  Even height for yuv420p video.
    label_h = h // 12 * 2
    img = Image.new("RGB", (w, h + label_h), "white")
    x = 0
    for t in tiles:
        img.paste(t, (x, 0))
        x += t.width
    ImageDraw.Draw(img).text((w / 2, h + label_h / 2), label, fill="black",
                             font=ImageFont.load_default(size=h // 9), anchor="mm")
    out = out_dir / f"{label.lower().replace(' ', '_')}.png"
    img.save(str(out))
    return out


def generate(
    input_dir: Path | str,
    output_dir: Path | str,
    export_pdf_flag: bool = True,
) -> Path:
    input_dir  = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    latent_dir = output_dir / "latent_analysis_channels"
    latent_dir.mkdir(parents=True, exist_ok=True)

    tree    = load_svg(TEMPLATE)
    replaced = 0
    # Cache exr_path per (scene, test_type, frame, ltype) to avoid re-resolving
    _channel_cache: dict[tuple, Path] = {}
    strips: dict[int, list] = {}  # crop_idx -> [(template x, crop png)]

    for elem, href in find_all_images(tree):
        parsed = parse_href(href)
        if parsed is None:
            continue

        if isinstance(parsed, LatentHref):
            exr, ch = resolve_latent_exr(parsed, input_dir)
            if exr.exists():
                out_png = _make_channel_png(exr, ch, latent_dir)
                key = (parsed.scene, parsed.test_type, parsed.frame, parsed.ltype, ch)
                _channel_cache[key] = out_png
                set_image_href(elem, out_png.as_uri())
                replaced += 1
            else:
                print(f"  [missing exr] {exr}")

        elif isinstance(parsed, CropHref):
            from src.figures.common.data_loader import LATENT_TYPE_TO_FILE, SCENE_TO_RUN
            run = SCENE_TO_RUN.get(parsed.scene, parsed.scene)
            tmpl = LATENT_TYPE_TO_FILE[parsed.ltype]
            fname = tmpl.format(frame=parsed.frame)
            exr = input_dir / run / parsed.test_type / "images" / fname
            ch = parsed.channel
            base_key = (parsed.scene, parsed.test_type, parsed.frame, parsed.ltype, ch)
            base_png = _channel_cache.get(base_key)
            if base_png is None and exr.exists():
                base_png = _make_channel_png(exr, ch, latent_dir)
                _channel_cache[base_key] = base_png
            if base_png is not None and base_png.exists():
                out_crop = _make_crop_png(base_png, parsed.crop_idx, ch, latent_dir)
                set_image_href(elem, out_crop.as_uri())
                strips.setdefault(parsed.crop_idx, []).append((float(elem.get("x", 0)), out_crop))
                replaced += 1
            elif not exr.exists():
                print(f"  [missing exr] {exr}")

    # Same reference latent as the template crops; its channel PNGs are all in latent_dir.
    for idx, channels in _EXTRA_CROP_CHANNELS.items():
        strips[idx] = [(i, _make_crop_png(latent_dir / f"ch{ch}.png", idx, ch, latent_dir))
                       for i, ch in enumerate(channels)]

    for idx, items in strips.items():
        PRESENTATION_DIR.mkdir(exist_ok=True)
        out = _make_labelled_strip([p for _, p in sorted(items)], _CROP_LABELS[idx - 1], PRESENTATION_DIR)
        print(f"  crop strip → {out}")

    svg_dir = output_dir / "svg"
    svg_dir.mkdir(parents=True, exist_ok=True)
    out_svg = svg_dir / "latent_analysis.svg"
    save_svg(tree, out_svg)
    print(f"latent_analysis: {replaced} hrefs → {out_svg}")

    if export_pdf_flag:
        pdf = output_dir / out_svg.with_suffix(".pdf").name
        try:
            export_pdf(out_svg, pdf)
            print(f"  exported → {pdf}")
        except Exception as e:
            print(f"  PDF export failed: {e}")

    return out_svg


def generate_vae(
    input_dir: Path | str,
    output_dir: Path | str,
    export_pdf_flag: bool = True,
) -> Path:
    """Four latent channels per VAE (template: templates/latent_vae_comparison.svg)."""
    input_dir  = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    latent_dir = output_dir / "latent_vae"
    latent_dir.mkdir(parents=True, exist_ok=True)

    tree     = load_svg(TEMPLATE_VAE)
    replaced = 0

    # Corner labels: tspan "label-<slot>" is written from the channel its
    # "image-<slot>" actually rendered, so the two cannot drift apart.
    labels = {e.get("id"): e for e in tree.iter(_TSPAN_TAG) if e.get("id")}

    # Group each model's slots and order them in reading order, so the channel a
    # slot gets depends on where it sits in the figure, not on the template's
    # z-order.  Works for a side-by-side pair or a stacked one.
    slots: dict[str, list] = {}
    for elem, href in find_all_images(tree):
        parsed = parse_href(href)
        if isinstance(parsed, ViewLatentHref):
            slots.setdefault(parsed.scene, []).append((elem, parsed))

    for scene, entries in slots.items():
        entries.sort(key=lambda e: (float(e[0].get("y", 0)), float(e[0].get("x", 0))))
        channels = _VAE_CHANNELS.get(scene)
        if channels is None:
            print(f"  [no channels configured] {scene}")
            continue

        # Ascending, so the block reads low → high however the tuple is typed.
        for (elem, parsed), ch in zip(entries, sorted(channels)):
            exr, _ = resolve_latent_exr(parsed, input_dir)
            if not exr.exists():
                print(f"  [missing exr] {exr}")
                continue

            clip_max = latent_abs_percentile(exr, _VAE_PERCENTILE)
            out_png  = latent_dir / f"{scene}_ch{ch}.png"
            latent_channel_png(exr, out_png, ch, upsample=_UPSAMPLE, gamma=_GAMMA,
                               clip_max=clip_max)
            set_image_href(elem, out_png.as_uri())
            replaced += 1

            label = labels.get((elem.get("id") or "").replace("image-", "label-", 1))
            if label is not None:
                label.text = _channel_label(ch)

            print(f"  {scene}: ch{ch}  clip=±{clip_max:.3f}  ({exr.parent.parent.name}/views)")

    svg_dir = output_dir / "svg"
    svg_dir.mkdir(parents=True, exist_ok=True)
    out_svg = svg_dir / "latent_vae_comparison.svg"
    save_svg(tree, out_svg)
    print(f"latent_vae_comparison: {replaced} hrefs → {out_svg}")

    if export_pdf_flag:
        pdf = output_dir / out_svg.with_suffix(".pdf").name
        try:
            export_pdf(out_svg, pdf)
            print(f"  exported → {pdf}")
        except Exception as e:
            print(f"  PDF export failed: {e}")

    return out_svg


def dump_vae_channels(
    input_dir: Path | str,
    output_dir: Path | str,
    ltype: str = _VAE_VIEW_LTYPE,
    index: int = _VAE_VIEW_INDEX,
) -> list[Path]:
    """Write a labelled contact sheet of every latent channel, per VAE.

    Reads the same reference view as the figure, so the channel numbers on the
    sheets match _VAE_CHANNELS; the tiles use the same posneg mapping and
    per-model normalisation too.
    """
    import pyexr

    input_dir  = Path(input_dir).resolve()
    sheet_dir  = Path(output_dir).resolve() / "vae_channels"
    sheet_dir.mkdir(parents=True, exist_ok=True)

    sheets = []
    for scene in _VAE_CHANNELS:
        exr, _ = resolve_latent_exr(
            ViewLatentHref(scene, index, ltype, 0), input_dir)
        if not exr.exists():
            print(f"  [missing exr] {exr}")
            continue

        clip_max = latent_abs_percentile(exr, _VAE_PERCENTILE)
        ch_dir = sheet_dir / scene
        ch_dir.mkdir(parents=True, exist_ok=True)
        n_ch = len(pyexr.open(str(exr)).channels)

        tiles = []
        for ch in range(n_ch):
            png = ch_dir / f"ch{ch:02d}.png"
            latent_channel_png(exr, png, ch, upsample=2, gamma=_GAMMA,
                               clip_max=clip_max)
            tiles.append(Image.open(str(png)).convert("RGB"))

        cols  = 8
        rows  = (n_ch + cols - 1) // cols
        tw, th = tiles[0].size
        pad, label_h = 4, 16
        sheet = Image.new("RGB", (cols * (tw + pad) + pad,
                                  rows * (th + label_h + pad) + pad), (255, 255, 255))
        draw = ImageDraw.Draw(sheet)
        for i, tile in enumerate(tiles):
            x = pad + (i % cols) * (tw + pad)
            y = pad + (i // cols) * (th + label_h + pad)
            sheet.paste(tile, (x, y))
            draw.text((x + 2, y + th + 2), f"ch{i}", fill=(0, 0, 0))

        out = sheet_dir / f"{scene}_channels.png"
        sheet.save(str(out))
        print(f"{scene}: {n_ch} channels, clip=±{clip_max:.3f} → {out}")
        sheets.append(out)

    return sheets


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir",   required=True,            type=Path)
    p.add_argument("--output-dir",  default=Path("figures/"), type=Path)
    p.add_argument("--no-pdf",      action="store_true")
    p.add_argument("--dump-vae-channels", action="store_true",
                   help="Write per-VAE channel contact sheets and exit")
    args = p.parse_args()
    if args.dump_vae_channels:
        dump_vae_channels(args.input_dir, args.output_dir)
        return
    generate(args.input_dir, args.output_dir, export_pdf_flag=not args.no_pdf)
    generate_vae(args.input_dir, args.output_dir, export_pdf_flag=not args.no_pdf)


if __name__ == "__main__":
    main()
