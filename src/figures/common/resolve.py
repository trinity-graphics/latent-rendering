"""
Parse standardised SVG href strings into actual file paths.

Href scheme (baked into SVG templates):
  runs/{scene}/{test_type}/{frame}/{variant}.png
  debug/{scene}/{frame}/{variant}.png
  latents/{scene}/{test_type}/{frame}/{ltype}/ch{N}.png
  debug_latents/{scene}/{frame}/{ltype}/ch{N}.png
  view_latents/{scene}/{index}/{ltype}/ch{N}.png
  crops/{scene}/{test_type}/{frame}/{ltype}/ch{N}_b{M}.png

scene       — lowercase label (cbox, lamp, living, dining, veach, lamp_naive, …)
test_type   — move_camera | move_light | move_object | move_all
frame       — integer frame number
index       — reference-view index (views/ holds one set per view)
variant     — flat image variant key (final, post_ambient, best_rgb, …)
ltype       — latent type key (latent_ref, processsed, refined, …)
              for view_latents: latent_sample | latent_processed
N           — EXR channel index (integer)
M           — crop box index (integer)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

from src.figures.common.data_loader import (
    _DEBUG_VARIANTS,
    _TEST_VARIANTS,
    LATENT_TYPE_TO_FILE,
    SCENE_TO_RUN,
    VIEW_LATENT_TYPE_TO_FILE,
)

# ---------------------------------------------------------------------------
# Parsed href result types
# ---------------------------------------------------------------------------

class RunsHref(NamedTuple):
    scene: str
    test_type: str
    frame: int
    variant: str

class DebugHref(NamedTuple):
    scene: str
    frame: int
    variant: str

class LatentHref(NamedTuple):
    scene: str
    test_type: str
    frame: int
    ltype: str
    channel: int

class DebugLatentHref(NamedTuple):
    scene: str
    frame: int
    ltype: str
    channel: int

class ViewLatentHref(NamedTuple):
    scene: str
    index: int
    ltype: str
    channel: int

class CropHref(NamedTuple):
    scene: str
    test_type: str
    frame: int
    ltype: str
    channel: int
    crop_idx: int

_RE_RUNS        = re.compile(r'^runs/([^/]+)/([^/]+)/(\d+)/([^/]+)\.png$')
_RE_DEBUG       = re.compile(r'^debug/([^/]+)/(\d+)/([^/]+)\.png$')
_RE_LATENTS     = re.compile(r'^latents/([^/]+)/([^/]+)/(\d+)/([^/]+)/ch(\d+)\.png$')
_RE_DBG_LAT     = re.compile(r'^debug_latents/([^/]+)/(\d+)/([^/]+)/ch(\d+)\.png$')
_RE_VIEW_LAT    = re.compile(r'^view_latents/([^/]+)/(\d+)/([^/]+)/ch(\d+)\.png$')
_RE_CROPS       = re.compile(r'^crops/([^/]+)/([^/]+)/(\d+)/([^/]+)/ch(\d+)_b(\d+)\.png$')


def parse_href(href: str):
    """Parse a standardised href into a typed NamedTuple, or None if unrecognised."""
    m = _RE_RUNS.match(href)
    if m:
        return RunsHref(m[1], m[2], int(m[3]), m[4])
    m = _RE_DEBUG.match(href)
    if m:
        return DebugHref(m[1], int(m[2]), m[3])
    m = _RE_LATENTS.match(href)
    if m:
        return LatentHref(m[1], m[2], int(m[3]), m[4], int(m[5]))
    m = _RE_DBG_LAT.match(href)
    if m:
        return DebugLatentHref(m[1], int(m[2]), m[3], int(m[4]))
    m = _RE_VIEW_LAT.match(href)
    if m:
        return ViewLatentHref(m[1], int(m[2]), m[3], int(m[4]))
    m = _RE_CROPS.match(href)
    if m:
        return CropHref(m[1], m[2], int(m[3]), m[4], int(m[5]), int(m[6]))
    return None


# ---------------------------------------------------------------------------
# Resolve parsed hrefs to filesystem paths
# ---------------------------------------------------------------------------

def resolve_flat(parsed: RunsHref | DebugHref, input_dir: Path) -> Path:
    """Resolve a RunsHref or DebugHref to its output file path."""
    run = SCENE_TO_RUN.get(parsed.scene, parsed.scene)
    if isinstance(parsed, RunsHref):
        tmpl = _TEST_VARIANTS[parsed.variant]
        fname = tmpl.format(frame=parsed.frame)
        return input_dir / run / parsed.test_type / "images" / fname
    else:
        tmpl = _DEBUG_VARIANTS[parsed.variant]
        fname = tmpl.format(frame=parsed.frame)
        return input_dir / run / "debug" / fname


def resolve_latent_exr(
    parsed: LatentHref | DebugLatentHref | ViewLatentHref,
    input_dir: Path,
) -> tuple[Path, int]:
    """Resolve a latent href to (exr_path, channel_index)."""
    run = SCENE_TO_RUN.get(parsed.scene, parsed.scene)
    if isinstance(parsed, ViewLatentHref):
        fname = VIEW_LATENT_TYPE_TO_FILE[parsed.ltype].format(index=parsed.index)
        return input_dir / run / "views" / fname, parsed.channel
    tmpl = LATENT_TYPE_TO_FILE[parsed.ltype]
    fname = tmpl.format(frame=parsed.frame)
    if isinstance(parsed, LatentHref):
        exr = input_dir / run / parsed.test_type / "images" / fname
    else:
        exr = input_dir / run / "debug" / fname
    return exr, parsed.channel


def resolve_crop_exr(parsed: CropHref, input_dir: Path) -> tuple[Path, int]:
    """Resolve a crop href to (exr_path, channel_index). The crop box index is embedded in the href."""
    run = SCENE_TO_RUN.get(parsed.scene, parsed.scene)
    tmpl = LATENT_TYPE_TO_FILE[parsed.ltype]
    fname = tmpl.format(frame=parsed.frame)
    exr = input_dir / run / parsed.test_type / "images" / fname
    return exr, parsed.channel


# ---------------------------------------------------------------------------
# Convenience: resolve any href to a path (flat images) or EXR+channel
# ---------------------------------------------------------------------------

def resolve_href(href: str, input_dir: Path) -> Path | tuple[Path, int] | None:
    """
    Resolve any standardised href.

    Returns:
      Path            — for flat (runs/debug) images
      (Path, int)     — for latent/crop hrefs (EXR path + channel index)
      None            — if the href is not in the standardised scheme
    """
    parsed = parse_href(href)
    if parsed is None:
        return None
    if isinstance(parsed, (RunsHref, DebugHref)):
        return resolve_flat(parsed, input_dir)
    if isinstance(parsed, (LatentHref, DebugLatentHref, ViewLatentHref)):
        return resolve_latent_exr(parsed, input_dir)
    if isinstance(parsed, CropHref):
        return resolve_crop_exr(parsed, input_dir)
    return None
