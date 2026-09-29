r"""
SVG href rewriting utilities.

SVG templates store image paths as xlink:href attributes on <image> elements.
These helpers load a template, replace those paths with absolute (or relative)
paths pointing at the current run data, and save the result.

Usage
-----
    tree = load_svg(template_path)
    for elem, old_href in find_images_by_pattern(tree, r"paper_data/.*/CBox/move_camera_\d+\.png"):
        set_image_href(elem, "/abs/path/to/new_image.png")
    save_svg(tree, output_path)
"""

from __future__ import annotations

import base64
import mimetypes
import re
import subprocess
from collections.abc import Iterator
from pathlib import Path
from xml.etree import ElementTree as ET

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

# SVG / Inkscape namespace declarations
# "" = default namespace so ElementTree writes <svg> not <svg:svg>
_NS = {
    "":        "http://www.w3.org/2000/svg",
    "xlink":   "http://www.w3.org/1999/xlink",
    "inkscape":"http://www.inkscape.org/namespaces/inkscape",
    "sodipodi":"http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd",
    "dc":      "http://purl.org/dc/elements/1.1/",
    "cc":      "http://creativecommons.org/ns#",
    "rdf":     "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
}

_HREF_ATTR = "{http://www.w3.org/1999/xlink}href"
_IMAGE_TAG = "{http://www.w3.org/2000/svg}image"


def _register_namespaces() -> None:
    for prefix, uri in _NS.items():
        ET.register_namespace(prefix, uri)


def load_svg(template_path: Path | str) -> ET.ElementTree:
    """Parse an SVG file, preserving all namespace declarations."""
    _register_namespaces()
    return ET.parse(str(template_path))


def get_image_href(elem: ET.Element) -> str | None:
    """Return the xlink:href value of an <image> element, or None."""
    return elem.get(_HREF_ATTR)


def set_image_href(elem: ET.Element, new_path: str | Path) -> None:
    """Update the xlink:href of an <image> element."""
    elem.set(_HREF_ATTR, str(new_path))


def find_images_by_pattern(
    tree: ET.ElementTree,
    pattern: str,
) -> Iterator[tuple[ET.Element, str]]:
    """Yield (element, current_href) for every <image> whose href matches pattern."""
    compiled = re.compile(pattern)
    for elem in tree.iter(_IMAGE_TAG):
        href = get_image_href(elem)
        if href and compiled.search(href):
            yield elem, href


def find_all_images(tree: ET.ElementTree) -> Iterator[tuple[ET.Element, str]]:
    """Yield (element, href) for every <image> element that has an href."""
    for elem in tree.iter(_IMAGE_TAG):
        href = get_image_href(elem)
        if href:
            yield elem, href


_NO_INDENT_TAGS = frozenset({
    "{http://www.w3.org/2000/svg}text",
    "{http://www.w3.org/2000/svg}tspan",
})


def _safe_indent(elem: ET.Element, level: int = 0, space: str = "  ") -> None:
    """Like ET.indent() but never adds whitespace inside <text>/<tspan> elements."""
    if elem.tag in _NO_INDENT_TAGS:
        return
    children = list(elem)
    if not children:
        return
    child_prefix = "\n" + space * (level + 1)
    my_prefix    = "\n" + space * level
    if not (elem.text and elem.text.strip()):
        elem.text = child_prefix
    for i, child in enumerate(children):
        is_last = i == len(children) - 1
        if not (child.tail and child.tail.strip()):
            child.tail = my_prefix if is_last else child_prefix
        if child.tag not in _NO_INDENT_TAGS:
            _safe_indent(child, level + 1, space)


def save_svg(tree: ET.ElementTree, output_path: Path | str) -> None:
    """Write the (modified) SVG tree to disk with human-readable indentation."""
    _register_namespaces()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    _safe_indent(tree.getroot())
    tree.write(str(output_path), encoding="utf-8", xml_declaration=True)


def embed_images(tree: ET.ElementTree) -> int:
    """Replace absolute file-path hrefs with base64 data URIs.

    Makes the SVG self-contained so it renders correctly in VSCode, browsers,
    and any viewer that blocks external file:// references.
    Returns the number of images embedded.
    """
    count = 0
    for elem in tree.iter(_IMAGE_TAG):
        href = elem.get(_HREF_ATTR)
        if not href:
            continue
        path_str = href.removeprefix("file://")
        p = Path(path_str)
        if p.is_absolute() and p.exists():
            mime = mimetypes.guess_type(str(p))[0] or "image/png"
            data = base64.b64encode(p.read_bytes()).decode()
            elem.set(_HREF_ATTR, f"data:{mime};base64,{data}")
            count += 1
    return count


def embed_svg_file(svg_path: Path | str) -> int:
    """Load an SVG, embed all resolvable local images as base64, overwrite in place."""
    tree = load_svg(svg_path)
    count = embed_images(tree)
    if count:
        save_svg(tree, svg_path)
    return count


def export_pdf(svg_path: Path | str, pdf_path: Path | str,
               text_to_path: bool = True) -> None:
    """Export an SVG to PDF via the Inkscape CLI (requires Inkscape ≥ 1.0).

    ``text_to_path`` converts every glyph to a vector outline so the PDF embeds
    no fonts.  It is on by default because Inkscape's cairo exporter otherwise
    emits a CID Type 0C subset alongside the Type 1C one whenever a ligature is
    shaped, and the camera-ready font check rejects CID Type 0C.  See
    ``common/fonts.py``.
    """
    cmd = [
        "inkscape",
        f"--export-filename={pdf_path}",
        "--export-type=pdf",
        *(["--export-text-to-path"] if text_to_path else []),
        str(svg_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Inkscape export failed:\n{result.stderr}")


# ---------------------------------------------------------------------------
# Metric placeholder substitution
# ---------------------------------------------------------------------------

# Anchor on `_move_` so scene names with underscores (lamp_naive) are parsed correctly.
_PLACEHOLDER_RE = re.compile(r'\{([a-z0-9_]+)_(move_[a-z]+)_(pre|final)_(lpips|mse|msex100)_f(\d+)\}')
_TSPAN_TAG  = "{http://www.w3.org/2000/svg}tspan"
_TEXT_TAG   = "{http://www.w3.org/2000/svg}text"
_DECIMAL_RE = re.compile(r'^[\d.]+$')


def substitute_metric_placeholders(
    tree: ET.ElementTree,
    metrics_csv_path: Path | str,
) -> int:
    """Replace {scene_testtype_stage_metric_frame} placeholders in SVG tspan text.

    Placeholder format: {scene}_{test_type}_{stage}_{metric}_f{N}
      scene     — cbox | lamp | lamp_naive | living | dining | veach
      test_type — move_camera | move_light | move_object | move_all
      stage     — pre | final
      metric    — lpips | mse | msex100
      frame     — f<N> (specific frame step)

    Handles the two-line template format: a tspan ending with ` /` followed by a
    sibling tspan containing the msex100 value.  After substitution those pairs
    are merged into a single `value / value` tspan and the font is restored to
    its original (double of the half-size used in the template).

    Metrics CSV columns: run, scene, variant, test_type, step, lpips_pre, mse_pre,
                         lpips_final, mse_final

    Returns the number of placeholder substitutions made.
    """
    if not _HAS_PANDAS:
        raise ImportError("pandas is required for substitute_metric_placeholders")

    import pandas as pd

    from src.figures.common.data_loader import SCENE_TO_RUN
    all_metrics = pd.read_csv(str(metrics_csv_path), skipinitialspace=True)

    count = 0

    def _lookup(scene: str, test_type: str, stage: str, metric: str, frame_str: str) -> str:
        col = f"{'mse' if metric == 'msex100' else metric}_{stage}"
        scale = 100.0 if metric == "msex100" else 1.0
        run = SCENE_TO_RUN.get(scene, scene)
        sub = all_metrics[all_metrics["run"] == run]
        if sub.empty:
            sub = all_metrics[(all_metrics["scene"] == scene)
                              & (all_metrics["test_type"] == test_type)]
        else:
            sub = sub[sub["test_type"] == test_type]
        if sub.empty:
            return "---"
        if "step" in all_metrics.columns:
            step = int(frame_str)
            row = sub[sub["step"] == step]
            if row.empty:
                return "---"
        else:
            row = sub
        val = row.iloc[0][col]
        if pd.isna(val):
            return "---"
        return f"{val * scale:.3f}"

    substituted: set[int] = set()

    for elem in tree.iter(_TSPAN_TAG):
        txt = elem.text or ""
        def _replace(m: re.Match) -> str:
            nonlocal count
            result = _lookup(m.group(1), m.group(2), m.group(3), m.group(4), m.group(5))
            count += 1
            return result
        new_txt = _PLACEHOLDER_RE.sub(_replace, txt)
        if new_txt != txt:
            elem.text = new_txt
            substituted.add(id(elem))

    # ── Merge two-line pairs and restore font size ────────────────────────────
    # Templates store each metric as two sibling tspans:
    #   t1.text = "0.123 /"   (lpips, half font-size)
    #   t2.text = "4.567"     (msex100, half font-size)
    # After substitution, merge into "0.123 / 4.567" at original font-size.
    for text_elem in tree.iter(_TEXT_TAG):
        tspans = [c for c in text_elem if c.tag == _TSPAN_TAG]
        to_remove = []
        i = 0
        while i < len(tspans) - 1:
            t1, t2 = tspans[i], tspans[i + 1]
            if id(t1) not in substituted and id(t2) not in substituted:
                i += 1
                continue
            v1 = (t1.text or "").strip()
            v2 = (t2.text or "").strip()
            if v1.endswith("/") and _DECIMAL_RE.match(v2.rstrip("---")):
                merged = v1[:-1].rstrip() + " / " + v2
                t1.text = merged
                # Restore font: template uses half-size, so double it
                style = t1.attrib.get("style", "")
                m = re.search(r"font-size:([\d.]+)px", style)
                if m:
                    restored = float(m.group(1)) * 2
                    t1.attrib["style"] = style.replace(
                        m.group(0), f"font-size:{restored:.5g}px"
                    )
                to_remove.append(t2)
                i += 2
                continue
            i += 1
        for t in to_remove:
            text_elem.remove(t)

    return count


# ---------------------------------------------------------------------------
# Higher-level: replace hrefs by regex → path mapping
# ---------------------------------------------------------------------------

def apply_href_map(
    tree: ET.ElementTree,
    mapping: dict[str, str | Path],
    verbose: bool = False,
) -> int:
    """Replace hrefs according to a {old_href_pattern: new_path} dict.

    Each key is a regex; the first matching key wins.
    Returns the total number of replacements made.
    """
    compiled = [(re.compile(pat), str(new)) for pat, new in mapping.items()]
    count = 0
    for elem, href in find_all_images(tree):
        for pattern, new_path in compiled:
            if pattern.search(href):
                if verbose:
                    print(f"  {href!r}\n    -> {new_path!r}")
                set_image_href(elem, new_path)
                count += 1
                break
    return count
