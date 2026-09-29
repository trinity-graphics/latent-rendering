"""Paper-matching fonts, and PDF output the camera-ready font check accepts.

The submission system accepts only embedded Type 1 or TrueType fonts.  Both PDF
writers used here default to something else:

  * matplotlib's PDF backend writes Type 3 fonts unless ``pdf.fonttype`` is 42,
    and with 42 it writes a CID TrueType subset;
  * Inkscape (cairo) writes a Type 1C subset *plus* a CID Type 0C subset of the
    same face, because ligatures ("fi" in "final", "ffi" in "efficient") are
    shaped to glyph ids that have no simple encoding.  That CID Type 0C subset
    is what the checker rejects, and it appears for every font — swapping the
    typeface does not avoid it.

Rather than gamble on which subset form a backend happens to emit, every PDF
this package writes has its text converted to vector outlines: Inkscape via
``--export-text-to-path`` (see ``svg_utils.export_pdf``), matplotlib via a
Ghostscript ``-dNoOutputFonts`` pass (see ``outline_pdf`` below).  The result
embeds no fonts at all, so no font check can fail.

Glyph shapes still come from Nimbus — URW's Times/Helvetica clones, the same
family the paper body text sets — so figure text matches the surrounding text.
"""

from __future__ import annotations

import shutil
import subprocess
import warnings
from pathlib import Path

# Fallbacks after the Nimbus entries only matter on a machine without
# fonts-urw-base35 installed; Nimbus Roman is metrically a Times clone.
SERIF = ["Nimbus Roman", "Nimbus Roman No9 L", "Times New Roman", "Times", "DejaVu Serif"]
SANS  = ["Nimbus Sans", "Nimbus Sans L", "Helvetica", "DejaVu Sans"]
MONO  = ["Nimbus Mono PS", "Nimbus Mono L", "Courier", "DejaVu Sans Mono"]

#: rcParams shared by every matplotlib figure in the paper.
PAPER_RC: dict = {
    "font.family":      "serif",
    "font.serif":       SERIF,
    "font.sans-serif":  SANS,
    "font.monospace":   MONO,
    # Nimbus Roman for mathtext too, so $...$ labels do not fall back to DejaVu.
    "mathtext.fontset": "custom",
    "mathtext.rm":      "Nimbus Roman",
    "mathtext.it":      "Nimbus Roman:italic",
    "mathtext.bf":      "Nimbus Roman:bold",
    # Belt and braces: if Ghostscript is unavailable and text is left as text,
    # 42 still beats the Type 3 default, which no checker accepts.
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
    "svg.fonttype":     "none",
}


def apply(**overrides) -> None:
    """Install the paper rcParams, plus any per-figure overrides."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(PAPER_RC)
    if overrides:
        plt.rcParams.update(overrides)


def outline_pdf(pdf_path: Path | str) -> bool:
    """Convert all text in a PDF to vector outlines, in place.

    Images are passed through uncompressed-as-found: downsampling and JPEG
    re-encoding are disabled so the rendered image grids are untouched.
    Returns True if the file was rewritten, False (with a warning) if
    Ghostscript is not installed.
    """
    pdf_path = Path(pdf_path)
    gs = shutil.which("gs")
    if gs is None:
        warnings.warn(
            f"Ghostscript not found; {pdf_path.name} keeps its embedded fonts. "
            "Install ghostscript so figure text is outlined for the camera-ready.",
            stacklevel=2,
        )
        return False

    tmp = pdf_path.with_suffix(".outlined.tmp.pdf")
    cmd = [
        gs, "-q", "-dNOPAUSE", "-dBATCH",
        "-sDEVICE=pdfwrite",
        "-dNoOutputFonts",                        # text -> paths
        "-dDownsampleColorImages=false",
        "-dDownsampleGrayImages=false",
        "-dDownsampleMonoImages=false",
        "-dAutoFilterColorImages=false",
        "-dAutoFilterGrayImages=false",
        "-dColorImageFilter=/FlateEncode",        # lossless, as matplotlib wrote them
        "-dGrayImageFilter=/FlateEncode",
        "-dColorConversionStrategy=/LeaveColorUnchanged",
        f"-sOutputFile={tmp}",
        str(pdf_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Ghostscript outlining failed for {pdf_path}:\n{result.stderr}")
    tmp.replace(pdf_path)
    return True


def savefig(fig, out_path: Path | str, **kwargs) -> Path:
    """``fig.savefig`` that outlines the text when the target is a PDF."""
    out_path = Path(out_path)
    fig.savefig(out_path, **kwargs)
    if out_path.suffix.lower() == ".pdf":
        outline_pdf(out_path)
    return out_path
