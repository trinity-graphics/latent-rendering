#!/usr/bin/env bash
# Generates the figures and tables from outputs/ into figures/; PDF export needs Inkscape.
# Usage: scripts/figures.sh [generate_all args ...]
source "$(dirname "$0")/common.sh"
args=("$@")
if ! command -v inkscape >/dev/null; then
    echo "Inkscape not found, writing SVGs only."
    args+=(--no-pdf)
fi
"${PYTHON[@]}" -m src.figures.generate_all --input-dir outputs --output-dir figures "${args[@]}"
echo
echo "To browse the result videos, run \`python -m http.server\` and open http://localhost:8000/outputs/results.html"
