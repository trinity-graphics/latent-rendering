"""
JSONL → pandas DataFrame helpers for figure scripts and tables.py.

The main codebase writes two kinds of JSONL files:
  - <run>/metrics.jsonl          training metrics, one JSON object per iteration
  - <run>/<experiment>.jsonl     test metrics, one JSON object per test frame

Both are newline-delimited JSON; each line is a flat dict.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_jsonl(path: Path | str) -> pd.DataFrame:
    """Read a JSONL file into a DataFrame (one row per line)."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def aggregate_training_metrics(
    input_dir: Path | str,
    run_name: str,
) -> pd.DataFrame:
    """Load per-iteration training metrics for one run.

    Returns a DataFrame with columns like: iteration, loss, mse, lpips, flip, ...
    """
    path = Path(input_dir) / run_name / "metrics.jsonl"
    return load_jsonl(path)


def aggregate_test_metrics(
    input_dir: Path | str,
    run_name: str,
    test_experiment: str | None = None,
) -> pd.DataFrame:
    """Load per-frame test metrics for one run.

    If test_experiment is None, discovers the first *.jsonl file in the run
    dir that is not 'metrics.jsonl'.

    Returns a DataFrame with columns like:
      frame_id, test_type, lpips, prerefine_lpips, mse, prerefine_mse, ...
    """
    run_dir = Path(input_dir) / run_name
    if test_experiment is not None:
        path = run_dir / f"{test_experiment}.jsonl"
    else:
        candidates = [
            p for p in run_dir.glob("*.jsonl")
            if p.name != "metrics.jsonl"
        ]
        if not candidates:
            raise FileNotFoundError(f"No test JSONL found in {run_dir}")
        path = candidates[0]
    return load_jsonl(path)


def get_frame_metrics(
    df: pd.DataFrame,
    frame_id: int,
    test_type: str | None = None,
) -> pd.Series:
    """Extract a single row for (frame_id[, test_type]) from a test metrics DataFrame.

    The JSONL format stores frame_id in a column named 'frame_id' or 'it'.
    test_type is stored in a 'test_type' column when present.
    """
    id_col = "frame_id" if "frame_id" in df.columns else "it"
    mask = df[id_col] == frame_id
    if test_type is not None and "test_type" in df.columns:
        mask &= df["test_type"] == test_type
    rows = df[mask]
    if rows.empty:
        raise KeyError(f"No row for frame_id={frame_id}, test_type={test_type!r}")
    return rows.iloc[0]
