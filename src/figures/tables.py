"""
JSONL → CSV + LaTeX tables.

Directory structure expected:
    root_output_dir/
        run_name/
            run_config.yaml
            experiment_name/
                experiment_name.jsonl

Run names are split on the first "-" to give (scene, variant):
    CBox-1024-Best  →  scene="cbox", variant="1024-Best"

Scene is confirmed / overridden from run_config.yaml's `scene_file` field
if the file is present.

Two LaTeX tables are emitted:
    table_mean.tex      — mean metrics across all frames
    table_keyframe.tex  — metrics at per-(scene, test_type) keyframes

Usage
-----
    python -m src.figures.tables --input-dir outputs/ --output-dir figures/
    python -m src.figures.tables --input-dir outputs/ --metric-group decoded_post_ambient
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Hardcoded keyframes: {(scene, test_type): step}
# Fill these in for each scene/test pair you care about.
# ---------------------------------------------------------------------------
KEYFRAMES: dict[tuple[str, str], int] = {
    # ("cbox",   "move_camera"): 30,
    # ("cbox",   "move_light"):  20,
    # ("cbox",   "move_object"): 15,
    # ("lamp",   "move_camera"): 10,
    # ("living", "move_light"):  25,
	("cbox", "move_light"): 59,
	("cbox", "move_camera"): 0,
	("cbox", "move_object"): 59,
	("lamp", "move_light"): 59,
	("lamp", "move_camera"): 30,
	("lamp", "move_object"): 59,
	("living", "move_light"): 59,
	("living", "move_camera"): 49,
	("living", "move_object"): 55,
	("dining", "move_light"): 59,
	("dining", "move_camera"): 30,
	("dining", "move_object"): 59,
	("veach", "move_light"): 59,
	("veach", "move_camera"): 59,
	("veach", "move_object"): 57,
	("bedroom",   "move_camera"): 59,
	("living2",   "move_camera"): 59,
	("classroom", "move_camera"): 59,
}

# Sub-dict keys to pull metrics from, keyed by a short suffix used in column names
METRIC_GROUPS: dict[str, str] = {
    "pre":   "decoded_post_ambient",   # pre-refinement
    "final": "decoded_final",          # post-refinement
}

# (jsonl_key, LaTeX display name, multiply-by scale, decimal places)
METRICS: list[tuple[str, str, float, int]] = [
    ("lpips", "LPIPS",           1.0,   2),
    ("mse",   r"MSE$\times$100", 100.0, 2),
]

# Ordered ablation columns: (variant_name, metric_suffix, LaTeX header)
ABLATION_COLS: list[tuple[str, str, str]] = [
    ("1024-NoAmb", "pre",   r"$\tilde{I}_i^e$ (\cref{eq:signedrender})"),
    ("1024-NoOcc", "pre",   r"$+\ \tilde{I}_i^a$ (\cref{eq:ambient})"),
    ("1024-Best",  "pre",   r"$+\ \tilde{I}_i^c$ (\cref{eq:occ})"),
    ("1024-Best",  "final", r"{+ Refiner}"),
]

# ---------------------------------------------------------------------------
# VAE ablation table: one scene + test type, one row per VAE.
#
# Same rendering-term columns as ABLATION_COLS, but the varying factor is the
# VAE rather than the scene.  Run naming differs between the two sources:
#   outputs/VAE/  →  <Model>-Lamp[-NoAmb|-NoOcc]   (full method = no suffix)
#   outputs/      →  Lamp-1024-{NoAmb|NoOcc|Best}  (full method = -Best)
# so each row carries its own prefix and full-method suffix.
# ---------------------------------------------------------------------------
VAE_TABLE_TEST = "move_camera"

# (display name, subdir under input_dir, run-name prefix, suffix of full-method run)
VAE_ROWS: list[tuple[str, str, str, str]] = [
    ("SD3.5",      "",    "Lamp-1024",  "-Best"),   # the default VAE (base.yaml)
    ("FLUX.1",     "VAE", "Flux1-Lamp", ""),
    ("FLUX.2",     "VAE", "Flux-Lamp",  ""),
    ("Qwen-Image", "VAE", "Qwen-Lamp",  ""),
]

# (run suffix — None means the row's full-method run, metric suffix, LaTeX header).
# Headers are taken from ABLATION_COLS so the two tables cannot drift apart.
VAE_ABLATION_COLS: list[tuple[str | None, str, str]] = [
    ("-NoAmb", "pre",   ABLATION_COLS[0][2]),
    ("-NoOcc", "pre",   ABLATION_COLS[1][2]),
    (None,     "pre",   ABLATION_COLS[2][2]),
    (None,     "final", ABLATION_COLS[3][2]),
]

SCENE_DISPLAY: dict[str, str] = {
    "cbox":   r"\makecell{Cornell\\Box}",
    "lamp":   "Lamp",
    "living": r"\makecell{Living\\Room}",
    "dining": "Dining Room",
    "veach":  "Veach Bidir",
}

TEST_DISPLAY: dict[str, str] = {
    "move_camera": r"Cam.",
    "move_object": r"Obj.",
    "move_light":  r"Light",
}

# Desired row order within each scene
TEST_ORDER = ["move_camera", "move_object", "move_light"]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _scene_from_yaml(run_dir: Path) -> str | None:
    cfg = run_dir / "run_config.yaml"
    if not cfg.exists():
        return None
    with cfg.open() as f:
        data = yaml.safe_load(f)
    # Ignore run configs this doesn't understand (e.g. from older versions of the code).
    scene_file = data.get("scene_file") if isinstance(data, dict) else None
    if not isinstance(scene_file, str) or not scene_file:
        return None
    p = Path(scene_file)
    stem = p.stem
    prefix = stem.split("_")[0].lower()
    if prefix == "scene":
        # e.g. scenes/lamp/scene_*.xml  →  use parent dir "lamp"
        # e.g. scenes/living-room-3/scene_*.xml  →  "living"
        return p.parent.name.split("-")[0].lower()
    return prefix


def _parse_run_name(run_name: str) -> tuple[str, str]:
    """'CBox-1024-Best' → ('cbox', '1024-Best')."""
    parts = run_name.split("-", 1)
    scene = parts[0].lower()
    variant = parts[1] if len(parts) > 1 else run_name
    return scene, variant


def discover_runs(input_dir: Path) -> list[tuple[str, str, str]]:
    """Return [(run_name, scene, variant)] for each run directory."""
    runs = []
    for run_dir in sorted(input_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        scene, variant = _parse_run_name(run_dir.name)
        # Prefer scene name from yaml if available
        scene = _scene_from_yaml(run_dir) or scene
        runs.append((run_dir.name, scene, variant))
    return runs


def discover_experiments(run_dir: Path) -> list[str]:
    """Return test-type names (subdirs containing a matching .jsonl file)."""
    exps = []
    for exp_dir in sorted(run_dir.iterdir()):
        if exp_dir.is_dir() and (exp_dir / f"{exp_dir.name}.jsonl").exists():
            exps.append(exp_dir.name)
    return exps


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _get_metric(record: dict, key: str, group: str) -> float | None:
    g = record.get(group)
    if not isinstance(g, dict):
        return None
    val = g.get(key)
    return float(val) if isinstance(val, (int, float)) else None


def collect_all_metrics(
    input_dir: Path,
    metric_group: str = METRIC_GROUPS,
) -> pd.DataFrame:
    """
    Returns a tidy DataFrame — one row per frame — with columns:
        run, scene, variant, test_type, step, <metric keys...>
    """
    metric_keys = [k for k, *_ in METRICS]
    rows = []
    for run_name, scene, variant in discover_runs(input_dir):
        run_dir = input_dir / run_name
        for test_type in discover_experiments(run_dir):
            jsonl = run_dir / test_type / f"{test_type}.jsonl"
            for record in _load_jsonl(jsonl):
                row = {
                    "run": run_name, "scene": scene,
                    "variant": variant, "test_type": test_type,
                    "step": record.get("step"),
                }
                for suffix, group in METRIC_GROUPS.items():
                    for key in metric_keys:
                        row[f"{key}_{suffix}"] = _get_metric(record, key, group)
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def build_mean_df(all_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        f"{key}_{suffix}"
        for suffix in METRIC_GROUPS
        for key, *_ in METRICS
    ]
    return (
        all_df
        .groupby(["run", "scene", "variant", "test_type"])[metric_cols]
        .mean()
        .reset_index()
    )



def build_keyframe_df(all_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        f"{key}_{suffix}"
        for suffix in METRIC_GROUPS
        for key, *_ in METRICS
    ]
    rows = []
    for (run, scene, variant, test_type), grp in all_df.groupby(
        ["run", "scene", "variant", "test_type"]
    ):
        step = KEYFRAMES.get((scene, test_type))
        if step is None:
            continue
        frame = grp[grp["step"] == step]
        if frame.empty:
            print(f"  warning: step {step} not found for {run}/{test_type}")
            continue
        rec = {"run": run, "scene": scene, "variant": variant, "test_type": test_type}
        for col in metric_cols:
            rec[col] = frame.iloc[0][col]
        rows.append(rec)
    return pd.DataFrame(rows)



# ---------------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------------

# def _fmt(val: float | None, scale: float, decimals: int) -> str:
#     if val is None or (isinstance(val, float) and pd.isna(val)):
#         return "---"
#     return f"{val * scale:.{decimals}f}"


def _fmt_cell(row: pd.Series, suffix: str) -> str:
    lpips = row.get(f"lpips_{suffix}")
    mse   = row.get(f"mse_{suffix}")
    if lpips is None or mse is None or pd.isna(lpips) or pd.isna(mse):
        return "---"
    return f"{lpips:.3f} / {mse * 100:.3f}"


def build_latex_table(
    df: pd.DataFrame,
    caption: str = "Quantitative results.",
    label: str = "tab:results",
    frame_lookup: dict[tuple[str, str], int] | None = None,
) -> str:
    if df.empty:
        return "% No data available\n"

    scenes = [s for s in ["cbox", "lamp", "living", "dining", "veach"]
              if s in df["scene"].unique()]

    col_spec = "ll" + "c" * len(ABLATION_COLS)
    col_headers = " & ".join(
        [r"{Scene}", r"{Variants}"] + [hdr for _, _, hdr in ABLATION_COLS]
    )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        col_headers + r" \\",
        r"\midrule",
    ]

    for si, scene in enumerate(scenes):
        scene_df = df[df["scene"] == scene]
        # Collect present test types in desired order
        test_types = [t for t in TEST_ORDER if t in scene_df["test_type"].unique()]
        n_rows = len(test_types)
        scene_label = SCENE_DISPLAY.get(scene, scene.title())

        for ti, test_type in enumerate(test_types):
            row_df = scene_df[scene_df["test_type"] == test_type]
            test_label = TEST_DISPLAY.get(test_type, test_type)
            test_label = TEST_DISPLAY.get(test_type, test_type)
            if frame_lookup is not None:
                step = frame_lookup.get((scene, test_type))
                if step is not None:
                    test_label += rf"\ (t\,=\,{step})"

            if ti == 0:
                scene_cell = rf"\multirow{{{n_rows}}}{{*}}{{\rotatebox{{0}}{{{scene_label}}}}}"
            else:
                scene_cell = ""

            cells = [scene_cell, test_label]
            for variant, suffix, _ in ABLATION_COLS:
                sub = row_df[row_df["variant"] == variant]
                if sub.empty:
                    cells.append("---")
                else:
                    cells.append(_fmt_cell(sub.iloc[0], suffix))

            lines.append(" & ".join(cells) + r" \\")

        if si < len(scenes) - 1:
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{-2mm}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"

def build_keyframe_latex_table(
    df: pd.DataFrame,
    caption: str = "Metrics at keyframes.",
    label: str = "tab:keyframe",
) -> str:
    """Keyframe table: one row per (scene, test_type), Best variant only, pre+post refinement."""
    if df.empty:
        return "% No data available\n"

    scenes = [s for s in ["cbox", "lamp", "living", "dining", "veach"]
              if s in df["scene"].unique()]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"{Scene} & {Variants} & {Frame} & "
        r"$+\ \tilde{I}_i^c$ (\cref{eq:occ}) & {+ Refiner} \\",
        r"\midrule",
    ]

    for si, scene in enumerate(scenes):
        scene_df = df[(df["scene"] == scene) & (df["variant"] == "1024-Best")]
        test_types = [t for t in TEST_ORDER if t in scene_df["test_type"].unique()]
        n_rows = len(test_types)
        scene_label = SCENE_DISPLAY.get(scene, scene.title())

        for ti, test_type in enumerate(test_types):
            row_df = scene_df[scene_df["test_type"] == test_type]
            test_label = TEST_DISPLAY.get(test_type, test_type)
            step = KEYFRAMES.get((scene, test_type), "---")

            scene_cell = (
                rf"\multirow{{{n_rows}}}{{*}}{{\rotatebox{{0}}{{{scene_label}}}}}"
                if ti == 0 else ""
            )

            if row_df.empty:
                pre_cell = final_cell = "---"
            else:
                pre_cell   = _fmt_cell(row_df.iloc[0], "pre")
                final_cell = _fmt_cell(row_df.iloc[0], "final")

            lines.append(
                f"{scene_cell} & {test_label} & {step} & {pre_cell} & {final_cell} \\\\"
            )

        if si < len(scenes) - 1:
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{-2mm}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def _mean_run_metrics(jsonl: Path) -> dict[str, float] | None:
    """Mean of every METRICS key over all frames in one run's jsonl, or None."""
    if not jsonl.exists():
        return None
    records = _load_jsonl(jsonl)
    if not records:
        return None
    means: dict[str, float] = {}
    for suffix, group in METRIC_GROUPS.items():
        for key, *_ in METRICS:
            vals = [v for r in records
                    if (v := _get_metric(r, key, group)) is not None]
            means[f"{key}_{suffix}"] = sum(vals) / len(vals) if vals else None
    return means


def build_vae_df(input_dir: Path) -> pd.DataFrame:
    """One row per (VAE, ablation column) with mean metrics for VAE_TABLE_TEST."""
    rows = []
    for display, subdir, prefix, full_suffix in VAE_ROWS:
        base = input_dir / subdir if subdir else input_dir
        for col_suffix, metric_suffix, _ in VAE_ABLATION_COLS:
            run = prefix + (full_suffix if col_suffix is None else col_suffix)
            jsonl = base / run / VAE_TABLE_TEST / f"{VAE_TABLE_TEST}.jsonl"
            means = _mean_run_metrics(jsonl)
            if means is None:
                print(f"  warning: no data for {jsonl}")
                continue
            rows.append({
                "vae": display, "run": run, "test_type": VAE_TABLE_TEST,
                "column": col_suffix or "full", "metric_suffix": metric_suffix,
                **means,
            })
    return pd.DataFrame(rows)


def build_vae_latex_table(
    df: pd.DataFrame,
    caption: str = "VAE ablation.",
    label: str = "tab:vae",
) -> str:
    """Rendering-term ablation with one row per VAE, for a single scene/test."""
    if df.empty:
        return "% No data available\n"

    col_spec = "l" + "c" * len(VAE_ABLATION_COLS)
    col_headers = " & ".join([r"{VAE}"] + [hdr for _, _, hdr in VAE_ABLATION_COLS])

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        col_headers + r" \\",
        r"\midrule",
    ]

    for display, *_ in VAE_ROWS:
        cells = [display]
        for col_suffix, metric_suffix, _ in VAE_ABLATION_COLS:
            sub = df[(df["vae"] == display)
                     & (df["column"] == (col_suffix or "full"))
                     & (df["metric_suffix"] == metric_suffix)]
            cells.append("---" if sub.empty else _fmt_cell(sub.iloc[0], metric_suffix))
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{-2mm}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def make_tables(
    input_dir: Path | str,
    output_dir: Path | str,
) -> None:
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_df = collect_all_metrics(input_dir)
    if all_df.empty:
        print(f"tables: no test results found in {input_dir}, skipping")
        return
    all_df.to_csv(output_dir / "metrics_all.csv", index=False)
    print(f"tables: raw data  → {output_dir / 'metrics_all.csv'}")

    # Mean table
    mean_df = build_mean_df(all_df)
    mean_df.to_csv(output_dir / "metrics_mean.csv", index=False)
    tex = build_latex_table(mean_df, "Mean metrics across all frames.", "tab:mean")
    (output_dir / "table_mean.tex").write_text(tex, encoding="utf-8")
    print(f"tables: mean table → {output_dir / 'table_mean.tex'}")

    # VAE ablation table (single scene/test, one row per VAE).
    # Placed before the keyframe block, which returns early when KEYFRAMES is empty.
    vae_df = build_vae_df(input_dir)
    if vae_df.empty:
        print("tables: no VAE runs found — skipping VAE table")
    else:
        vae_df.to_csv(output_dir / "metrics_vae.csv", index=False)
        tex = build_vae_latex_table(
            vae_df,
            caption=(r"VAE ablation over rendering terms, Lamp / camera motion, "
                     r"mean over all frames.  LPIPS / MSE$\times 100$."),
            label="tab:vae",
        )
        (output_dir / "table_vae.tex").write_text(tex, encoding="utf-8")
        print(f"tables: VAE table → {output_dir / 'table_vae.tex'}")

    # Keyframe table
    if not KEYFRAMES:
        print("tables: KEYFRAMES dict is empty — skipping keyframe table")
        return
    key_df = build_keyframe_df(all_df)
    key_df.to_csv(output_dir / "metrics_keyframe.csv", index=False)
    tex = build_keyframe_latex_table(
        key_df,
        caption=r"Per-scene keyframe LPIPS / MSE$\times 100$ for the full pipeline, pre- and post-refinement.",
        label="tab:keyframe",
    )
    (output_dir / "table_keyframe.tex").write_text(tex, encoding="utf-8")
    print(f"tables: keyframe table → {output_dir / 'table_keyframe.tex'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir",    required=True, type=Path)
    p.add_argument("--output-dir",   default=Path("figures/"), type=Path)
    # p.add_argument("--metric-group", default=METRIC_GROUPS,
    #                help="JSONL sub-dict key to read metrics from")
    args = p.parse_args()
    make_tables(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()