#!/usr/bin/env bash
# Shared helpers for the reproduction scripts; source this file, then call parse_args and run_configs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
read -ra PYTHON <<< "${PYTHON:-uv run python}"

DRY_RUN=0
OVERRIDES=()

# Usage: parse_args [--dry-run] [--config_key value ...]
parse_args() {
    for arg in "$@"; do
        if [[ $arg == --dry-run ]]; then
            DRY_RUN=1
        else
            OVERRIDES+=("$arg")
        fi
    done
}

# Prints the status of a run config: done, test (checkpoints exist but tests don't), train, or missing (no scene file).
run_status() {
    "${PYTHON[@]}" - "$1" <<'EOF'
import sys
from pathlib import Path

sys.path.insert(0, "src")
from config import load_config

cfg = load_config(sys.argv[1])
out = Path("outputs") / cfg["out_dir"]


def tested(exp):
    exp_dir = out / exp
    return (exp_dir / f"{exp}.jsonl").exists() and any(exp_dir.glob("*.mp4"))


ckpts = [out / "scene.pt"] + ([out / "refiner.pt"] if cfg["use_refiner"] else [])
if not Path(cfg["scene_file"]).exists():
    print("missing")
elif all(tested(exp) for exp in cfg["experiments"]):
    print("done")
elif all(ckpt.exists() for ckpt in ckpts):
    print("test")
else:
    print("train")
EOF
}

# Trains and tests each run config in turn, skipping completed runs and resuming at testing when possible.
run_configs() {
    local cfg status failed=()
    for cfg in "$@"; do
        [[ $(basename "$cfg") == _* ]] && continue

        status=$(run_status "$cfg")
        local flags=()
        case $status in
            done)
                echo "[skip] $cfg (already complete)"
                continue
                ;;
            missing)
                echo "[fail] $cfg (scene file not found)"
                failed+=("$cfg")
                continue
                ;;
            test)
                flags=(--test_only)
                ;;
        esac

        echo "[run]  $cfg${flags:+ (testing only)}"
        local cmd=("${PYTHON[@]}" -m src.main --config "$cfg" "${flags[@]}" "${OVERRIDES[@]}")
        if (( DRY_RUN )); then
            echo "       ${cmd[*]}"
        elif ! "${cmd[@]}"; then
            echo "[fail] $cfg"
            failed+=("$cfg")
        fi
    done

    if (( ${#failed[@]} )); then
        echo
        echo "${#failed[@]} run(s) failed:"
        printf '  %s\n' "${failed[@]}"
        return 1
    fi
}
