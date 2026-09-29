#!/usr/bin/env bash
# Main ablation (Cornell box, lamp, living room) and the extra scenes (dining room, Veach bidir).
# Usage: scripts/1_main.sh [--dry-run] [--config_key value ...]
source "$(dirname "$0")/common.sh"
parse_args "$@"
run_configs configs/runs/ablation/*.yaml configs/runs/extra/*.yaml
