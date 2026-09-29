#!/usr/bin/env bash
# Textured interior scenes (bedroom, classroom, living room 2).
# Usage: scripts/4_textured.sh [--dry-run] [--config_key value ...]
source "$(dirname "$0")/common.sh"
parse_args "$@"
run_configs configs/runs/textured/*.yaml
