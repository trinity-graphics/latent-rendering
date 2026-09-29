#!/usr/bin/env bash
# Naive baseline (standard PRB rendering into the latent space) on the lamp scene.
# Usage: scripts/2_naive.sh [--dry-run] [--config_key value ...]
source "$(dirname "$0")/common.sh"
parse_args "$@"
run_configs configs/runs/naive/*.yaml
