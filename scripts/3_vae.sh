#!/usr/bin/env bash
# VAE study: FLUX.1, FLUX.2 and Qwen-Image VAEs on the Cornell box and lamp scenes.
# Usage: scripts/3_vae.sh [--dry-run] [--config_key value ...]
source "$(dirname "$0")/common.sh"
parse_args "$@"
run_configs configs/runs/vae/*.yaml
