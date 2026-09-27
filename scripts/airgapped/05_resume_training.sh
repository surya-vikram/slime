#!/usr/bin/env bash
# ==============================================================================
# 05_resume_training.sh - Resume an Interrupted Chimera MixRL Run
# ==============================================================================
set -Eeuo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <RUN_NAME> [additional options]" >&2
    echo "Example: $0 chimera-mixrl-4gpus-512x8-20260927-140000" >&2
    exit 1
fi

RUN_NAME="$1"
shift

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

echo "Resuming run: $RUN_NAME..."
RESUME=1 RUN_NAME="$RUN_NAME" bash "$SCRIPT_DIR/04_run_training.sh" "$@"
