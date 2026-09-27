#!/usr/bin/env bash
# ==============================================================================
# 03_run_preflight_dryrun.sh - Zero-FLOP Preflight & Manifest Verification
# ==============================================================================
set -Eeuo pipefail

BASE_DIR="/nvme_zone3/home/ekamai1/chimera/mixrl"

echo "=== Running Zero-FLOP Dry-Run Preflight in Slime Container ==="
echo "Verifying model shards, YaRN 32K context, 512x8 quotas, patches, and scorer connection..."

mkdir -p "$BASE_DIR/runs"

docker run --rm \
  --ipc=host \
  --net=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$BASE_DIR/models/chimera-muon-nemotron-105k-yarn32k-iter3478:/data/models/chimera-muon-nemotron-105k-yarn32k-iter3478:ro" \
  -v "$BASE_DIR/datasets/chimera-eval-data:/data/datasets/chimera-eval-data:ro" \
  -v "$BASE_DIR/repos/transformers:/workspace/transformers:ro" \
  -v "$BASE_DIR/repos/slime:/workspace/slime" \
  -v "$BASE_DIR/runs:/data/runs" \
  -w /workspace/slime \
  -e DRY_RUN=1 \
  -e DATA_ROOT=/data \
  -e HF_CHECKPOINT=/data/models/chimera-muon-nemotron-105k-yarn32k-iter3478/hf \
  -e MCORE_CHECKPOINT=/data/models/chimera-muon-nemotron-105k-yarn32k-iter3478/mcore \
  -e MIXRL_DATA_DIR=/data/datasets/chimera-eval-data \
  -e MIXRL_CODE_AUDIT_DIR=/data/datasets/chimera-eval-data/audits/apps \
  -e CHIMERA_TRANSFORMERS_ROOT=/workspace/transformers \
  -e MIXRL_SCORER_URL=http://127.0.0.1:18020 \
  -e RUNS_ROOT=/data/runs \
  suryavikram6/slime:pinned \
  bash run_mixrl.sh
