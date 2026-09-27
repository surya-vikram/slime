#!/usr/bin/env bash
# Run INSIDE the pinned Slime container; no network or remote GPU required.
set -Eeuo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
CHIMERA_TRANSFORMERS_ROOT=${CHIMERA_TRANSFORMERS_ROOT:-$(dirname "$REPO_ROOT")/transformers}
LOCAL_VALIDATION_DIR=${LOCAL_VALIDATION_DIR:-${TMPDIR:-/tmp}/chimera-local-$(date +%Y%m%d-%H%M%S)}
LOCAL_GPU_CHECKS=${LOCAL_GPU_CHECKS:-1}
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$LOCAL_VALIDATION_DIR"
cd "$REPO_ROOT"
bash -n examples/chimera/train.sh scripts/models/chimera.sh
python3 -m pytest -q tests/test_chimera_integration.py tests/test_chimera_context.py \
    tests/test_chimera_geometry.py tests/test_chimera_mixrl*.py \
    2>&1 | tee "$LOCAL_VALIDATION_DIR/unit-tests.log"
if [[ "$LOCAL_GPU_CHECKS" == 1 ]]; then
    python3 tests/local_chimera_forward.py --transformers-root "$CHIMERA_TRANSFORMERS_ROOT" \
        --output "$LOCAL_VALIDATION_DIR/hf-forward.json" 2>&1 | tee "$LOCAL_VALIDATION_DIR/hf-forward.log"
    python3 tests/local_chimera_router.py --output "$LOCAL_VALIDATION_DIR/native-router.json" \
        2>&1 | tee "$LOCAL_VALIDATION_DIR/native-router.log"
    python3 tests/local_chimera_capture.py --output "$LOCAL_VALIDATION_DIR/sglang-capture.json" \
        2>&1 | tee "$LOCAL_VALIDATION_DIR/sglang-capture.log"
elif [[ "$LOCAL_GPU_CHECKS" != 0 ]]; then
    echo 'LOCAL_GPU_CHECKS must be 0 or 1' >&2; exit 1
fi
echo "Component validation complete: $LOCAL_VALIDATION_DIR"
echo 'Full-model SGLang/MCore parity, distributed optimizer and H200 capacity are separate gates.'
