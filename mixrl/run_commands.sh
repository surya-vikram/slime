#!/usr/bin/env bash
# MixRL run: 6 training GPUs (0-5) + judge on GPUs 6,7 (TP=2), EP=1
# 1008 prompts/step x 16 responses = 16128 samples; eval every 10 steps, save weights every 20 steps.
# Usage (from anywhere): bash <slime>/mixrl/run_commands.sh
# Before running: set MODEL_NAME (and BASE_DIR if different) in mixrl/config.env.
set -euo pipefail

RUN_NAME=${RUN_NAME:-mix-16r}
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."        # slime checkout root
source mixrl/config.env

# 1. Judge on GPUs 6,7. With a native vLLM install judge.sh stays in the foreground,
#    so run it in the background and wait until it serves (JUDGE_USE_DOCKER=1 waits by itself).
if [[ "$JUDGE_USE_DOCKER" == 1 ]]; then
    mixrl/judge.sh
else
    nohup mixrl/judge.sh > "$BASE_DIR/judge_${RUN_NAME}.log" 2>&1 &
    echo "judge starting (log: $BASE_DIR/judge_${RUN_NAME}.log)"
    until curl -sf "http://$JUDGE_HOST:$JUDGE_PORT/v1/models" >/dev/null; do sleep 10; done
    echo "judge ready"
fi

# 2. Reward service (must report the judge reachable and no "cannot grade")
mixrl/reward.sh

# 3. Check, then start
mixrl/run.sh tasks         # expect: 16 of 16 tasks enabled; 1008 prompts per step x 16 responses = 16128 samples
mixrl/run.sh preflight "$RUN_NAME"
mixrl/run.sh start "$RUN_NAME"

# Logs:        $BASE_DIR/runs/chimera/mixrl/$RUN_NAME/logs/train.log
# Checkpoints: $BASE_DIR/runs/chimera/mixrl/$RUN_NAME/checkpoints/ (every 20 steps, weights only)
