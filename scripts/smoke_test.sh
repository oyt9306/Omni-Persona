#!/usr/bin/env bash
# Fast install check: 40 stratified items (all 4 scenarios, answerable +
# unanswerable) through serve -> infer -> judge -> score. ~10 min on one GPU.
#
#   scripts/smoke_test.sh [MODEL_PATH_OR_HF_ID] [RUN_NAME] [LIMIT]
#
# Honours the same env vars as run_eval.sh (BENCH, ASSET_ROOT, GPU, PORT, ...).
# The printed numbers are a subset, NOT the paper's main table — a successful
# smoke run means the plumbing works. Use run_eval.sh for the real 750-item row.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-Qwen/Qwen2.5-Omni-3B}"
RUN_NAME="${2:-smoke}"
LIMIT="${3:-40}"

JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://localhost:8091/v1}"
curl -sf "${JUDGE_BASE_URL%/v1}/health" > /dev/null \
  || { echo "[smoke] judge endpoint not reachable at $JUDGE_BASE_URL"; exit 1; }

"$HERE/scripts/run_eval.sh" "$MODEL" "$RUN_NAME" "$LIMIT"

echo
echo "[smoke] OK — install works. Full benchmark:  scripts/run_eval.sh <MODEL> <NAME>"
