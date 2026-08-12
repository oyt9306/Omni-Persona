#!/usr/bin/env bash
# One full Omni-Persona v2.2 evaluation cycle for a single model:
#   serve -> infer -> judge -> score  (server is always torn down on exit)
#
#   scripts/run_eval.sh <MODEL_PATH_OR_HF_ID> <RUN_NAME> [LIMIT]
#
# LIMIT > 0 evaluates a stratified subset (all 4 scenarios, both answerability
# classes); omit it or pass 0 for the full 750-item benchmark.
#
# Configure via env (defaults target a standard checkout):
#   BENCH       benchmark json/jsonl        ASSET_ROOT  sample_*/concept_* assets
#   GPU  PORT   serving GPU index / port    OUT_ROOT    results directory
#   PYTHON VLLM_BIN GPU_MEM_UTIL            JUDGE_BASE_URL JUDGE_MODEL
#
# Published reference rows are expected approximate values: the LLM judge is
# non-deterministic and shifts Ans/Cal by about +-1 point run to run, while
# 1-FA and TA are keyword-derived and reproduce exactly.
set -euo pipefail

MODEL="${1:?usage: run_eval.sh <MODEL_PATH_OR_HF_ID> <RUN_NAME> [LIMIT]}"
RUN_NAME="${2:?usage: run_eval.sh <MODEL_PATH_OR_HF_ID> <RUN_NAME> [LIMIT]}"
LIMIT="${3:-0}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="${BENCH:-$HERE/data/augmented_context_query_pairs_v2_2.json}"
ASSET_ROOT="${ASSET_ROOT:-$HERE/data/lsd}"
OUT_ROOT="${OUT_ROOT:-$HERE/results}"
GPU="${GPU:-0}"
PORT="${PORT:-8001}"
PYTHON="${PYTHON:-python3}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://localhost:8091/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.4-mini}"
NUM_WORKERS="${NUM_WORKERS:-1}"
JUDGE_WORKERS="${JUDGE_WORKERS:-8}"

OUT="$OUT_ROOT/$RUN_NAME"
mkdir -p "$OUT"
[ "$LIMIT" -gt 0 ] && EXPECT_N="$LIMIT" || EXPECT_N=750

echo "[eval] run=$RUN_NAME model=$MODEL limit=${LIMIT:-full} out=$OUT"

cleanup() { "$HERE/scripts/serve_model.sh" --stop "$GPU" "$PORT" || true; }
trap cleanup EXIT INT TERM

"$HERE/scripts/serve_model.sh" "$MODEL" "$GPU" "$PORT" "$OUT/serve.log" &
SERVE_PID=$!
until curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; do
  sleep 10
  kill -0 "$SERVE_PID" 2>/dev/null || { echo "[eval] server failed — see $OUT/serve.log"; exit 1; }
done

cd "$HERE"
"$PYTHON" -m omni_persona.infer \
  --input "$BENCH" --asset-root "$ASSET_ROOT" \
  --base-url "http://localhost:${PORT}/v1" --model m \
  --output "$OUT/predictions.jsonl" \
  --num-workers "$NUM_WORKERS" --max-tokens 256 --temperature 0.0 \
  --limit "$LIMIT" 2>&1 | tee "$OUT/infer.log"

cleanup
trap - EXIT INT TERM

"$PYTHON" -m omni_persona.judge \
  --predictions "$OUT/predictions.jsonl" \
  --output "$OUT/judge_results.jsonl" \
  --judge-base-url "$JUDGE_BASE_URL" --judge-model "$JUDGE_MODEL" \
  --num-workers "$JUDGE_WORKERS" 2>&1 | tee "$OUT/judge.log"

"$PYTHON" -m omni_persona.score "$OUT/judge_results.jsonl" \
  --name "$RUN_NAME" --expect-n "$EXPECT_N" \
  --json-out "$OUT/scores.json" --csv-out "$OUT/scores.csv" | tee "$OUT/score.log"
