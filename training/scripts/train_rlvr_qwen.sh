#!/usr/bin/env bash
# Omni-Persona RLVR on Qwen2.5-Omni -- the configuration behind the released
# checkpoints. Two GPUs by default; GRAD_ACCUM is scaled so the effective batch
# stays at 8 regardless of how many you use.
#
#   MODEL=Qwen/Qwen2.5-Omni-7B CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_rlvr_qwen.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

MODEL="${MODEL:-Qwen/Qwen2.5-Omni-3B}"
RUN_NAME="${RUN_NAME:-omni-persona-rlvr-$(basename "$MODEL")}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the directory holding Benchmark_tot/ and the query assets}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://localhost:8091/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.4-mini}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC=$(tr ',' '\n' <<<"$CUDA_VISIBLE_DEVICES" | grep -c .)
GRAD_ACCUM="${GRAD_ACCUM:-$(( 8 / NPROC > 0 ? 8 / NPROC : 1 ))}"

# Qwen2.5-Omni ships a Talker that a text-only objective never trains.
export USE_THINKER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[rlvr] model=$MODEL gpus=$CUDA_VISIBLE_DEVICES grad_accum=$GRAD_ACCUM"
echo "[rlvr] mixture localize=0.30 verify=0.30 text_qa=0.40 | unanswerable ratio 0.6"

torchrun --nproc_per_node="$NPROC" --master_port="${MASTER_PORT:-12732}" \
  -m omni_persona_rlvr.train \
    --model "$MODEL" \
    --rl-json data/rl_pairs.json \
    --lv-jsonl data/localize_verify.jsonl \
    --data-root "$DATA_ROOT" \
    --output-dir "outputs/${RUN_NAME}" \
    --judge-base-url "$JUDGE_BASE_URL" \
    --judge-model "$JUDGE_MODEL" \
    --localize-weight 0.30 --verify-weight 0.30 --textqa-weight 0.40 \
    --unans-ratio 0.6 \
    --num-generations 8 \
    --max-completion-length 512 \
    --temperature 1.1 \
    --beta 0.04 \
    --max-steps "${MAX_STEPS:-100}" \
    --save-steps "${SAVE_STEPS:-50}" \
    --grad-accum "$GRAD_ACCUM" \
    ${DEEPSPEED:+--deepspeed "$DEEPSPEED"}
