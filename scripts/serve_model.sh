#!/usr/bin/env bash
# Serve a Qwen2.5-Omni checkpoint with vLLM on a single GPU and block until killed.
#
#   scripts/serve_model.sh <MODEL_PATH_OR_HF_ID> [GPU] [PORT] [LOG]
#   scripts/serve_model.sh --stop [GPU] [PORT]      # teardown (server + GPU memory)
#
# The model is registered under the name "m": pass `--model m` to the client.
# Ctrl-C tears the server down. scripts/run_eval.sh drives this automatically.
#
# Env: VLLM_BIN      path to the vllm executable
#      READY_TIMEOUT seconds to wait for /health (default 1800)
#      GPU_MEM_UTIL  fraction of GPU memory vLLM may claim (default 0.85).
#                    Lower it when sharing the GPU; it does not change results,
#                    only KV-cache size and throughput.
set -u

VLLM="${VLLM_BIN:-vllm}"

stop_server() {
  local gpu="$1" port="$2" uuid p
  fuser -k "${port}"/tcp 2>/dev/null
  sleep 2
  # vLLM spawns a detached EngineCore worker: clear every process of ours on this GPU.
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$gpu" 2>/dev/null) || return 0
  for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader \
             | grep "$uuid" | cut -d, -f1); do
    [ "$(ps -o user= -p "$p" 2>/dev/null)" = "$(whoami)" ] && kill -9 "$p" 2>/dev/null
  done
  sleep 3
  echo "[serve] stopped (gpu=$gpu port=$port)"
}

if [ "${1:-}" = "--stop" ]; then
  stop_server "${2:-0}" "${3:-8001}"
  exit 0
fi

MODEL="${1:?usage: serve_model.sh <MODEL_PATH_OR_HF_ID> [GPU] [PORT] [LOG]}"
GPU="${2:-0}"
PORT="${3:-8001}"
LOG="${4:-serve.log}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"

mkdir -p "$(dirname "$LOG")"
stop_server "$GPU" "$PORT"

echo "[serve] model=$MODEL gpu=$GPU port=$PORT log=$LOG"
env CUDA_VISIBLE_DEVICES="$GPU" USE_THINKER=1 VLLM_ENGINE_READY_TIMEOUT_S=1800 \
  "$VLLM" serve "$MODEL" \
    --host 0.0.0.0 --port "$PORT" \
    --served-model-name m \
    --trust-remote-code \
    --dtype bfloat16 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len 17408 \
    --limit-mm-per-prompt '{"image":5,"audio":5}' \
    --mm-processor-cache-gb 0 \
  > "$LOG" 2>&1 &
SRV=$!
trap 'stop_server "$GPU" "$PORT"' EXIT INT TERM

el=0
until curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; do
  sleep 10; el=$((el + 10))
  kill -0 "$SRV" 2>/dev/null || { echo "[serve] died — see $LOG"; tail -20 "$LOG"; exit 1; }
  [ "$el" -ge "$READY_TIMEOUT" ] && { echo "[serve] timeout after ${el}s — see $LOG"; exit 1; }
done
echo "[serve] healthy after ${el}s → http://localhost:${PORT}/v1"

wait "$SRV"
