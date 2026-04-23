#!/usr/bin/env bash
# serve_gemma4.sh — Start Gemma 4 multimodal via vLLM for BuildwellAI vision pipeline
#
# Run on HPC (pvc9) with GPU access:
#   ./serve_gemma4.sh            # 27B model (recommended for accuracy)
#   ./serve_gemma4.sh e4b        # E4B 4.5B model (fast, lower VRAM)
#   ./serve_gemma4.sh 12b        # 12B model (balanced)

set -eo pipefail

VARIANT="${1:-27b}"
PORT="${PORT:-8000}"

case "$VARIANT" in
  e4b)    MODEL="google/gemma-4-e4b-it";   TENSOR_PARALLEL=1; MAX_MODEL_LEN=8192  ;;
  12b)    MODEL="google/gemma-4-12b-it";   TENSOR_PARALLEL=1; MAX_MODEL_LEN=8192  ;;
  27b|*)  MODEL="google/gemma-4-27b-it";   TENSOR_PARALLEL=2; MAX_MODEL_LEN=8192  ;;
esac

echo "Starting Gemma 4 ($VARIANT) on port $PORT..."
echo "Model: $MODEL, tensor_parallel=$TENSOR_PARALLEL"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --dtype bfloat16 \
  --tensor-parallel-size "$TENSOR_PARALLEL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --port "$PORT" \
  --trust-remote-code \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.90 \
  2>&1 | tee "gemma4-${VARIANT}-serve.log"
