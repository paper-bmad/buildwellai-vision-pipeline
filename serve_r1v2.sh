#!/usr/bin/env bash
# serve_r1v2.sh — Start Skywork R1V2-38B multimodal via vLLM for BuildwellAI vision pipeline
#
# Skywork R1V2-38B is a reasoning-focused VLM that excels at documents, charts, and STEM
# diagrams — making it a strong fit for architectural drawing analysis (vs. general photo
# QA, where Gemma 4 / Qwen-VL remain stronger).
#
# Benchmarks: MMMU 73.6%, OlympiadBench 62.6%, AIME24 78.9%, RealWorldQA 68.9%
# License: MIT
#
# Run on HPC (pvc9) with GPU access:
#   ./serve_r1v2.sh             # AWQ variant — single 30GB GPU (recommended)
#   ./serve_r1v2.sh fp16        # FP16 variant — 2× A100

set -eo pipefail

VARIANT="${1:-awq}"
PORT="${PORT:-8000}"

case "$VARIANT" in
  fp16)   MODEL="Skywork/Skywork-R1V2-38B";     TENSOR_PARALLEL=2; MAX_MODEL_LEN=8192  ;;
  awq|*)  MODEL="Skywork/Skywork-R1V2-38B-AWQ"; TENSOR_PARALLEL=1; MAX_MODEL_LEN=8192  ;;
esac

echo "Starting Skywork R1V2-38B ($VARIANT) on port $PORT..."
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
  2>&1 | tee "r1v2-${VARIANT}-serve.log"
