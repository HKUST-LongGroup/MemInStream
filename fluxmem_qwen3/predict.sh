#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Set the dataset and checkpoint paths, then run: bash predict.sh
INPUT="${INPUT:-YOUR_DATASET/annotations.json}"
MODEL_PATH="${MODEL_PATH:-YOUR_MODEL/Qwen3-VL-8B-Instruct}"
OUTPUT="${OUTPUT:-results/fluxmem.jsonl}"

python -u predict.py \
  --input "$INPUT" \
  --model-path "$MODEL_PATH" \
  --output "$OUTPUT" \
  --num-gpus "${NUM_GPUS:-1}" --time-window 256 --fps 1 \
  --video-chunk-t 4 --cpu-threads 4 --attn-implementation sdpa "$@"
