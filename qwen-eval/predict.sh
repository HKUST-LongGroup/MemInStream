#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Set the dataset and checkpoint paths, then run: bash predict.sh
INPUT="${INPUT:-YOUR_DATASET/annotations.json}"
MODEL_PATH="${MODEL_PATH:-YOUR_MODEL/Qwen2.5-VL-7B-Instruct}"
OUTPUT="${OUTPUT:-results/predictions.jsonl}"

python -u predict.py \
  --input "$INPUT" \
  --model-path "$MODEL_PATH" \
  --output "$OUTPUT" \
  --model "${MODEL:-qwen25}" --history-seconds 600 --fps 1 --num-gpus "${NUM_GPUS:-1}" "$@"
