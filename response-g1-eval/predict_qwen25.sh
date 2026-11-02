#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Set the dataset and checkpoint paths, then run: bash predict_qwen25.sh
INPUT="${INPUT:-YOUR_DATASET/annotations.json}"
MODEL_PATH="${MODEL_PATH:-YOUR_MODEL/Qwen2.5-VL-7B-Instruct}"
OUTPUT="${OUTPUT:-results/response_g1_qwen25.jsonl}"

python -u predict.py \
  --input "$INPUT" \
  --model-path "$MODEL_PATH" \
  --output "$OUTPUT" \
  --model qwen25 --profile ovobench --fps 1 --vision-batch-size 8 "$@"
