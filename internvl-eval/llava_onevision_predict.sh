#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Set the dataset and checkpoint paths, then run: bash llava_onevision_predict.sh
INPUT="${INPUT:-YOUR_DATASET/annotations.json}"
MODEL_PATH="${MODEL_PATH:-YOUR_MODEL/llava-onevision-qwen2-7b-ov-hf}"
OUTPUT="${OUTPUT:-results/llava_onevision.jsonl}"

python -u predict.py \
  --input "$INPUT" \
  --model-path "$MODEL_PATH" \
  --output "$OUTPUT" \
  --model llava_onevision --history-seconds 60 --fps 1 --num-gpus "${NUM_GPUS:-1}" "$@"
