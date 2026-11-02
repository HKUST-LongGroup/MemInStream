#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Set the dataset and checkpoint paths, then run: bash predict.sh
INPUT="${INPUT:-YOUR_DATASET/annotations.json}"
MODEL_PATH="${MODEL_PATH:-YOUR_MODEL/Flash-VStream-Qwen-7b}"
OUTPUT="${OUTPUT:-results/flash_vstream.jsonl}"
PROCESSOR_PATH="${PROCESSOR_PATH:-YOUR_MODEL/Qwen2-VL-7B-Instruct}"
python -u predict.py \
  --input "$INPUT" \
  --model-path "$MODEL_PATH" \
  --output "$OUTPUT" \
  --processor-path "$PROCESSOR_PATH" --history-seconds 256 --num-gpus "${NUM_GPUS:-1}" "$@"
