#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Set the dataset and checkpoint paths, then run: bash predict.sh
INPUT="${INPUT:-YOUR_DATASET/annotations.json}"
MODEL_PATH="${MODEL_PATH:-YOUR_MODEL/InternVL3_5-8B}"
OUTPUT="${OUTPUT:-results/internvl35.jsonl}"

python -u predict.py \
  --input "$INPUT" \
  --model-path "$MODEL_PATH" \
  --output "$OUTPUT" \
  --model internvl35 --history-seconds 60 --fps 1 --num-gpus "${NUM_GPUS:-1}" "$@"
