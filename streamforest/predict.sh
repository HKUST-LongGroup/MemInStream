#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Set the dataset and checkpoint paths, then run: bash predict.sh
INPUT="${INPUT:-YOUR_DATASET/annotations.json}"
MODEL_PATH="${MODEL_PATH:-YOUR_MODEL/StreamForest-Qwen2-7B}"
OUTPUT="${OUTPUT:-results/streamforest.jsonl}"
VISION_TOWER="${VISION_TOWER:-YOUR_MODEL/siglip-so400m-patch14-384}"
python -u predict.py \
  --input "$INPUT" \
  --model-path "$MODEL_PATH" \
  --output "$OUTPUT" \
  --vision-tower "$VISION_TOWER" --history-seconds 600 --fps 1 --num-gpus "${NUM_GPUS:-1}" "$@"
