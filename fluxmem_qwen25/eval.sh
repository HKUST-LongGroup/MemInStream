#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

INPUT="${INPUT:-YOUR_DATASET/annotations.json}"
PREDICTIONS="${PREDICTIONS:-results/fluxmem.jsonl}"
python evaluate.py --input "$INPUT" --predictions "$PREDICTIONS" \
  --output-dir results/scores "$@"
