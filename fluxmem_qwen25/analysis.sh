#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

INPUT="${INPUT:-YOUR_DATASET/annotations.json}"
PREDICTIONS="${PREDICTIONS:-results/fluxmem.jsonl}"
python analyze_evidence.py --input "$INPUT" --predictions "$PREDICTIONS" \
  --output-dir results/evidence_analysis "$@"
