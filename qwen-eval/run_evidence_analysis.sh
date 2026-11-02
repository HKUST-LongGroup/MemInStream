#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Install optional plotting dependencies: python -m pip install -r requirements-analysis.txt
PREDICTIONS="${PREDICTIONS:-results/predictions.jsonl}"
python analyze_evidence.py --input "$PREDICTIONS" --output-dir results/evidence_analysis "$@"
