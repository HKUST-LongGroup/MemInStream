#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Activate a fresh Python 3.10 environment first (CUDA 12.6 wheel build).
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip check
