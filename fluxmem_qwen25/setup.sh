#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Activate a fresh Python 3.11 environment first.
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip check
# For the default attention backend, also install a matching FlashAttention 2 wheel.
