#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
# Run inside a NEW activated Python 3.11 environment.
# Requires an NVIDIA driver compatible with the CUDA 12.4 wheel build.
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch, transformers, av; from qwen3_fluxmem.model import Qwen3VLFluxMemForConditionalGeneration; print('torch', torch.__version__, 'transformers', transformers.__version__, 'CUDA', torch.cuda.is_available())"
