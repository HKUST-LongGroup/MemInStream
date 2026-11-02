#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Run after: conda create -n flash-vstream-eval python=3.10.14 -y
#            conda activate flash-vstream-eval
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
# Install the prebuilt wheel directly. The sdist's download-and-rename step can
# fail with EXDEV on clusters where temporary files and pip cache use different mounts.
python - <<'PY'
import platform
import sys
import torch
if not (sys.version_info[:2] == (3, 10) and platform.system() == 'Linux'
        and platform.machine() == 'x86_64' and torch.__version__.split('+')[0] == '2.6.0'
        and str(torch.version.cuda).startswith('12.') and not torch._C._GLIBCXX_USE_CXX11_ABI):
    raise SystemExit('This wheel requires Linux x86_64, Python 3.10, Torch 2.6, CUDA 12 and CXX11 ABI=False.')
PY
python -m pip install --no-cache-dir \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
python -m pip check
python check_environment.py
