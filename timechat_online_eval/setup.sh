#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Run INSIDE a dedicated Python 3.10 conda environment on a Linux CUDA host.
python -c 'import sys; assert sys.version_info[:2] == (3,10), "Use Python 3.10"'
command -v ffmpeg >/dev/null
command -v ffprobe >/dev/null
python -m pip install --upgrade 'pip<26' setuptools wheel packaging ninja
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
# Install the official wheel directly: the sdist's os.rename fails when pip's
# build directory and wheel cache are on different filesystems (EXDEV).
python - <<'PY'
import platform, sys, torch
assert platform.system() == 'Linux' and platform.machine() == 'x86_64'
assert sys.version_info[:2] == (3, 10)
assert torch.__version__.startswith('2.5.1') and torch.version.cuda.startswith('12.')
assert not torch._C._GLIBCXX_USE_CXX11_ABI, 'This wheel requires CXX11 ABI FALSE'
PY
python -m pip install einops==0.8.2
python -m pip install --no-deps 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
python -m pip check
python - <<'PY'
import torch, flash_attn, sys
from pathlib import Path
from predict import verify_upstream
print('Official source:', verify_upstream())
sys.path.insert(0, str(Path('upstream/eval').resolve()))
from qwen2_5_vl.modeling_qwen2_5_vl_DTD import Qwen2_5_VLForConditionalGeneration
assert torch.cuda.is_available(), 'No CUDA GPU visible'
assert torch.cuda.is_bf16_supported(), 'BF16-capable GPU required'
print('GPU:', torch.cuda.get_device_name(), 'torch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('Model:', Qwen2_5_VLForConditionalGeneration.__module__)
PY
