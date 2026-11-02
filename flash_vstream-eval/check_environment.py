#!/usr/bin/env python3
"""Check dependency imports, official custom classes and CUDA before a smoke run."""
import json
import argparse
from pathlib import Path
from official_imports import load_official, validate_source

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--upstream-root', default=str(Path(__file__).resolve().parent / 'upstream'))
args = parser.parse_args()
source = validate_source(args.upstream_root)
print(f'Official source: {source}', flush=True)

import torch
import torchvision
import transformers
import flash_attn
import decord
if transformers.__version__ != '4.45.0':
    raise RuntimeError('Use transformers==4.45.0; the official custom model imports its internal APIs')
models, vision = load_official(args.upstream_root)
print(f'Official model package: {models.__file__}', flush=True)
print(f'Official vision package: {vision.__file__}', flush=True)
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise RuntimeError('CUDA with BF16 support is required')
print(json.dumps(dict(torch=torch.__version__, torchvision=torchvision.__version__,
                      transformers=transformers.__version__, flash_attn=flash_attn.__version__,
                      decord=decord.__version__, cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(), gpu_count=torch.cuda.device_count(),
                      cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI), indent=2))
