"""Model-loading helpers. Imports stay lazy so --dry-run needs only Python."""
import json
from pathlib import Path


SYSTEM_MESSAGE = 'Answer the question using the supplied video and any provided dialogue history.'


def load_settings(args, device_index, expected_type):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required; --dry-run works without torch/CUDA')
    if device_index is not None:
        torch.cuda.set_device(device_index)
    path = Path(args.model_path).expanduser().resolve()
    if not (path / 'config.json').is_file():
        raise FileNotFoundError(f'Local checkpoint missing: {path}; run download_models.py first')
    config = json.loads((path / 'config.json').read_text())
    if config.get('model_type') != expected_type:
        raise ValueError(f'{path}: expected model_type={expected_type}, got {config.get("model_type")}')
    dtype_name = args.dtype
    if dtype_name == 'auto':
        dtype_name = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
    if dtype_name == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise ValueError('This GPU does not support bfloat16; use --dtype float16')
    kwargs = dict(local_files_only=True, torch_dtype=getattr(torch, dtype_name),
                  low_cpu_mem_usage=True,
                  device_map=args.device_map if device_index is None else {'': device_index})
    return torch, path, config, kwargs


def check_budget(input_tokens, args, context_limit=None):
    budget = args.max_input_tokens
    if context_limit:
        budget = min(budget, context_limit)
    if input_tokens + args.max_new_tokens > budget:
        raise ValueError(f'{input_tokens} input + {args.max_new_tokens} output tokens exceed '
                         f'budget {budget}; explicitly lower --fps / --history-seconds or '
                         'raise --max-input-tokens within the model context limit')


def require_answer(answer):
    answer = answer.strip()
    if not answer:
        raise ValueError('Model returned an empty answer')
    return answer
