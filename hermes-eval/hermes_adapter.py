"""Thin adapter around the pinned, unmodified HERMES inference modules."""
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent


def verify_upstream(root):
    root = Path(root).resolve()
    manifest = json.loads((HERE / 'upstream_manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'HERMES source differs from pinned revision: {path}')
    return manifest


def reset_memory(model):
    # Official clear_cache omits activity/position/hook state. Independent samples
    # must not inherit it, including after an exception halfway through a chunk.
    model._layer_position_ids.clear()
    model.clear_cache()
    model._position_ids_cache = [None] * model.num_layers
    if hasattr(model, 'token_activity_cache'):
        model.token_activity_cache = [None] * model.num_layers
    model.last_encoded_frames = 0
    model.total_processed_frames = 0
    if hasattr(model, '_set_rotary_required_len'):
        model._set_rotary_required_len(0)


def run_stream(model, frames, prompt, args, tensor_factory=lambda x: x, max_new_tokens=None):
    reset_memory(model)
    model.encode_init_prompt()
    chunks = []
    for start in range(0, len(frames), args.chunk_frames):
        chunk = tensor_factory(frames[start:start + args.chunk_frames])
        model.encode_video_chunk(chunk)
        model.predict_and_compress()
        lengths = model._get_cache_seq_len_per_layer()
        chunks.append(dict(start=start, frames=len(chunk), min_kv=min(lengths), max_kv=max(lengths)))
    # mc=False prevents the official single-choice-only 'Best option: (' prefix.
    wrapped = model.get_prompt(prompt, mc=False)
    answer = model.question_answering(
        {'question': prompt, 'formatted_question': prompt, 'prompt': wrapped},
        max_new_tokens=max_new_tokens if max_new_tokens is not None else (args.max_new_tokens or 256), temperature=0,
        repetition_penalty=args.repetition_penalty,
    ).strip()
    if not answer:
        raise ValueError('HERMES returned an empty answer')
    return answer, {'chunks': chunks, 'wrapped_prompt': wrapped,
                    'max_new_tokens': max_new_tokens if max_new_tokens is not None else (args.max_new_tokens or 256),
                    'temperature': 0, 'repetition_penalty': args.repetition_penalty}


class HermesPredictor:
    def __init__(self, args, device_index=0):
        # Each official loader uses device_map=auto. Restrict it to this replica's
        # GPU BEFORE torch/transformers import; internal HERMES tensors assume one device.
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        devices = visible.split(',') if visible is not None else [str(i) for i in range(args.num_gpus)]
        if device_index >= len(devices) or not devices[device_index].strip():
            raise ValueError(f'No visible CUDA device for worker {device_index}')
        os.environ['CUDA_VISIBLE_DEVICES'] = devices[device_index].strip()
        import torch
        import transformers
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU required; --dry-run works without GPU/model weights')
        torch.cuda.set_device(0)
        torch.set_num_threads(args.cpu_threads)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.model == 'qwen25' and transformers.__version__ != '4.57.1':
            raise RuntimeError('Qwen HERMES requires the official transformers==4.57.1 environment')
        if args.model == 'llavaov':
            direct = importlib.metadata.distribution('transformers').read_text('direct_url.json')
            commit = json.loads(direct or '{}').get('vcs_info', {}).get('commit_id')
            if commit != '66bc4def9505fa7c7fe4aa7a248c34a026bb552b':
                raise RuntimeError('LLaVA HERMES requires its pinned transformers git revision; see requirements-llava.txt')
        config = Path(args.model_path).expanduser() / 'config.json'
        if not config.is_file():
            raise FileNotFoundError(f'Use a local Hugging Face checkpoint directory: {config.parent}')
        expected = {'qwen25': 'qwen2_5_vl', 'llavaov': 'llava_onevision'}[args.model]
        if json.loads(config.read_text()).get('model_type') != expected:
            raise ValueError(f'--model {args.model} needs model_type={expected}')
        root = Path(args.hermes_root).resolve()
        verify_upstream(root)
        sys.path.insert(0, str(root))
        name = 'qwenvl_hermes' if args.model == 'qwen25' else 'llavaov_hermes'
        module = importlib.import_module(f'inference.{name}')
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise RuntimeError('Another inference package shadows --hermes-root')
        self.model, self.processor = module.load_model(
            model_path=str(config.parent), kv_size=args.kv_size, streaming=True,
            device='cuda', sample_fps=args.fps,
        )
        self.args, self.torch = args, torch
        self.info = {'torch': torch.__version__, 'transformers': transformers.__version__,
                     'model_class': type(self.model).__name__, 'dtype': str(self.model.dtype),
                     'decoder_attention': self.model.language_model.config._attn_implementation,
                     'system_prompt_policy': args.system_prompt,
                     'visual_preprocessing': args.visual_preprocessing}

    def generate(self, prompt, frames, max_new_tokens=None):
        with self.torch.inference_mode():
            answer, diagnostics = run_stream(self.model, frames, prompt, self.args, self.torch.from_numpy, max_new_tokens=max_new_tokens)
        diagnostics.update(self.info)
        return answer, diagnostics
