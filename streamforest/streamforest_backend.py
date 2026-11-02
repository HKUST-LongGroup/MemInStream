"""Thin adapter around the pinned, unmodified StreamForest model implementation.

Loading and generation follow lmms_eval/models/streamforest.py. Benchmark text
and sampling are supplied separately; no language/vision/memory weights change.
"""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SYSTEM_PROMPT = 'Answer the question using the supplied video and any provided dialogue history.'


def verify_upstream(root):
    root = Path(root).expanduser().resolve()
    manifest = json.loads((HERE / 'upstream_manifest.json').read_text())
    for relative, expected in manifest['sha256'].items():
        path = root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Official source differs from the pinned snapshot: {path}')
    actual = {p.relative_to(root).as_posix() for p in (root / 'llava').rglob('*.py')}
    if actual - manifest['sha256'].keys():
        raise ValueError('Unexpected Python files in the official llava package')
    return manifest


def validate_config(config):
    if config.get('architectures') != ['LlavaQwenForCausalLM']:
        raise ValueError('Use an official StreamForest Qwen2 checkpoint, not a Qwen2.5/3-VL checkpoint')
    if config.get('mm_projector_type') not in {'tome729_fstw_pemf', 'tome16_mlp'}:
        raise ValueError('Unsupported projector: expected released StreamForest/Pretrain weights')
    if config.get('mm_patch_merge_type') != 'flat':
        raise ValueError('This adapter requires the released flat visual-token layout')


def render_prompt(prompt, system_prompt='aligned'):
    from llava.constants import DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
    if DEFAULT_IMAGE_TOKEN in prompt:
        raise ValueError('Benchmark text contains a reserved <image> marker')
    conv = conv_templates['qwen_2'].copy()
    if system_prompt == 'aligned':
        conv.system = '<|im_start|>system\n' + SYSTEM_PROMPT
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + prompt)
    conv.append_message(conv.roles[1], None)
    stop = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    return conv.get_prompt(), stop


def decode_official(path, start, end, args):
    """Optional official dynamic_fps1 sampler; unlike aligned mode, caps uniformly."""
    from llava.video_utils import VIDEO_READER_FUNCS
    frames, indices, source_fps, duration = VIDEO_READER_FUNCS['decord'](
        video_path=str(path), num_frames=args.max_frames, sample='dynamic_fps1',
        fix_start=None, min_num_frames=4, max_num_frames=args.max_frames,
        client=None, clip=[start, end], local_num_frames=1)
    if not len(frames):
        raise ValueError('Official reader returned no frames')
    timestamps = [float(i) / source_fps for i in indices]
    # Official decord floors start*fps; refuse a frame outside the requested window.
    if min(timestamps) < start - 1e-7 or max(timestamps) >= end:
        raise ValueError('Official reader sampled outside the requested window; use --sampling aligned')
    return frames, dict(decoded_frames=len(frames), input_frames=len(frames), padding_frames=0,
                        source_fps=source_fps, source_frame_indices=[int(i) for i in indices],
                        first_frame_time=timestamps[0], last_frame_time=timestamps[-1],
                        reader_duration=float(duration))


class Predictor:
    def __init__(self, args, device_index=None):
        root = Path(args.streamforest_root).expanduser().resolve()
        verify_upstream(root)
        sys.path.insert(0, str(root))
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('StreamForest requires a CUDA GPU; use --dry-run for CPU validation')
        rank = 0 if device_index is None else device_index
        torch.cuda.set_device(rank)
        torch.manual_seed(args.seed)
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True  # official evaluator setting
        from check_model_imports import load_builder
        load_pretrained_model = load_builder(root).load_pretrained_model
        from llava.model.language_model.llava_qwen import LlavaQwenConfig
        from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
        from llava.constants import IMAGE_TOKEN_INDEX
        import llava
        if not Path(llava.__file__).resolve().is_relative_to(root):
            raise RuntimeError(f'Wrong llava package imported: {llava.__file__}')
        config = LlavaQwenConfig.from_pretrained(args.model_path)
        validate_config(config.to_dict())
        # Official checkpoints contain a private absolute SigLIP path.
        # Remap only that location; keep every projector/PEMF/FSTW setting intact.
        overwrite = {'mm_vision_tower': args.vision_tower} if args.vision_tower else {}
        self.tokenizer, self.model, self.processor, context = load_pretrained_model(
            args.model_path, None, 'StreamForest-Qwen2-7B', multimodal=True,
            device_map=f'cuda:{rank}', attn_implementation=args.attn_implementation,
            overwrite_config=overwrite)
        self.model.eval()
        limits = [args.max_input_tokens, int(context)]
        for key in ('tokenizer_model_max_length', 'max_position_embeddings'):
            value = getattr(self.model.config, key, None)
            if isinstance(value, int) and value > 0:
                limits.append(value)
        self.context_limit = min(limits)
        self.args, self.torch = args, torch
        self.tokenizer_image_token = tokenizer_image_token
        self.stopping_class, self.image_index = KeywordsStoppingCriteria, IMAGE_TOKEN_INDEX
        self.device = torch.device(f'cuda:{rank}')
        self.text_tokens = self.input_tokens = self.visual_tokens = 0
        # Check the full multimodal length BEFORE upstream can silently truncate it.
        self.model.get_model().mm_projector.register_forward_hook(self._check_visual_budget)
        self.info = dict(model_class=type(self.model).__name__,
                         projector=self.model.config.mm_projector_type,
                         vision_encode_type=self.model.config.vision_encode_type,
                         vision_tower=self.model.config.mm_vision_tower,
                         context_limit=self.context_limit, dtype=str(self.model.dtype),
                         generation_config=self.model.generation_config.to_dict())
        print(json.dumps(self.info, ensure_ascii=False), flush=True)

    def _check_visual_budget(self, module, positional, output):
        self.visual_tokens = output.numel() // output.shape[-1]
        self.input_tokens = self.text_tokens - 1 + self.visual_tokens
        if self.input_tokens + self.args.max_new_tokens > self.context_limit:
            raise ValueError(f'{self.input_tokens} multimodal input tokens + '
                             f'{self.args.max_new_tokens} generation tokens exceed {self.context_limit}; '
                             'reduce history/FPS/dialogue length explicitly')

    def generate(self, prompt, frames):
        torch = self.torch
        rendered, stop = render_prompt(prompt, self.args.system_prompt)
        ids = self.tokenizer_image_token(rendered, self.tokenizer, self.image_index,
                                        return_tensors='pt').unsqueeze(0).to(self.device)
        self.text_tokens = ids.shape[1]
        self.input_tokens = self.visual_tokens = 0
        if self.text_tokens + self.args.max_new_tokens > self.context_limit:
            raise ValueError('Text alone exceeds the context budget')
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        mask = ids.ne(pad).to(self.device)
        pixels = self.processor.preprocess(frames, return_tensors='pt')['pixel_values']
        pixels = pixels.to(device=self.device, dtype=torch.float16)
        stopping = self.stopping_class([stop], self.tokenizer, ids)
        with torch.inference_mode():
            output = self.model.generate(
                ids, attention_mask=mask, pad_token_id=pad, images=[pixels],
                modalities=['video'], use_cache=True, do_sample=False, temperature=0,
                top_p=None, num_beams=1, max_new_tokens=self.args.max_new_tokens,
                stopping_criteria=[stopping])
        # Upstream generate passes inputs_embeds to HF: output contains generated
        # tokens only. Slicing by input_ids length would erase the answer.
        answer = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
        if not answer:
            raise ValueError('Model returned an empty answer')
        return answer, self.input_tokens
