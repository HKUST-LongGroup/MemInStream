"""Causal video adapter around the unmodified official Flash-VStream implementation."""
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
from official_imports import load_official


def sample_indices(timestamps, start, end, fps, max_frames, torch):
    """Official extract_frames linspace, then official fetch_video frame selection.

    Restrict source frame presentation times BEFORE either sampling stage. Selecting
    the second-stage indices before decoding is equivalent to writing all candidate
    JPEGs and subsequently opening only the frames chosen by fetch_video.
    """
    import numpy as np
    pts = np.asarray(timestamps)[:, 0]
    valid = np.flatnonzero((pts >= start) & (pts < end))
    if not len(valid):
        raise ValueError(f'No causal source frames in [{start}, {end})')
    count = min(max(round((end - start) * fps), 4), 14400)
    candidates = torch.linspace(0, len(valid) - 1, count).round().long().tolist()
    count_out = min(count, max_frames)
    selected = torch.linspace(0, count - 1, count_out).round().long().tolist()
    indices = [int(valid[candidates[i]]) for i in selected]
    return indices, count


def decode_window(path, start, end, args):
    import numpy as np
    import torch
    from PIL import Image
    from decord import VideoReader, cpu
    vr = VideoReader(str(path), ctx=cpu(0), num_threads=args.decoder_threads)
    if len(vr) == 0:
        raise ValueError('Empty video')
    timestamps = np.asarray(vr.get_frame_timestamp(list(range(len(vr)))))
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps[:, 0]) < 0):
        raise ValueError('Video has invalid or non-monotonic presentation timestamps')
    duration = float(timestamps[-1, 1])
    end = min(end, duration)
    if end <= start:
        raise ValueError(f'Empty video window [{start}, {end})')
    indices, candidate_count = sample_indices(timestamps, start, end, args.extract_fps, args.max_frames, torch)
    frames = vr.get_batch(indices).asnumpy()
    images = []
    for frame in frames:
        # Match the official extraction script's JPEG quality=100 intermediate.
        buffer = BytesIO()
        Image.fromarray(frame).save(buffer, format='JPEG', quality=100)
        buffer.seek(0)
        with Image.open(buffer) as image:
            images.append(image.convert('RGB'))
    return images, end, dict(
        source_duration=duration, extracted_candidate_frames=candidate_count,
        selected_frames=len(images), source_frame_indices=indices,
        source_frame_times=[float(timestamps[i, 0]) for i in indices],
        padding_frames=len(images) % 2,
    )


def build_messages(images, prompt, args):
    messages = []
    if args.chat_system == 'qwen':
        messages.append({'role': 'system', 'content':
                         'Answer the question using the supplied video and any provided dialogue history.'})
    messages.append({'role': 'user', 'content': [
        {'type': 'video', 'video': images, 'max_frames': args.max_frames, 'max_pixels': args.max_pixels},
        {'type': 'text', 'text': prompt},
    ]})
    return messages


class Predictor:
    def __init__(self, args):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU required; use --dry-run for CPU-only validation')
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('The official BF16 path requires a BF16-capable CUDA GPU')
        models, vision = load_official(args.upstream_root)
        FlashVStreamQwen2VLModel = models.FlashVStreamQwen2VLModel
        FlashVStreamQwen2VLConfig = models.FlashVStreamQwen2VLConfig
        FlashVStreamQwen2VLProcessor = models.FlashVStreamQwen2VLProcessor
        DEFAULT_FLASH_MEMORY_CONFIG = models.DEFAULT_FLASH_MEMORY_CONFIG
        process_vision_info = vision.process_vision_info
        raw_config = json.loads((Path(args.model_path) / 'config.json').read_text())
        if raw_config.get('model_type') != 'flash_vstream_qwen2_vl':
            raise ValueError('Use zhang9302002/Flash-VStream-Qwen-7b, not a base Qwen checkpoint')
        config = FlashVStreamQwen2VLConfig.from_pretrained(args.model_path, local_files_only=True)
        if getattr(config.vision_config, 'flash_memory_config', None) is None:
            config.vision_config.flash_memory_config = deepcopy(DEFAULT_FLASH_MEMORY_CONFIG)
        self.model = FlashVStreamQwen2VLModel.from_pretrained(
            args.model_path, config=config, device_map='cuda', trust_remote_code=True,
            local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2',
        ).eval()
        self.processor = FlashVStreamQwen2VLProcessor.from_pretrained(
            args.processor_path, local_files_only=True)
        # As in inference_mcq_vqa.py: fill missing defaults after model loading.
        self.memory = config.vision_config.flash_memory_config
        for key, value in DEFAULT_FLASH_MEMORY_CONFIG.items():
            self.memory.setdefault(key, value)
        self.args, self.torch, self.process_vision_info = args, torch, process_vision_info
        print(json.dumps({'model': type(self.model).__name__, 'flash_memory_config': self.memory,
                          'generation_config': self.model.generation_config.to_dict()}, default=str), flush=True)

    def generate(self, prompt, images):
        messages = build_messages(images, prompt, self.args)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Deliberately no official "Best option: (" suffix: benchmark supports multiselect.
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors='pt', flash_memory_config=self.memory)
        names = ('input_ids', 'attention_mask', 'pixel_values_videos', 'video_grid_thw', 'visual_position_ids')
        generation_inputs = {key: inputs[key].cuda() for key in names}
        with self.torch.inference_mode():
            generated = self.model.generate(**generation_inputs, max_new_tokens=128, top_k=1, do_sample=False)
        trimmed = [out[len(ids):] for ids, out in zip(inputs.input_ids, generated)]
        answer = self.processor.batch_decode(trimmed, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)[0].strip()
        if not answer:
            raise ValueError('Model returned an empty response')
        return answer, dict(input_tokens=int(inputs.input_ids.shape[1]),
                            video_grid_thw=inputs.video_grid_thw.tolist(),
                            pixel_values_videos_shape=list(inputs.pixel_values_videos.shape),
                            flash_memory_config=dict(self.memory))
