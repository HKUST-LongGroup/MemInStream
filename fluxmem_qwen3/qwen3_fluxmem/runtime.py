"""Qwen3 video processor, checkpoint loading and FluxMem generation."""
import gc
import json
from pathlib import Path
import torch
import transformers
from transformers import AutoProcessor
from .model import Qwen3VLFluxMemForConditionalGeneration, prepare_compressed_inputs
from .video import fetch_video


class Qwen3FluxMemPredictor:
    def __init__(self, args, stats_path):
        if transformers.__version__ != '4.57.1':
            raise RuntimeError('This Qwen3 adapter requires transformers==4.57.1')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU required for checkpoint inference; --dry-run works on CPU')
        self.args, self.stats_path = args, Path(stats_path)
        checkpoint = Path(args.model_path).expanduser().resolve()
        config = json.loads((checkpoint/'config.json').read_text())
        if config.get('model_type') != 'qwen3_vl':
            raise ValueError('Use a dense Qwen3-VL checkpoint, e.g. Qwen3-VL-8B-Instruct; Qwen2.5/MoE are unsupported')
        torch.manual_seed(1234)
        if args.cpu_threads is not None:
            torch.set_num_threads(args.cpu_threads)
        self.model = Qwen3VLFluxMemForConditionalGeneration.from_pretrained(str(checkpoint),
            dtype=torch.bfloat16, device_map='auto', local_files_only=True,
            attn_implementation=args.attn_implementation).eval()
        # Use checkpoint's Qwen3 image/video processors. No Qwen2.5 processor substitutions.
        self.processor = AutoProcessor.from_pretrained(str(checkpoint), local_files_only=True)
        self.diagnostics = {}
        self.model.model.visual.register_forward_pre_hook(self._record_visual, with_kwargs=True)
        print(json.dumps(dict(model_class=type(self.model).__name__, model_type=self.model.config.model_type,
                              python_device=str(self.model.device), transformers=transformers.__version__), ensure_ascii=False), flush=True)

    def _record_visual(self, module, positional, keyword):
        pixels = positional[0] if positional else keyword.get('hidden_states')
        grid = keyword.get('grid_thw')
        self.diagnostics.setdefault('visual_calls', []).append(dict(pixel_shape=list(pixels.shape),
            grid_thw=grid.detach().cpu().tolist() if grid is not None else None))

    def predict(self, plan):
        self.diagnostics = {}
        args = self.args
        element = dict(type='video', video=plan['video_path'], video_start=plan['video_start'],
            video_end=plan['video_end'], anchor_end=plan['anchor_end'], sample_id=plan['input_index'],
            fps=args.fps, min_frames=args.min_frames, max_frames=args.max_frames,
            min_pixels=args.min_pixels, max_pixels=args.max_pixels)
        # Same sampled frame selection as the original FluxMem helpers; Qwen3 uses
        # patch_size 16, merge_size 2 (32 pixel factor) and actual timestamp metadata.
        frames, sample_fps = fetch_video(element, image_factor=32, return_video_sample_fps=True)
        metadata = element['_sample_metadata']
        self.diagnostics.update(sampled_frames=len(frames), sample_fps=sample_fps, video_metadata=metadata)
        messages = [{'role':'user','content':[{'type':'video','video':'sampled-window'},
                                              {'type':'text','text':plan['prompt']}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], videos=[frames], video_metadata=[metadata],
            do_sample_frames=False, do_resize=False, return_tensors='pt', padding=True)
        inputs.pop('video_metadata', None)
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            packed, stats = prepare_compressed_inputs(self.model, inputs, short_frames=args.short_frames,
                medium_frames=args.medium_frames, video_chunk_t=args.video_chunk_t)
            self.diagnostics.update(stats)
            # Original runner uses max_new_tokens=128 and the checkpoint's sampling
            # configuration. Limit expansion to the supported single-sequence path.
            output = self.model.generate(**packed, max_new_tokens=128, use_cache=True, cache_implementation="dynamic",
                                         num_beams=1, num_return_sequences=1,
                                         return_dict_in_generate=False)
        answer = self.processor.batch_decode(output[:, stats['retained_tokens']:],
            skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        if not answer:
            raise ValueError('Model returned an empty answer')
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        with self.stats_path.open('a') as stream:
            stream.write(json.dumps(dict(input_index=plan['input_index'], **stats))+'\n')
        return answer

    def release_after_error(self):
        gc.collect()
        torch.cuda.empty_cache()
