#!/usr/bin/env python3
"""LLaVA-OneVision Qwen2-7B OV (Hugging Face format) streaming prediction."""
from model_common import SYSTEM_MESSAGE, check_budget, load_settings, require_answer


class LlavaOnevisionPredictor:
    def __init__(self, args, device_index=None):
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
        torch, path, config, kwargs = load_settings(args, device_index, 'llava_onevision')
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            str(path), attn_implementation=args.attn_implementation, **kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(str(path), local_files_only=True, use_fast=False)
        self.context_limit = self.model.config.text_config.max_position_embeddings
        self.args, self.torch, self.dtype = args, torch, kwargs['torch_dtype']

    def generate(self, prompt, frames):
        if any(token in prompt for token in ('<image>', '<video>')):
            raise ValueError('Prompt contains reserved LLaVA visual tokens')
        messages = [
            {'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM_MESSAGE}]},
            {'role': 'user', 'content': [
                {'type': 'video'},
                {'type': 'text', 'text': f'The video frames are in chronological order, '
                 f'sampled at {self.args.fps:g} FPS.\n' + prompt},
            ]},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Already decoded and sampled by the shared ffmpeg pipeline. Keep the
        # official 384px normalization/resize, but disable temporal resampling.
        # In transformers 4.52.4 do_sample_frames is a video-processor
        # attribute, not an accepted per-call kwarg for this processor.
        self.processor.video_processor.do_sample_frames = False
        inputs = self.processor(text=[text], videos=[frames],
                                return_tensors='pt', truncation=False)
        tokens = inputs['input_ids'].shape[1]
        check_budget(tokens, self.args, self.context_limit)
        processed_frames = inputs['pixel_values_videos'].shape[1]
        if processed_frames != len(frames):
            raise ValueError(f'Processor changed frame count: {len(frames)} -> {processed_frames}')
        device = self.model.get_input_embeddings().weight.device
        # BatchFeature.to(dtype=...) casts floating tensors only; token IDs stay integers.
        inputs = inputs.to(device=device, dtype=self.dtype)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, num_beams=1,
                                         max_new_tokens=self.args.max_new_tokens, use_cache=True)
        answer = self.processor.batch_decode(output[:, tokens:], skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)[0]
        return require_answer(answer), tokens


if __name__ == '__main__':
    from predict_streaming import main
    raise SystemExit(main(model='llava_onevision'))
