#!/usr/bin/env python3
"""InternVL3.5-8B streaming benchmark prediction, using the official custom model."""
from copy import deepcopy

from model_common import SYSTEM_MESSAGE, check_budget, load_settings, require_answer


def prepare_query(model, prompt, frame_count, fps):
    # Same conversation and image-token expansion as official model.chat().
    # One 448x448 tile per frame, matching the model card's video max_num=1.
    if any(token in prompt for token in ('<image>', '<img>', '</img>', '<IMG_CONTEXT>')):
        raise ValueError('Prompt contains reserved InternVL image tokens')
    prefix = ''.join(f'Frame{i + 1}: <image>\n' for i in range(frame_count))
    question = (prefix + f'The video frames are in chronological order, sampled at {fps:g} FPS.\n'
                + prompt)
    template = deepcopy(model.conv_template)
    template.messages = []
    template.system_message = SYSTEM_MESSAGE
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    image_tokens = '<img>' + '<IMG_CONTEXT>' * model.num_image_token + '</img>'
    query = query.replace('<image>', image_tokens)
    return query, template.sep.strip()


class InternVL35Predictor:
    def __init__(self, args, device_index=None):
        from transformers import AutoModel, AutoTokenizer
        torch, path, config, kwargs = load_settings(args, device_index, 'internvl_chat')
        if config.get('llm_config', {}).get('model_type') != 'qwen3':
            raise ValueError('Expected the OpenGVLab/InternVL3_5-8B Qwen3 checkpoint')
        if args.attn_implementation == 'flash_attention_2':
            from transformers.utils import is_flash_attn_2_available
            if not is_flash_attn_2_available():
                raise RuntimeError('Install flash-attn or use --attn-implementation eager')
        self.model = AutoModel.from_pretrained(
            str(path), trust_remote_code=True,
            use_flash_attn=args.attn_implementation == 'flash_attention_2', **kwargs).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(path), trust_remote_code=True, use_fast=False, local_files_only=True)
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        self.image_size = config.get('force_image_size') or config['vision_config']['image_size']
        self.context_limit = config['llm_config'].get('max_position_embeddings')
        self.args, self.torch, self.dtype = args, torch, kwargs['torch_dtype']

    def generate(self, prompt, frames):
        from PIL import Image
        from torchvision import transforms as T
        from torchvision.transforms.functional import InterpolationMode
        query, separator = prepare_query(self.model, prompt, len(frames), self.args.fps)
        inputs = self.tokenizer(query, return_tensors='pt', truncation=False)
        tokens = inputs['input_ids'].shape[1]
        check_budget(tokens, self.args, self.context_limit)
        text_device = self.model.language_model.get_input_embeddings().weight.device
        vision_device = next(self.model.vision_model.parameters()).device
        transform = T.Compose([
            T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        # Bounded vision batches avoid encoding every frame at once on the GPU.
        # The official generate() accepts precomputed visual_features.
        features = []
        with self.torch.inference_mode():
            for offset in range(0, len(frames), self.args.vision_batch_size):
                chunk = frames[offset:offset + self.args.vision_batch_size]
                pixels = self.torch.stack([transform(Image.fromarray(frame).convert('RGB')) for frame in chunk])
                pixels = pixels.to(device=vision_device, dtype=self.dtype)
                features.append(self.model.extract_feature(pixels).to(text_device))
            visual_features = self.torch.cat(features)
            output = self.model.generate(
                # Non-None pixel_values selects the multimodal branch. When
                # visual_features is supplied, generate does not encode pixels again.
                pixel_values=pixels, visual_features=visual_features,
                input_ids=inputs['input_ids'].to(text_device),
                attention_mask=inputs['attention_mask'].to(text_device),
                eos_token_id=self.tokenizer.convert_tokens_to_ids(separator),
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False, num_beams=1, max_new_tokens=self.args.max_new_tokens)
        # InternVL calls the LM with inputs_embeds: output contains generated
        # tokens only, unlike LLaVA. Do NOT slice by the prompt length here.
        answer = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        return require_answer(answer.split(separator)[0]), tokens


if __name__ == '__main__':
    from predict_streaming import main
    raise SystemExit(main(model='internvl35'))
