"""Optional CPU API check with tiny random LLaVA weights in its pinned environment.
No pretrained weights or tokenizer files are downloaded; this is not accuracy.
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from transformers import LlavaOnevisionConfig, LlavaOnevisionForConditionalGeneration
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'upstream/HERMES')]
from inference import llavaov_hermes as official
from hermes_adapter import run_stream

class Tokenizer:
    eos_token_id = 31
    def __call__(self, text, return_tensors=None):
        return SimpleNamespace(input_ids=torch.tensor([[1, 2]]) if return_tensors else [1, 2])
    def decode(self, ids, **kw):
        return 'A'

class Processor:
    tokenizer = Tokenizer()
    def video_processor(self, video, **kw):
        return SimpleNamespace(pixel_values_videos=video.unsqueeze(0).float())

config = LlavaOnevisionConfig(
    text_config=dict(model_type='qwen2', vocab_size=32, hidden_size=32, intermediate_size=64,
                     num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2),
    vision_config=dict(model_type='siglip_vision_model', hidden_size=32, intermediate_size=64,
                       num_hidden_layers=1, num_attention_heads=4, image_size=8, patch_size=2),
    image_token_index=29, video_token_index=30, vision_feature_layer=-1,
)
config._attn_implementation = 'sdpa'
base = LlavaOnevisionForConditionalGeneration(config).eval()
with patch.object(official.AutoProcessor, 'from_pretrained', return_value=Processor()), \
     patch.object(official.LlavaOnevisionForConditionalGeneration, 'from_pretrained', return_value=base):
    model, processor = official.load_model('unused', device='cpu', kv_size=4)
args = SimpleNamespace(chunk_frames=2, max_new_tokens=3, repetition_penalty=1.)
with patch('torch.cuda.ipc_collect'), patch('torch.cuda.empty_cache'), \
     patch('torch.cuda.max_memory_allocated', return_value=0), torch.inference_mode():
    for _ in range(2):
        answer, diagnostics = run_stream(model, torch.zeros(4, 8, 8, 3), 'Which object?', args)
        assert len(diagnostics['chunks']) == 2
        assert len(model.conv_history) == 1
    print('SMOKE OK', answer, diagnostics)
