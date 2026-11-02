"""Optional: tiny random model API smoke test, no checkpoint downloads.
Run in the pinned Qwen environment. CUDA memory utilities are mocked on CPU;
the real official encoder, compressor and decoder execute. Not an accuracy test.
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from transformers import Qwen2_5_VLConfig,Qwen2_5_VLForConditionalGeneration
from transformers.feature_extraction_utils import BatchFeature
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'upstream/HERMES')]
from inference import qwenvl_hermes as official
from hermes_adapter import run_stream
class Tokenizer:
 eos_token_id=31
 def __call__(self,text,return_tensors=None):
  return SimpleNamespace(input_ids=torch.tensor([[1,2]]) if return_tensors else [1,2])
 def decode(self,ids,**kw):return 'A'
class Processor:
 tokenizer=Tokenizer()
 def __call__(self,**kw):
  return BatchFeature({'pixel_values_videos':torch.randn(16,24),'video_grid_thw':torch.tensor([[1,4,4]])})
config=Qwen2_5_VLConfig(text_config=dict(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,rope_scaling={'type':'mrope','mrope_section':[1,1,2]}),vision_config=dict(depth=1,hidden_size=32,intermediate_size=64,num_heads=4,in_channels=3,patch_size=2,spatial_merge_size=2,temporal_patch_size=2,out_hidden_size=32,window_size=8,fullatt_block_indexes=[0]))
config._attn_implementation='sdpa'
base=Qwen2_5_VLForConditionalGeneration(config).eval()
with patch.object(official.Qwen2_5_VLProcessor,'from_pretrained',return_value=Processor()),patch.object(official.Qwen2_5_VLForConditionalGeneration,'from_pretrained',return_value=base):
 model,processor=official.load_model('unused',device='cpu',kv_size=4)
args=SimpleNamespace(chunk_frames=2,max_new_tokens=3,repetition_penalty=1.)
with patch('torch.cuda.ipc_collect'),patch('torch.cuda.empty_cache'),patch('torch.cuda.max_memory_allocated',return_value=0),torch.inference_mode():
 answer,diagnostics=run_stream(model,torch.zeros(4,8,8,3), 'Which object?', args)
 print('SMOKE OK',answer,diagnostics)
