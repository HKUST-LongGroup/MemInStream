"""Real tiny Qwen3-VL forward/cache tests. No checkpoint download or CUDA needed."""
import unittest
import copy
import tempfile
from unittest.mock import patch
import importlib.util
if any(importlib.util.find_spec(name) is None for name in ['torch', 'transformers']):
    raise unittest.SkipTest('Install this method requirements to run real-model integration tests')
import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from qwen3_fluxmem.model import Qwen3VLFluxMemForConditionalGeneration, prepare_compressed_inputs, memory_coordinates, encode_video


class Qwen3ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        config = Qwen3VLConfig(
            text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                head_dim=8, max_position_embeddings=2048,
                rope_scaling={'rope_type':'default', 'mrope_section':[1,1,2]}),
            vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=4,
                patch_size=2, spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=32,
                num_position_embeddings=16, deepstack_visual_indexes=[0,1]),
            image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63)
        config._attn_implementation = 'eager'
        cls.model = Qwen3VLFluxMemForConditionalGeneration(config).eval()
        cls.native = Qwen3VLForConditionalGeneration(config).eval()
        cls.native.load_state_dict(cls.model.state_dict())

    def inputs(self, t=6):
        ids = [1]
        for frame in range(t):
            ids += [2, 3, 62] + [61]*4 + [63]
        ids += [4, 5]
        # Exactly repeated frames trigger temporal similarity pruning.
        pixels = torch.randn(16, 3*2*2*2).repeat(t, 1)
        return dict(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1,len(ids),dtype=torch.long),
                    video_grid_thw=torch.tensor([[t,4,4]]), pixel_values_videos=pixels)

    def test_no_pruning_matches_native_logits_and_generation(self):
        inputs=self.inputs(2)
        with torch.inference_mode():
            packed, stats=prepare_compressed_inputs(self.model,inputs,short_frames=8)
            original=self.native(**inputs,use_cache=False).logits
            actual=self.model(**packed,use_cache=False).logits
            self.assertEqual(stats['input_tokens'],stats['retained_tokens'])
            torch.testing.assert_close(actual,original,rtol=1e-5,atol=1e-5)
            original_ids=self.native.generate(**inputs,max_new_tokens=3,do_sample=False,eos_token_id=None,pad_token_id=0)
            actual_ids=self.model.generate(**packed,max_new_tokens=3,do_sample=False,eos_token_id=None,pad_token_id=0)
            self.assertTrue(torch.equal(original_ids,actual_ids))

    def test_pruning_keeps_text_and_native_positions_and_deepstack(self):
        inputs=self.inputs()
        with torch.inference_mode():
            full,fullstats=prepare_compressed_inputs(self.model,inputs,short_frames=100)
            packed,stats=prepare_compressed_inputs(self.model,inputs,short_frames=2,medium_frames=2)
        self.assertLess(stats['retained_visual_tokens'],stats['visual_tokens'])
        before=inputs['input_ids'][inputs['input_ids']!=61]
        after=packed['input_ids'][packed['input_ids']!=61]
        self.assertTrue(torch.equal(before,after))
        self.assertTrue(torch.equal(packed['flux_position_ids'][:,:,-2:],full['flux_position_ids'][:,:,-2:]))
        for feature in packed['flux_deepstack']:
            self.assertEqual(len(feature),stats['retained_visual_tokens'])
        self.assertEqual(int(packed['flux_rope_delta'])+stats['retained_tokens'],int(full['flux_position_ids'].max())+1)
        with torch.inference_mode():
            first=self.model(**packed,use_cache=True)
            self.assertEqual(first.past_key_values.get_seq_length(),stats['retained_tokens'])
            cache_pos=torch.tensor([stats['retained_tokens']])
            continuation=dict(input_ids=torch.tensor([[6]]),attention_mask=torch.ones(1,stats['retained_tokens']+1,dtype=torch.long),
                cache_position=cache_pos, past_key_values=first.past_key_values,
                **{k:v for k,v in packed.items() if k.startswith('flux_')})
            with patch.object(self.model.model.language_model,'_deepstack_process',wraps=self.model.model.language_model._deepstack_process) as deep:
                second=self.model(**continuation,use_cache=True)
                self.assertEqual(deep.call_count,0)
                self.assertEqual(second.past_key_values.get_seq_length(),stats['retained_tokens']+1)
            out=self.model.generate(**packed,max_new_tokens=4,do_sample=False,eos_token_id=None,pad_token_id=0)
            self.assertEqual(out.shape[1],stats['retained_tokens']+4)

    def test_deepstack_kept_indices_exact_and_generation_matches_manual_decode(self):
        inputs=self.inputs(4)
        with torch.inference_mode():
            packed,stats=prepare_compressed_inputs(self.model,inputs,short_frames=2,medium_frames=2)
            _,original_deep=encode_video(self.model,inputs['pixel_values_videos'],inputs['video_grid_thw'])
            # The last temporal grid is always retained, including every DeepStack layer.
            for full,pruned in zip(original_deep,packed['flux_deepstack']):
                torch.testing.assert_close(full[-4:],pruned[-4:])
            first=self.model(**packed,use_cache=True)
            tokens=[first.logits[:,-1].argmax(-1).item()]
            cache=first.past_key_values
            for step in range(2):
                result=self.model(input_ids=torch.tensor([[tokens[-1]]]), past_key_values=cache,
                    attention_mask=torch.ones(1,stats['retained_tokens']+step+1,dtype=torch.long),
                    cache_position=torch.tensor([stats['retained_tokens']+step]),use_cache=True,
                    **{k:v for k,v in packed.items() if k.startswith('flux_')})
                cache=result.past_key_values;tokens.append(result.logits[:,-1].argmax(-1).item())
            generated=self.model.generate(**packed,max_new_tokens=3,do_sample=False,eos_token_id=None,pad_token_id=0)
            self.assertEqual(generated[0,-3:].tolist(),tokens)

    def test_vision_chunking_matches_unchunked_including_deepstack(self):
        inputs=self.inputs(3)
        with torch.inference_mode():
            features,deep=encode_video(self.model,inputs['pixel_values_videos'],inputs['video_grid_thw'])
            chunked,chunkdeep=encode_video(self.model,inputs['pixel_values_videos'],inputs['video_grid_thw'],2)
        torch.testing.assert_close(features,chunked,atol=1e-5,rtol=1e-5)
        for a,b in zip(deep,chunkdeep):torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)

    def test_local_memory_coordinates_are_not_native_rope(self):
        inputs=self.inputs(3)
        coords,visual=memory_coordinates(inputs['input_ids'],inputs['video_grid_thw'],self.model.config)
        self.assertEqual(coords[0,0,visual].tolist(),[0]*4+[1]*4+[2]*4)
        self.assertEqual(coords[1,0,visual].tolist(),[0,0,1,1]*3)
        native,_=self.model.model.get_rope_index(inputs['input_ids'],video_grid_thw=inputs['video_grid_thw'])
        self.assertFalse(torch.equal(coords[:,0,visual],native[:,0,visual]))

    def test_reject_padding_and_multiple_videos(self):
        inputs=self.inputs()
        inputs['attention_mask'][0,0]=0
        with self.assertRaisesRegex(ValueError,'Padding'):prepare_compressed_inputs(self.model,inputs)
        inputs=self.inputs();inputs['video_grid_thw']=inputs['video_grid_thw'].repeat(2,1)
        with self.assertRaisesRegex(ValueError,'one video'):prepare_compressed_inputs(self.model,inputs)

    def test_sdpa_checkpoint_reload_and_compressed_generation(self):
        # Actual from_pretrained exercises weight naming and generation mixin.
        with tempfile.TemporaryDirectory() as tmp:
            self.native.save_pretrained(tmp)
            loaded=Qwen3VLFluxMemForConditionalGeneration.from_pretrained(tmp,attn_implementation='sdpa').eval()
            inputs=self.inputs(4)
            with torch.inference_mode():
                packed,stats=prepare_compressed_inputs(loaded,inputs,short_frames=2,medium_frames=2)
                result=loaded.generate(**packed,max_new_tokens=3,do_sample=False,
                    eos_token_id=None,pad_token_id=0,cache_implementation='dynamic')
            self.assertLess(stats['retained_visual_tokens'],stats['visual_tokens'])
            self.assertEqual(result.shape[1],stats['retained_tokens']+3)


if __name__=='__main__':unittest.main()
