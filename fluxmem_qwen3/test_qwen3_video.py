"""Real video processor/decode tests with synthetic inputs (CPU)."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import importlib.util
if any(importlib.util.find_spec(name) is None for name in ['torch', 'transformers', 'tokenizers', 'av']):
    raise unittest.SkipTest('Install this method requirements to run real-model integration tests')
import torch
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import Qwen2TokenizerFast, Qwen2VLImageProcessorFast, Qwen3VLProcessor, Qwen3VLVideoProcessor
from qwen3_fluxmem import video
from qwen3_fluxmem.model import prepare_compressed_inputs
import test_qwen3_model as tiny


class VideoTests(unittest.TestCase):
    def test_torchvision_anchor_metadata_uses_actual_selected_times(self):
        # 100 source frames at 10 fps, but capped at four samples anchored to end.
        frames=torch.zeros(100,3,32,32,dtype=torch.uint8)
        ele=dict(video='unused',video_start=50.,video_end=60.,fps=1.,min_frames=4,max_frames=4,anchor_end=True)
        with patch.object(video.io,'read_video',return_value=(frames,torch.empty(0),{'video_fps':10.})):
            sampled,_=video._read_video_torchvision(ele)
        self.assertEqual(len(sampled),4)
        self.assertEqual(ele['_sample_metadata']['frames_indices'],[69.,79.,89.,99.])
        self.assertEqual(ele['_sample_metadata']['fps'],10.)
        # Those are clip-local seconds 6.9..9.9, NOT 0..7.5 from an inferred average fps.
        timestamps=Qwen3VLProcessor._calculate_timestamps(None,ele['_sample_metadata']['frames_indices'],10,2)
        np.testing.assert_allclose(timestamps,[7.4,9.4])

    def test_native_processor_metadata_deepstack_and_generation(self):
        vocab={f't{i}':i for i in range(60)}
        vocab.update({'<|image_pad|>':60,'<|video_pad|>':61,'<|vision_start|>':62,'<|vision_end|>':63})
        backend=Tokenizer(WordLevel(vocab,unk_token='t0'));backend.pre_tokenizer=Whitespace()
        tokenizer=Qwen2TokenizerFast(tokenizer_object=backend,unk_token='t0',pad_token='t0',
            eos_token='t59',additional_special_tokens=['<|image_pad|>','<|video_pad|>','<|vision_start|>','<|vision_end|>'])
        processor=Qwen3VLProcessor(tokenizer=tokenizer,
            image_processor=Qwen2VLImageProcessorFast(patch_size=2,temporal_patch_size=2,merge_size=2),
            video_processor=Qwen3VLVideoProcessor(patch_size=2,temporal_patch_size=2,merge_size=2))
        frames=torch.randint(0,256,(8,3,8,8),dtype=torch.uint8)
        metadata=dict(total_num_frames=100,frames_indices=[10,20,30,40,50,60,70,80],fps=10.)
        inputs=processor(text=['t1 <|vision_start|><|video_pad|><|vision_end|> t2'],videos=[frames],
            video_metadata=[metadata],do_sample_frames=False,do_resize=False,return_tensors='pt',padding=True)
        self.assertEqual(inputs['video_grid_thw'].tolist(),[[4,4,4]])
        self.assertEqual(int((inputs['input_ids']==61).sum()),16)
        tiny.Qwen3ModelTests.setUpClass()
        model=tiny.Qwen3ModelTests.model
        with torch.inference_mode():
            packed,stats=prepare_compressed_inputs(model,inputs,short_frames=2,medium_frames=2)
            generated=model.generate(**packed,max_new_tokens=2,do_sample=False,eos_token_id=None,pad_token_id=0)
        self.assertEqual(generated.shape[1],stats['retained_tokens']+2)

    def test_real_video_decode_to_query_window(self):
        import av
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'clip.mp4'
            with av.open(str(path),'w') as container:
                stream=container.add_stream('mpeg4',rate=10);stream.width=64;stream.height=64;stream.pix_fmt='yuv420p'
                for i in range(50):
                    frame=av.VideoFrame.from_ndarray(np.full((64,64,3),i*4,dtype=np.uint8),format='rgb24')
                    for packet in stream.encode(frame):container.mux(packet)
                for packet in stream.encode():container.mux(packet)
            element=dict(video=str(path),video_start=1.,video_end=4.,anchor_end=True,
                         fps=2.,min_frames=4,max_frames=6,min_pixels=1024,max_pixels=4096)
            with patch.object(video,'get_video_reader_backend',return_value='torchvision'):
                frames,_=video.fetch_video(element,image_factor=32,return_video_sample_fps=True)
            self.assertEqual(len(frames),6)
            self.assertEqual(frames.shape[-2]%32,0)
            self.assertEqual(frames.shape[-1]%32,0)
            times=np.asarray(element['_sample_metadata']['frames_indices'])/10+1
            self.assertTrue(np.all(times>=1))
            self.assertTrue(np.all(times<=4))


if __name__=='__main__':unittest.main()
