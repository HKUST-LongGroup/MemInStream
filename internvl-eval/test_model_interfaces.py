"""CPU adapter tests; optional real-library tests use metadata, not 7B/8B weights."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import subprocess
import sys
from unittest.mock import MagicMock, patch

import download_models
from model_common import check_budget, load_settings
from predict_streaming import parse_args


class InterfaceTests(unittest.TestCase):
    def test_budget_includes_generation_and_model_limit(self):
        args = SimpleNamespace(max_input_tokens=1000, max_new_tokens=100)
        check_budget(900, args, 2000)
        with self.assertRaises(ValueError):
            check_budget(901, args, 2000)
        with self.assertRaises(ValueError):
            check_budget(500, args, 512)

    def test_replica_is_pinned_and_loading_is_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'config.json').write_text('{"model_type": "internvl_chat"}')
            args = SimpleNamespace(model_path=tmp, dtype='auto', device_map='auto')
            cuda = MagicMock()
            torch = SimpleNamespace(cuda=cuda, bfloat16='bf16')
            with patch.dict('sys.modules', torch=torch):
                _, _, _, kwargs = load_settings(args, 1, 'internvl_chat')
            cuda.set_device.assert_called_once_with(1)
            self.assertEqual(kwargs['device_map'], {'': 1})
            self.assertTrue(kwargs['local_files_only'])

    def test_model_specific_attention_defaults(self):
        cli = ['--input', 'i.json', '--output', 'o.json', '--model-path', '/missing']
        self.assertEqual(parse_args(cli, model='internvl35').attn_implementation, 'eager')
        self.assertEqual(parse_args(cli, model='llava_onevision').attn_implementation, 'sdpa')

    def test_model_entry_points_use_final_prompt(self):
        root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as tmp:
            for name in ('internvl35', 'llava_onevision'):
                output = Path(tmp) / f'{name}.json'
                result = subprocess.run([
                    sys.executable, str(root / f'predict_filtered_{name}.py'),
                    '--input', str(root / 'example_samples.json'), '--output', str(output),
                    '--model-path', 'YOUR_MODEL', '--dry-run',
                ], capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                prediction = json.loads(output.read_text())[0]['prediction']
                self.assertEqual(prediction['config']['model'], name)
                self.assertIn('Select all correct', prediction['prompt'])

    def test_download_validates_missing_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'config.json').write_text('{"model_type":"llava_onevision"}')
            (root / 'model.safetensors.index.json').write_text(json.dumps({
                'weight_map': {'a': 'model-00001.safetensors', 'b': 'model-00002.safetensors'}}))
            (root / 'model-00001.safetensors').touch()
            with self.assertRaisesRegex(FileNotFoundError, '00002'):
                download_models.validate_checkpoint(root, 'llava_onevision')
            (root / 'model-00002.safetensors').touch()
            download_models.validate_checkpoint(root, 'llava_onevision')

    def test_download_pins_commit_and_can_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'weights'
            def fake_download(**kwargs):
                self.assertEqual(kwargs['revision'], 'commit-1')
                (root / 'config.json').write_text('{"model_type":"llava_onevision"}')
                (root / 'model.safetensors').touch()
            api = MagicMock()
            api.model_info.return_value.sha = 'commit-1'
            hub = SimpleNamespace(HfApi=lambda: api, snapshot_download=MagicMock(side_effect=fake_download))
            argv = ['--model', 'llava_onevision', '--local-dir', str(root)]
            with patch.dict('sys.modules', huggingface_hub=hub):
                download_models.main(argv)
                download_models.main(argv)
                api.model_info.return_value.sha = 'commit-2'
                with self.assertRaises(FileExistsError):
                    download_models.main(argv)
            self.assertEqual(hub.snapshot_download.call_count, 2)


@unittest.skipUnless(os.environ.get('MODEL_METADATA_ROOT'),
                     'set MODEL_METADATA_ROOT for real-library tiny-model integration')
class NativeInterfaceTests(unittest.TestCase):
    """Real official tokenizers/processors/model code, randomly initialized tiny layers.

    MODEL_METADATA_ROOT contains internvl35/ and llava/ folders downloaded with
    only JSON, Python, tokenizer/model/text files (no checkpoint weights).
    These validate tensor/token/API alignment, NOT pretrained model quality.
    """
    def test_llava_native_video_generation(self):
        import numpy as np
        import torch
        from transformers import AutoProcessor, LlavaOnevisionConfig, LlavaOnevisionForConditionalGeneration
        from predict_filtered_llava_onevision import LlavaOnevisionPredictor
        path = Path(os.environ['MODEL_METADATA_ROOT']) / 'llava'
        torch.manual_seed(7)
        config = LlavaOnevisionConfig.from_pretrained(path, local_files_only=True)
        for cfg in (config.text_config, config.vision_config):
            cfg.hidden_size, cfg.intermediate_size = 32, 64
            cfg.num_hidden_layers, cfg.num_attention_heads = 1, 4
        config.text_config.num_key_value_heads = 2
        engine = LlavaOnevisionPredictor.__new__(LlavaOnevisionPredictor)
        engine.model = LlavaOnevisionForConditionalGeneration(config).float().eval()
        engine.processor = AutoProcessor.from_pretrained(path, local_files_only=True, use_fast=False)
        engine.torch, engine.dtype, engine.context_limit = torch, torch.float32, 4096
        engine.args = SimpleNamespace(fps=1, max_input_tokens=4096, max_new_tokens=2)
        frames = np.zeros((3, 40, 60, 3), dtype=np.uint8)
        answer, tokens = engine.generate('What do you see?', frames)
        self.assertIsInstance(answer, str)
        self.assertGreater(tokens, 3 * 196)

    def test_internvl_native_visual_features_generation(self):
        import numpy as np
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        from predict_filtered_internvl35 import InternVL35Predictor, prepare_query
        path = Path(os.environ['MODEL_METADATA_ROOT']) / 'internvl35'
        torch.manual_seed(7)
        config = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        config.force_image_size = 28
        config.vision_config.image_size = 28
        for cfg in (config.llm_config, config.vision_config):
            cfg.hidden_size, cfg.intermediate_size = 32, 64
            cfg.num_hidden_layers, cfg.num_attention_heads = 1, 4
        config.llm_config.num_key_value_heads = 2
        config.llm_config.head_dim = 8
        engine = InternVL35Predictor.__new__(InternVL35Predictor)
        engine.model = AutoModel.from_config(config, trust_remote_code=True, use_flash_attn=False).float().eval()
        engine.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True,
                                                        use_fast=False, local_files_only=True)
        engine.model.img_context_token_id = engine.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        engine.image_size, engine.context_limit = 28, 4096
        engine.torch, engine.dtype = torch, torch.float32
        engine.args = SimpleNamespace(fps=1, max_input_tokens=4096, max_new_tokens=2, vision_batch_size=1)
        frames = np.zeros((3, 40, 60, 3), dtype=np.uint8)
        query, _ = prepare_query(engine.model, 'What do you see?', 3, 1)
        ids = engine.tokenizer(query)['input_ids']
        self.assertEqual(ids.count(engine.model.img_context_token_id), 3 * engine.model.num_image_token)
        answer, tokens = engine.generate('What do you see?', frames)
        self.assertIsInstance(answer, str)
        self.assertEqual(tokens, len(ids))


if __name__ == '__main__':
    unittest.main()
