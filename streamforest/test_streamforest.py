import ast
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import predict
from prompt_templates import make_prompt
from qwen_eval_utils import probe_video, read_rows
from score_predictions import summarize
from streamforest_backend import Predictor, validate_config, verify_upstream
from video_input import decode_video, window

HERE = Path(__file__).resolve().parent


def row(**updates):
    result = dict(source_dataset='test', source_id='a', video_path='missing.mp4',
                  question='Which object?', options=['A. Cup', 'B. Plate'],
                  correct_option='A', answer='SECRET', query_time=100,
                  level2='Attribute Recall', question_type='multiple-choice')
    result.update(updates)
    return result


def args(*extra):
    return predict.parse_args(['--input', 'unused.json', '--output', 'unused-out.json', *extra])


class PromptAndWindowTests(unittest.TestCase):
    def test_prompt_and_scorer_are_verbatim_reference(self):
        manifest = json.loads((HERE / 'reference_manifest.json').read_text())
        for name in ['prompt_templates.py', 'qwen_eval_utils.py', 'score_predictions.py']:
            self.assertEqual(hashlib.sha256((HERE/name).read_bytes()).hexdigest(), manifest['sha256'][name])
        self.assertTrue(manifest['fluxmem_prompt_identical'])

    def test_current_answer_and_future_dialogue_excluded(self):
        sample = row(dialogue_events=[
            dict(time=2, role='assistant', content='PAST'),
            dict(time=100, role='assistant', content='SECRET_AT_QUERY'),
            dict(time=101, role='assistant', content='SECRET_FUTURE')])
        text = make_prompt(sample, 40, 100)
        self.assertIn('PAST', text)
        self.assertNotIn('SECRET', text)
        self.assertEqual(text, make_prompt(dict(sample, answer='OTHER', correct_option='B'), 40, 100))

    def test_final_prompt_ignores_question_category(self):
        text = make_prompt(row(), 0, 100)
        self.assertIn('exactly match all correct', text)
        self.assertEqual(text, make_prompt(row(level2='Occurrence Verification'), 0, 100))

    def test_windows_and_duration_clamp(self):
        self.assertEqual(window(row(query_time='01:40'), 60), (40, 100))
        self.assertEqual(window(row(), 0), (0, 100))
        self.assertEqual(window(row(), 60, 90), (40, 90))
        with self.assertRaises(ValueError):
            window(row(), 60, 30)

    def test_bad_row_not_silently_dropped(self):
        a = args('--dry-run')
        result = predict.predict_row(5, row(query_time=0), a, None)
        self.assertEqual(result['prediction']['row_index'], 5)
        self.assertEqual(result['prediction']['status'], 'error')
        report, _ = summarize([result])
        self.assertEqual(report['overall']['total'], 1)
        self.assertEqual(report['overall']['accuracy'], 0)

    def test_budget_and_protocol_errors(self):
        result = predict.predict_row(0, row(query_time=1000), args('--dry-run', '--max-frames', '4'), None)
        self.assertIn('Frame budget', result['prediction']['error'])
        result = predict.predict_row(0, row(evaluation_protocol='forward'), args('--dry-run'), None)
        self.assertEqual(result['prediction']['status'], 'error')

    def test_upstream_integrity_and_checkpoint_guard(self):
        self.assertEqual(verify_upstream(HERE/'StreamForest')['revision'],
                         'f9d14670ff451ff5865638e17c9958e99bf2fec5')
        with self.assertRaises(ValueError):
            validate_config({'architectures': ['Qwen2_5_VLForConditionalGeneration']})
        validate_config(dict(architectures=['LlavaQwenForCausalLM'],
                             mm_projector_type='tome729_fstw_pemf', mm_patch_merge_type='flat'))

    def test_memory_budget_checks_full_multimodal_length(self):
        backend = Predictor.__new__(Predictor)
        backend.text_tokens = 25
        backend.context_limit = 100
        backend.args = types.SimpleNamespace(max_new_tokens=10)
        features = types.SimpleNamespace(shape=[1, 50, 8], numel=lambda: 400)
        backend._check_visual_budget(None, None, features)
        self.assertEqual(backend.input_tokens, 74)
        backend.context_limit = 80
        with self.assertRaises(ValueError):
            backend._check_visual_budget(None, None, features)

    def test_upstream_memory_is_fresh_per_sample(self):
        source = (HERE/'StreamForest/llava/model/multimodal_projector/projector_FSTW_PEMF.py').read_text()
        tree = ast.parse(source)
        forward = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        calls = [n.func.id for n in ast.walk(forward) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        self.assertIn('MemoryManager', calls)


class IntegrationTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('torch'), 'requires torch for mocked generation contract')
    def test_official_generate_contract_and_generated_only_decode(self):
        import torch
        import streamforest_backend as backend_module
        backend = Predictor.__new__(Predictor)
        backend.args = types.SimpleNamespace(system_prompt='aligned', max_new_tokens=8)
        backend.torch, backend.device, backend.image_index = torch, torch.device('cpu'), -200
        backend.context_limit = 100
        calls = {}
        backend.tokenizer_image_token = lambda *a, **kw: torch.tensor([7, -200, 8])
        backend.stopping_class = lambda *a: object()
        backend.processor = types.SimpleNamespace(preprocess=lambda *a, **kw: {'pixel_values': torch.zeros(2,3,4,4)})
        def generate(ids, **kwargs):
            calls.update(kwargs)
            backend.input_tokens = 12
            return torch.tensor([[65]])
        def decode(tokens, **kwargs):
            self.assertEqual(tokens.tolist(), [[65]])  # no slicing by text length
            return ['A']
        backend.model = types.SimpleNamespace(generate=generate)
        backend.tokenizer = types.SimpleNamespace(pad_token_id=None, eos_token_id=0, batch_decode=decode)
        with patch.object(backend_module, 'render_prompt', return_value=('rendered', '<|im_end|>')):
            self.assertEqual(backend.generate('question', None), ('A', 12))
        self.assertEqual(calls['modalities'], ['video'])
        self.assertFalse(calls['do_sample'])
        self.assertEqual(calls['num_beams'], 1)
        self.assertTrue(calls['use_cache'])
        self.assertEqual(calls['max_new_tokens'], 8)
        self.assertEqual(calls['images'][0].dtype, torch.float16)

    def test_multigpu_dry_run_order_and_final_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'input.json'
            samples = [row(source_id=str(i), query_time=i+5) for i in range(7)]
            source.write_text(json.dumps(samples))
            output = Path(tmp)/'output.json'
            run = subprocess.run([sys.executable, '-B', str(HERE/'predict.py'),
                '--input', str(source), '--output', str(output), '--dry-run', '--num-gpus', '3'],
                capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
            records = read_rows(output)
            self.assertEqual([r['prediction']['row_index'] for r in records], list(range(7)))
            self.assertTrue(all(r['prediction']['status']=='dry_run' for r in records))
            self.assertTrue(all('v2' in r['prediction']['prompt_version'] for r in records))
            self.assertFalse(Path(str(output)+'.accuracy.json').exists())
            with self.assertRaises(ValueError):
                summarize(records)

    def test_fake_inference_scores_errors_and_legacy_options(self):
        class FakeEngine:
            visual_tokens = 5
            info = {'model_class': 'FakeTestOnly'}
            torch = types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: None))
            def __init__(self, args):
                pass
            def generate(self, prompt, frames):
                return 'A', 20
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            video = tmp/'video.mp4'
            video.touch()
            good = row(video_path=str(video))
            good['Options'] = good.pop('options')
            source, output = tmp/'input.json', tmp/'output.jsonl'
            source.write_text(json.dumps([good, row(source_id='bad', query_time=0)]))
            with patch.object(predict, 'Predictor', FakeEngine), \
                 patch.object(predict, 'probe_video', return_value=(64, 64, 100)), \
                 patch.object(predict, 'decode_video', return_value=(None, {})), contextlib.redirect_stdout(io.StringIO()):
                code = predict.main(['--input', str(source), '--output', str(output)])
            self.assertEqual(code, 1)
            report = json.loads(Path(str(output)+'.accuracy.json').read_text())
            self.assertEqual(report['overall']['accuracy'], 0.5)
            self.assertEqual(report['overall']['prediction_failed_or_missing'], 1)
            with self.assertRaises(FileExistsError):
                predict.main(['--input', str(source), '--output', str(output), '--dry-run'])

    @unittest.skipUnless(shutil.which('ffmpeg') and importlib.util.find_spec('numpy'), 'requires ffmpeg/numpy')
    def test_real_video_future_frames_excluded_and_padding(self):
        for command in ('ffmpeg', 'ffprobe'):
            if not shutil.which(command):
                self.skipTest(f'{command} is unavailable')
            check = subprocess.run([command, '-version'], capture_output=True, text=True, timeout=10)
            if check.returncode:
                self.skipTest(f'{command} cannot start; install a working FFmpeg build')
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp)/'boundary.mkv'
            # Before t=3 red; at/after t=3 blue. Decode only [0,3).
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=red:s=64x64:r=10:d=3',
                '-f', 'lavfi', '-i', 'color=c=blue:s=64x64:r=10:d=2', '-filter_complex',
                '[0:v][1:v]concat=n=2:v=1:a=0[v]', '-map', '[v]', '-threads', '1', '-c:v', 'ffv1', str(video)],
                check=True, capture_output=True, timeout=30)
            metadata = probe_video(str(video), 30)
            frames, stats = decode_video(video, 0, 3, metadata, args())
            self.assertEqual(stats['decoded_frames'], 3)
            self.assertEqual(stats['padding_frames'], 1)
            self.assertTrue(np.all(frames[..., 0].mean(axis=(1,2)) > 200))
            self.assertTrue(np.all(frames[..., 2].mean(axis=(1,2)) < 20))
            np.testing.assert_array_equal(frames[-1], frames[-2])
            cropped, stats = decode_video(video, 1, 3, metadata, args())
            self.assertEqual(stats['decoded_frames'], 2)
            self.assertTrue(np.all(cropped[..., 2] < 20))


if __name__ == '__main__':
    unittest.main()
