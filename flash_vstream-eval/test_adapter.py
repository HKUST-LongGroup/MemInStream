import contextlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import backend
import evaluate
import predict
import official_imports
from prompt_templates import make_prompt
from score_predictions import summarize

HERE = Path(__file__).resolve().parent


def row(**changes):
    return dict(dict(source_dataset='test', source_id='a', video_path='missing.mp4',
                     question='What happened?', query_time=10, question_type='multiple-choice',
                     options=['red', 'blue', 'green'], correct_option=['A', 'C']), **changes)


class AdapterTests(unittest.TestCase):
    def test_explicit_package_import_ignores_generic_models(self):
        with tempfile.TemporaryDirectory() as folder:
            package = Path(folder)
            (package/'__init__.py').write_text('from .implementation import Model\n')
            (package/'implementation.py').write_text('class Model: pass\n')
            name = '_test_flash_explicit_import'
            try:
                with patch.dict(sys.modules, {'models': SimpleNamespace()}):
                    loaded = official_imports.load_package(name, package)
                    self.assertEqual(loaded.Model.__module__, name+'.implementation')
                    self.assertIs(official_imports.load_package(name, package), loaded)
            finally:
                for key in list(sys.modules):
                    if key == name or key.startswith(name+'.'):
                        del sys.modules[key]

    def test_missing_source_reports_exact_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError) as caught:
                official_imports.validate_source(folder)
            self.assertIn(str(Path(folder).resolve()/'Flash-VStream-Qwen/models/__init__.py'), str(caught.exception))

    def test_official_source_hashes(self):
        self.assertEqual(predict.verify_upstream(HERE / 'upstream'),
                         '8f6bde2f397f4846505df4d03364d97617fc02ee')

    def test_prompt_no_current_label_or_future_history(self):
        sample = row(dialogue_events=[dict(time=1, question='Past?', answer='past fact'),
                                          dict(time=10, question='Now?', answer='HIDDEN'),
                                          dict(time=11, question='Future?', answer='HIDDEN')])
        prompt = make_prompt(sample, 5, 10)
        self.assertIn('past fact', prompt)  # outside visual window remains in text
        self.assertNotIn('HIDDEN', prompt)
        self.assertEqual(prompt, make_prompt(dict(sample, correct_option='B', answer='secret'), 5, 10))

    def test_window_and_aliases(self):
        self.assertEqual(predict.window({'Question time': '01:02'}, 30), (32, 62))
        self.assertEqual(predict.window(row(), 0, 9.5), (0, 9.5))
        with self.assertRaises(ValueError):
            predict.window(row(query_time=0), 0)

    def test_causal_sampling_matches_official_two_stage(self):
        import numpy as np
        import torch
        for count in [1, 2, 121, 901, 30001]:
            stamps = np.stack([np.arange(count)/30, (np.arange(count)+1)/30], axis=1)
            start, end = 0., count/30
            got, candidate_count = backend.sample_indices(stamps, start, end, 1, 240, torch)
            n = min(max(round(end), 4), 14400)
            extracted = torch.linspace(0, count-1, n).round().long().clamp(0, count-1)
            ids = torch.linspace(0, n-1, min(n, 240)).round().long().tolist()
            expected = [int(v) for i, v in enumerate(extracted) if i in ids]
            self.assertEqual(candidate_count, n)
            self.assertEqual(got, expected)
            self.assertTrue(all(stamps[i, 0] < end for i in got))
        stamps = np.array([[0., .1], [.1, .2], [.2, .3], [.9, 1.0], [1., 1.1]])
        got, _ = backend.sample_indices(stamps, .15, 1., 1, 240, torch)
        self.assertTrue(all(.15 <= stamps[i, 0] < 1. for i in got))
        self.assertNotIn(4, got)
        with self.assertRaises(ValueError):
            backend.sample_indices(stamps, .3, .8, 1, 240, torch)

    def test_official_vision_helper_pixels_and_padding(self):
        import numpy as np
        from PIL import Image
        spec = importlib.util.spec_from_file_location('official_vision', HERE / 'upstream/Flash-VStream-Qwen/qwen_vl_utils/vision_process.py')
        official = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official)
        images = [Image.fromarray(np.full((224, 336, 3), i, dtype=np.uint8)) for i in range(9)]
        args = SimpleNamespace(chat_system='official', max_frames=6, max_pixels=200704)
        original = official.fetch_video(dict(type='video', video=images, max_frames=6, max_pixels=200704))
        selected = [images[i] for i in [0, 2, 3, 5, 6, 8]]
        _, adapted = official.process_vision_info(backend.build_messages(selected, 'test', args))
        self.assertEqual(len(original), len(adapted[0]))
        for left, right in zip(original, adapted[0]):
            self.assertTrue(np.array_equal(left, right))
        args.max_frames = 240
        _, padded = official.process_vision_info(backend.build_messages(images[:3], 'test', args))
        self.assertEqual(len(padded[0]), 4)
        self.assertTrue(np.array_equal(padded[0][-1], padded[0][-2]))

    def test_generation_contract(self):
        class Tensor:
            shape = (1, 3)
            def cuda(self): return self
            def __iter__(self): return iter([[1, 2, 3]])
            def tolist(self): return [[1, 2, 3]]
        class Inputs(dict):
            def __getattr__(self, key): return self[key]
        calls = {}
        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                calls['messages'], calls['template'] = messages, kwargs
                return 'template'
            def __call__(self, **kwargs):
                calls['processor'] = kwargs
                return Inputs({key: Tensor() for key in ('input_ids', 'attention_mask', 'pixel_values_videos',
                                                         'video_grid_thw', 'visual_position_ids')})
            def batch_decode(self, ids, **kwargs):
                calls['decode'] = (ids, kwargs)
                return [' A,C ']
        class Model:
            def generate(self, **kwargs):
                calls['generate'] = kwargs
                return [[1, 2, 3, 4, 5]]
        engine = backend.Predictor.__new__(backend.Predictor)
        engine.processor, engine.model = Processor(), Model()
        engine.memory = {'flash_memory_temporal_poolsize': 2}
        engine.torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
        engine.args = SimpleNamespace(chat_system='official', max_frames=240, max_pixels=200704)
        engine.process_vision_info = lambda messages: (None, ['frames'])
        answer, stats = engine.generate('exact user prompt', ['image'])
        self.assertEqual(answer, 'A,C')
        self.assertEqual(calls['processor']['text'], ['template'])
        self.assertEqual(calls['messages'][0]['content'][1]['text'], 'exact user prompt')
        self.assertEqual(len(calls['messages']), 1)
        self.assertEqual(calls['generate']['max_new_tokens'], 128)
        self.assertEqual(calls['generate']['top_k'], 1)
        self.assertIs(calls['generate']['do_sample'], False)
        self.assertIn('visual_position_ids', calls['generate'])
        self.assertEqual(calls['decode'], ([[4, 5]], dict(skip_special_tokens=True, clean_up_tokenization_spaces=False)))
        self.assertEqual(stats['input_tokens'], 3)

    def test_qwen_system_is_opt_in(self):
        args = SimpleNamespace(chat_system='qwen', max_frames=240, max_pixels=200704)
        self.assertEqual(backend.build_messages([], 'x', args)[0]['role'], 'system')

    def test_score_full_coverage_and_multiselect(self):
        rows = [row(source_id=str(i)) for i in range(4)]
        predictions = []
        for i, answer in enumerate(['C,A', 'A', 'A,B,C']):
            pred = dict(status='completed', input_index=i, answer=answer, response=answer)
            predictions.append(dict(rows[i], flash_vstream_prediction=pred))
        aligned = evaluate.align_predictions(rows, predictions)
        report, _ = summarize(aligned)
        self.assertEqual(report['overall']['accuracy'], .25)
        self.assertEqual(report['overall']['prediction_failed_or_missing'], 1)
        with self.assertRaises(ValueError):
            evaluate.align_predictions(rows, predictions + predictions[:1])
        with self.assertRaises(ValueError):
            evaluate.align_predictions(rows, [dict(predictions[0], question='different')])
        with self.assertRaises(ValueError):
            summarize([dict(row(), prediction={'status': 'dry_run'})])

    def test_real_spawn_resume_and_failure_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            inp, out = folder/'input.json', folder/'out.jsonl'
            inp.write_text(json.dumps([row(source_id=str(i)) for i in range(6)]))
            command = [sys.executable, str(HERE/'predict.py'), '--input', str(inp), '--output', str(out),
                       '--dry-run', '--num-gpus', '2']
            run = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stderr)
            records = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual([r['prediction']['row_index'] for r in records], list(range(6)))
            journal = Path(str(out)+'.progress.jsonl')
            # Resume a non-contiguous subset, including a truncated final append.
            journal.write_text('\n'.join(json.dumps(records[i]) for i in [1, 4])+'\n{"broken"')
            run = subprocess.run(command+['--resume'], capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(len(journal.read_text().splitlines()), 6)
            bad = subprocess.run(command+['--resume', '--history-seconds', '20'], capture_output=True, text=True, timeout=60)
            self.assertNotEqual(bad.returncode, 0)
            inp.write_text(json.dumps([row(query_time=0)]))
            failout = folder/'failed.jsonl'
            failed = subprocess.run([sys.executable, str(HERE/'predict.py'), '--input', str(inp),
                                     '--output', str(failout), '--dry-run'], capture_output=True, text=True, timeout=60)
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(json.loads(failout.read_text())['prediction']['status'], 'error')


if __name__ == '__main__':
    unittest.main()
