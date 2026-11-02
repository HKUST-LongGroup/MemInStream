import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hermes_adapter import run_stream, verify_upstream
from predict import parse_args, prepare_rows, predict_row, recover
from prompt_templates import make_prompt
from official_video import load_query_video, sample_indices
from evaluate import evaluate_rows, join_predictions


def row(**changes):
    value = dict(source_id='0', source_dataset='test', video_path='test.mp4', query_time=10,
                 question='Which objects?', options=['A. Cup', 'B. Plate', 'C. Ball'],
                 question_type='multiple-select', correct_option=['A', 'C'])
    value.update(changes)
    return value


def args(*extra):
    return parse_args(['--input', '/tmp/input.json', '--output', '/tmp/output.json',
                       '--model', 'qwen25', '--model-path', '/tmp/model', *extra])


class FakeModel:
    num_layers = 4
    def __init__(self):
        self._layer_position_ids = {0: 'stale'}
        self._position_ids_cache = ['stale'] * 4
        self.token_activity_cache = ['stale'] * 4
        self.conv_history = [('old', 'answer', None)]
        self.calls = []
    def clear_cache(self):
        self.calls.append('clear')
        self.kv_cache = None
        self.conv_history = []
    def encode_init_prompt(self):
        assert not self._layer_position_ids
        assert self._position_ids_cache == [None] * 4
        assert self.token_activity_cache == [None] * 4
        assert not self.conv_history
        self.calls.append('init')
    def encode_video_chunk(self, chunk):
        self.calls.append(('encode', len(chunk)))
    def predict_and_compress(self):
        self.calls.append('compress')
    def _get_cache_seq_len_per_layer(self):
        return [6000] * 4
    def get_prompt(self, text, mc=False):
        assert mc is False
        return text + '<assistant>'
    def question_answering(self, prompt, **kwargs):
        assert 'Best option' not in prompt['prompt']
        self.calls.append(('answer', kwargs))
        self.conv_history = [('current', 'A,C', None)]
        return 'A,C'


class AdapterTests(unittest.TestCase):
    def test_snapshot(self):
        self.assertEqual(verify_upstream(ROOT / 'upstream/HERMES')['revision'], '8d699b16a6bedb9086c1b39ec4253c6a1d1ce789')

    def test_history_and_prompt(self):
        r = row(dialogue_events=[dict(time=2, role='assistant', content='VISIBLE'),
                                 dict(time=10, role='assistant', content='CURRENT'),
                                 dict(time=11, role='assistant', content='FUTURE')])
        plan = prepare_rows([r], args())[0]
        self.assertEqual((plan['window_start'], plan['window_end']), (0, 10))
        self.assertEqual(plan['prompt'], make_prompt(r, 0, 10))
        self.assertIn('VISIBLE', plan['prompt'])
        self.assertNotIn('CURRENT', plan['prompt'])
        self.assertNotIn('FUTURE', plan['prompt'])
        self.assertNotIn('A,C', plan['prompt'])

    def test_no_future_protocol(self):
        with self.assertRaises(ValueError):
            prepare_rows([row(evaluation_protocol='forward')], args())

    def test_reset_and_chunk_order(self):
        model = FakeModel()
        answer, info = run_stream(model, list(range(34)), 'SELECT', args())
        self.assertEqual(answer, 'A,C')
        self.assertEqual(model.calls[:8], ['clear', 'init', ('encode', 16), 'compress', ('encode', 16), 'compress', ('encode', 2), 'compress'])
        self.assertEqual(model.calls[-1][1]['repetition_penalty'], 1.1)
        self.assertEqual(model.calls[-1][1]['temperature'], 0)
        self.assertEqual(len(info['chunks']), 3)
        run_stream(model, [1, 2], 'NEXT INDEPENDENT QUESTION', args())

    def test_same_sampling_for_both_models(self):
        import numpy as np
        frames = np.zeros((2, 28, 28, 3), dtype=np.uint8)
        class Engine:
            def generate(self, prompt, actual, max_new_tokens=None):
                self.prompt = prompt
                assert actual is frames
                assert max_new_tokens == 16
                return 'A,C', {}
        prompts = []
        for model in ['qwen25', 'llavaov']:
            a = args(); a.model = model
            e = Engine()
            with patch('predict.load_query_video', return_value=(frames, {})) as decoder:
                result = predict_row(4, row(), a, e)
            self.assertEqual(result['prediction']['status'], 'completed')
            self.assertEqual(result['prediction']['row_index'], 4)
            self.assertEqual(decoder.call_args.args[1], 10)
            self.assertEqual(decoder.call_args.args[2].model, model)
            prompts.append(e.prompt)
        self.assertEqual(prompts[0], prompts[1])

    def test_scoring_denominator_and_multiselect(self):
        rows = [row(), row(source_id='1'), row(source_id='2', question_type='open-ended', options=[], answer='cup')]
        predictions = [dict(rows[0], prediction=dict(row_index=0, status='completed', answer='C,A'))]
        joined = join_predictions(rows, predictions)
        qreport, _ = evaluate_rows(joined)
        self.assertEqual(qreport['overall']['accuracy'], 0.5)
        self.assertEqual(qreport['excluded_open_ended_or_other'], 1)
        freport, _ = evaluate_rows(joined, 'fluxmem')
        self.assertEqual(freport['objective']['accuracy_scored'], 1)
        self.assertEqual(freport['objective']['correct_over_total_lower_bound'], 0.5)
        joined[0]['prediction']['answer'] = 'A'
        self.assertEqual(evaluate_rows(joined)[0]['overall']['correct'], 0)

    def test_scoring_rejects_dry_duplicate_or_wrong_source(self):
        r = row()
        for status in ['dry_run', 'completed']:
            pred = dict(r, prediction=dict(row_index=0, status=status, answer='A,C'))
            with self.assertRaises(ValueError):
                join_predictions([r], [pred] if status == 'dry_run' else [pred, pred])
        with self.assertRaises(ValueError):
            join_predictions([r], [dict(r, correct_option='B', prediction=dict(row_index=0, status='completed', answer='A'))])

    def test_cli_parallel_and_partial_resume(self):
        with tempfile.TemporaryDirectory() as d:
            data, out = Path(d)/'input.json', Path(d)/'out.json'
            data.write_text(json.dumps([row(source_id=str(i)) for i in range(4)]))
            command = [sys.executable, str(ROOT/'predict.py'), '--input', str(data), '--output', str(out),
                       '--model', 'llavaov', '--model-path', '/tmp/no-weights', '--dry-run', '--num-gpus', '2']
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            records = json.loads(out.read_text())
            self.assertEqual([r['prediction']['row_index'] for r in records], [0,1,2,3])
            # Simulate a failed middle row. Only this nonzero index is dispatched on resume.
            journal = Path(str(out)+'.progress.jsonl')
            failed = dict(records[2], prediction=dict(records[2]['prediction'], status='error'))
            with journal.open('a') as stream:
                stream.write(json.dumps(failed)+'\n')
            second = subprocess.run(command+['--resume'], capture_output=True, text=True)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn('"pending": 1', second.stdout)
            self.assertTrue(all(r['prediction']['status']=='dry_run' for r in json.loads(out.read_text())))
            changed = subprocess.run(command+['--resume','--fps','2'], capture_output=True, text=True)
            self.assertNotEqual(changed.returncode, 0)
            scoring = subprocess.run([sys.executable,str(ROOT/'evaluate.py'),'--input',str(data),'--predictions',str(out),'--output-dir',str(Path(d)/'eval')],capture_output=True,text=True)
            self.assertNotEqual(scoring.returncode, 0)

    def test_official_query_prefix_keeps_original_frames_without_padding(self):
        import numpy as np
        from PIL import Image
        self.assertEqual(sample_indices(35, 10, 2, 2.5), [0, 5, 10, 15, 20])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i in range(35):
                color = 'red' if i < 25 else 'blue'
                Image.new('RGB', (56, 56), color).save(root / f'{i:03d}.png')
            a = args('--fps', '2')
            frames, stats = load_query_video(root, 2.5, a, {'fps': 10})
            self.assertEqual(stats['decoded_frames'], 5)
            self.assertEqual(stats['padding_frames'], 0)
            self.assertTrue(np.all(frames[..., 0] > 200))
            self.assertTrue(np.all(frames[..., 2] < 30))
            a.max_frames = 4
            with self.assertRaises(ValueError):
                load_query_video(root, 2.5, a, {'fps': 10})


if __name__ == '__main__':
    unittest.main()
