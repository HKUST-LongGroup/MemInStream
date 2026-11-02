import json
import os
import shutil
import subprocess
import tempfile
import time
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import unittest
from pathlib import Path

import predict_streaming as predict


def out_of_order_worker(connection, args, rank):
    connection.send(('ready', None))
    while True:
        task = connection.recv()
        if task is None:
            break
        index, row = task
        if index == 0:
            time.sleep(0.5)
        connection.send(('result', predict.predict_row(index, row, args, None, rank)))
    connection.close()


def fatal_worker(connection, args, rank):
    connection.send(('fatal', 'checkpoint initialization failed'))
    connection.close()


def crashed_worker(connection, args, rank):
    os._exit(7)


class PredictionTests(unittest.TestCase):
    def args(self, *extra):
        return predict.parse_args(['--input', 'in.json', '--output', 'out.json',
                                   '--model', 'internvl35', '--model-path', '/weights', *extra])

    def test_query_anchored_windows_and_empty_windows(self):
        self.assertEqual(predict.window({'query_time': 497}, 60), (437, 497))
        self.assertEqual(predict.window({'query_time': 20}, 60), (0, 20))
        self.assertEqual(predict.window({'query_time': '01:20'}, 0), (0, 80))
        self.assertEqual(predict.window({'query_time': 80}, 60, 70), (20, 70))
        with self.assertRaises(ValueError):
            predict.window({'query_time': 80}, 10, 60)
        with self.assertRaises(ValueError):
            predict.window({'query_time': 0}, 60)

    def test_prompt_excludes_ground_truth_and_annotations(self):
        row = dict(question='What color?', query_time=10, options=['A. Red', 'B. Blue'],
                   answer='SECRET_GOLD', evidence='SECRET_EVIDENCE', correct_option='B',
                   level1_model_outputs='SECRET_MODEL')
        prompt = predict.make_prompt(row, 0, 10)
        self.assertNotIn('SECRET', prompt)
        self.assertIn('A. Red', prompt)
        self.assertNotIn('A. A.', prompt)
        row['correct_option'] = 'A'
        self.assertEqual(prompt, predict.make_prompt(row, 0, 10))

    def test_all_choice_questions_allow_multiple_answers(self):
        for metadata in [{}, {'QA_type': 'mc_multi'}, {'question_type': 'multiple-select'},
                         {'question_type': 'multiple-choice'}]:
            row = dict(question='Which colors?', query_time=10, options=['Red', 'Blue'], **metadata)
            prompt = predict.make_prompt(row, 0, 10)
            self.assertIn('Select all correct options', prompt)
            self.assertIn('Do not select additional options merely because you are uncertain.', prompt)
            self.assertNotIn('only one option', prompt)

    def test_contextual_raw_fields_and_prior_answers(self):
        row = {
            'Question': 'Where is the utensil identified in my second question?',
            'Question time': 332, 'Options': ['A. Left', 'B. Center'],
            'Answer': 'SECRET_CURRENT_ANSWER', 'correct_option': 'B', 'correct_index': 1,
            'required_context_event_ids': ['q2', 'a2'], 'video_path': 'missing.mp4',
            'dialogue_events': [
                {'event_id': 'q1', 'time': 14, 'role': 'user', 'content': 'What color?'},
                {'event_id': 'a1', 'time': 18, 'role': 'assistant', 'content': 'Red.'},
                {'event_id': 'q2', 'time': 26, 'role': 'user', 'content': 'What tool?'},
                {'event_id': 'a2', 'time': 30, 'role': 'assistant', 'content': 'A rotary can opener.'},
                {'event_id': 'a3', 'time': 332, 'role': 'assistant', 'content': 'SECRET_AT_QUERY'},
                {'event_id': 'a4', 'time': 340, 'role': 'assistant', 'content': 'SECRET_FUTURE'},
            ],
        }
        result = predict.predict_row(0, row, self.args('--dry-run', '--history-seconds', '4'), None)
        self.assertEqual(result['prediction']['status'], 'dry_run')
        self.assertEqual(result['prediction']['window_start'], 328)
        prompt = result['prediction']['prompt']
        self.assertIn('a1 [18.000s] assistant: Red.', prompt)
        self.assertIn('a2 [30.000s] assistant: A rotary can opener.', prompt)
        self.assertLess(prompt.index('q1 ['), prompt.index('q2 ['))
        self.assertLess(prompt.index('a2 ['), prompt.index('Question:'))
        self.assertIn('A. Left\nB. Center', prompt)
        self.assertNotIn('SECRET', prompt)
        row.update(Answer='CHANGED_GOLD', correct_option='A', correct_index=0)
        self.assertEqual(prompt, predict.make_prompt(row, 328, 332))

    def test_normalized_dialogue_history_rounds(self):
        row = dict(question='Which tool?', query_time=50, dialogue_history=[
            {'timestamp': 20, 'question': 'Which color?', 'options': ['Red', 'Blue'], 'answer': 'B'},
            {'timestamp': 10, 'role': 'user', 'content': 'Remember this tool.'},
        ])
        prompt = predict.make_prompt(row, 40, 50)
        self.assertLess(prompt.index('[10.000s]'), prompt.index('[20.000s]'))
        self.assertIn('B. Blue', prompt)
        self.assertIn('assistant: B', prompt)

    def test_budget_does_not_reduce_fps(self):
        with self.assertRaisesRegex(ValueError, 'budget exceeded'):
            predict.decode_video('missing', 0, 100, (32, 32, 100), self.args('--max-frames', '2'))

    def test_prefix_boundary(self):
        args = self.args('--path-prefix', '/old', '/new')
        self.assertEqual(str(predict.resolve_video('/old/a.mp4', args)), '/new/a.mp4')
        self.assertEqual(str(predict.resolve_video('/older/a.mp4', args)), '/older/a.mp4')

    def test_cli_preserves_input_rows_and_reports_bad_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'input.json'
            rows = [dict(source_id='same', question='What?', query_time=t, video_path='missing.mp4',
                         options=None, answer='SECRET', keep=False) for t in [20, 0]]
            source.write_text(json.dumps(rows))
            for model in ['internvl35', 'llava_onevision']:
                output = root / f'{model}.json'
                argv = ['--input', str(source), '--output', str(output), '--model', model,
                        '--model-path', '/missing', '--history-seconds', '5', '--fps', '0.5', '--dry-run']
                self.assertEqual(predict.main(argv), 1)
                results = json.loads(output.read_text())
                self.assertEqual(len(results), 2)
                self.assertFalse(results[0]['keep'])
                self.assertEqual(results[0]['answer'], 'SECRET')
                self.assertEqual(results[0]['prediction']['window_start'], 15)
                self.assertEqual(results[1]['prediction']['status'], 'error')
                with self.assertRaises(FileExistsError):
                    predict.main(argv)

    def test_parallel_scheduler_no_duplicates_and_completion_order(self):
        args = self.args('--num-gpus', '2', '--dry-run')
        rows = [dict(source_id='duplicate-id', question='What?', query_time=20,
                     video_path='missing.mp4') for _ in range(8)]
        results = list(predict.parallel_results(rows, args, worker_target=out_of_order_worker))
        indices = [r['prediction']['row_index'] for r in results]
        self.assertEqual(sorted(indices), list(range(8)))
        self.assertNotEqual(indices[0], 0)
        self.assertEqual({r['prediction']['worker_index'] for r in results}, {0, 1})

    def test_parallel_worker_fatal_and_abrupt_exit(self):
        args = self.args('--num-gpus', '2', '--dry-run')
        for target, message in [(fatal_worker, 'checkpoint initialization failed'),
                                (crashed_worker, 'exited')]:
            with self.subTest(target=target.__name__):
                with self.assertRaisesRegex(RuntimeError, message):
                    list(predict.parallel_results([{}, {}], args, worker_target=target))

    def test_parallel_cli_both_models_and_output_formats(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'input.json'
            rows = [dict(source_id='same', question=f'Question {i}', query_time=20 if i != 3 else 0,
                         video_path='missing.mp4', answer='SECRET') for i in range(9)]
            source.write_text(json.dumps(rows))
            for model, suffix in [('internvl35', '.json'), ('llava_onevision', '.jsonl')]:
                output = root / (model + suffix)
                run = subprocess.run([sys.executable, str(Path(predict.__file__)),
                                      '--input', str(source), '--output', str(output),
                                      '--model', model, '--model-path', '/missing',
                                      '--num-gpus', '2', '--dry-run'], capture_output=True, text=True, timeout=30)
                self.assertEqual(run.returncode, 1, run.stderr)
                results = predict.read_rows(output)
                self.assertEqual([r['question'] for r in results], [r['question'] for r in rows])
                self.assertEqual([r['prediction']['row_index'] for r in results], list(range(9)))
                self.assertEqual(results[3]['prediction']['status'], 'error')
                self.assertEqual(len(predict.read_rows(output.with_suffix('.progress.jsonl'))), 9)

    def test_gpu_count_and_parallel_arguments(self):
        args = self.args('--num-gpus', '4')
        with patch.dict(sys.modules, torch=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2))):
            with self.assertRaisesRegex(ValueError, 'only 2 CUDA GPUs'):
                predict.validate_gpu_count(args)
        with self.assertRaises(SystemExit):
            self.args('--num-gpus', '0')
        with self.assertRaises(SystemExit):
            self.args('--num-gpus', '2', '--device-map', 'cuda:0')

    @unittest.skipUnless(os.environ.get('RUN_VIDEO_INTEGRATION') == '1' and shutil.which('ffmpeg') and shutil.which('ffprobe'), 'set RUN_VIDEO_INTEGRATION=1 with working ffmpeg/ffprobe')
    def test_real_decode_excludes_future_frames_without_padding(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'colors.mkv'
            # Blue starts exactly at the query boundary and must never be supplied.
            import numpy as np
            source_frames = np.zeros((40, 64, 64, 3), dtype=np.uint8)
            source_frames[:20, :, :, 0] = 255
            source_frames[20:, :, :, 2] = 255
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'rawvideo',
                            '-pix_fmt', 'rgb24', '-s', '64x64', '-r', '10', '-i', 'pipe:0',
                            '-threads', '1', '-c:v', 'ffv1', str(path)],
                           input=source_frames.tobytes(), check=True, capture_output=True, timeout=30)
            metadata = predict.probe_video(str(path), 30)
            for model in ['internvl35', 'llava_onevision']:
                args = self.args('--model', model, '--fps', '2', '--history-seconds', '0')
                frames, stats = predict.decode_video(path, 0, 2, metadata, args)
                self.assertEqual(stats['decoded_frames'], 4)
                self.assertTrue((frames[..., 0] > 200).all())
                self.assertTrue((frames[..., 2] < 20).all())
                frames, stats = predict.decode_video(path, 1, 1.5, metadata, args)
                self.assertEqual(stats['decoded_frames'], 1)
                self.assertEqual(stats['padding_frames'], 0)
                self.assertEqual(len(frames), 1)
            args = self.args('--fps', '0.5', '--history-seconds', '0')
            frames, stats = predict.decode_video(path, 0, 2, metadata, args)
            self.assertEqual(stats['decoded_frames'], 1)
            self.assertTrue((frames[..., 2] < 20).all())

            # Short windows, early queries, and end-of-video rounding all use
            # the requested 4s budget. Never pad with the future blue frames.
            args = self.args('--fps', '1', '--history-seconds', '4')
            for start, end in [(0, 2), (1, 1.5), (0, 0.2)]:
                frames, stats = predict.decode_video(path, start, end, metadata, args)
                self.assertEqual(len(frames), 4)
                self.assertEqual(stats['target_frames'], 4)
                self.assertEqual(stats['padding_frames'], 4 - stats['decoded_frames'])
                self.assertTrue((frames[..., 0] > 200).all())
                self.assertTrue((frames[..., 2] < 20).all())
                self.assertTrue((frames[-1] == frames[stats['decoded_frames'] - 1]).all())

    def test_three_decoded_frames_pad_to_requested_four_seconds(self):
        import numpy as np
        raw = np.arange(3 * 32 * 32 * 3, dtype=np.uint8).reshape(3, 32, 32, 3)
        args = self.args('--history-seconds', '4', '--fps', '1')
        for end in [4, 3.981]:
            with patch.object(predict.subprocess, 'run', return_value=SimpleNamespace(
                    returncode=0, stdout=raw.tobytes(), stderr=b'')):
                frames, stats = predict.decode_video('unused.mp4', 0, end, (32, 32, end), args)
            self.assertEqual(stats['decoded_frames'], 3)
            self.assertEqual(stats['input_frames'], 4)
            self.assertEqual(stats['padding_frames'], 1)
            np.testing.assert_array_equal(frames[:3], raw)
            np.testing.assert_array_equal(frames[3], raw[2])

    def test_requested_padding_budget_checked_before_decoding(self):
        args = self.args('--history-seconds', '4', '--fps', '2', '--max-frames', '4')
        with patch.object(predict.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'budget exceeded'):
                predict.decode_video('unused', 0, 1, (32, 32, 1), args)
            run.assert_not_called()

    def test_empty_video_cannot_be_padded(self):
        args = self.args('--history-seconds', '4')
        with patch.object(predict.subprocess, 'run', return_value=SimpleNamespace(
                returncode=0, stdout=b'', stderr=b'')):
            with self.assertRaisesRegex(ValueError, 'no source frames'):
                predict.decode_video('unused', 0, 0.2, (32, 32, 0.2), args)


if __name__ == '__main__':
    unittest.main()
