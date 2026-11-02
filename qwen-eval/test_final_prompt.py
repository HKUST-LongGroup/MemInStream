"""Regression checks for the single released prompt and prediction entry point."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from prompt_templates import make_prompt, prompt_version

HERE = Path(__file__).resolve().parent


class FinalPromptTests(unittest.TestCase):
    def sample(self):
        return dict(video_path='YOUR_DATASET/example.mp4', query_time=20,
                    source_dataset='example', source_id='one', question='Which objects?',
                    options=['A. Cup', 'B. Plate'], correct_option='A', answer='SECRET',
                    dialogue_events=[dict(time=2, role='assistant', content='PAST'),
                                     dict(time=20, role='assistant', content='SECRET_AT_QUERY'),
                                     dict(time=21, role='assistant', content='SECRET_FUTURE')])

    def test_final_choice_prompt_does_not_depend_on_labels(self):
        row = self.sample()
        text = make_prompt(row, 0, 20)
        self.assertIn('Select all correct options.', text)
        self.assertIn('Do not select additional options merely because you are uncertain.', text)
        self.assertIn('exactly match all correct options', text)
        self.assertIn('PAST', text)
        self.assertNotIn('SECRET', text)
        changed = dict(row, answer='OTHER', correct_option=['A', 'B'],
                       level2='Occurrence Verification', question_type='multiple-select')
        self.assertEqual(text, make_prompt(changed, 0, 20))
        self.assertEqual(prompt_version(), 'aligned_prompt_v2_20261020')

    def test_binary_and_open_answers(self):
        row = dict(self.sample(), options=None, question_type='binary')
        self.assertTrue(make_prompt(row, 0, 20).endswith('Return only True or False.'))
        row['question_type'] = 'open-ended'
        self.assertTrue(make_prompt(row, 0, 20).endswith('Return only the answer.'))

    def test_public_entry_point_uses_final_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / 'input.json', root / 'prediction.jsonl'
            source.write_text(json.dumps([self.sample()]))
            command = [sys.executable, '-B', str(HERE / 'predict.py'), '--input', str(source),
                       '--output', str(output), '--model-path', 'YOUR_MODEL', '--dry-run']
            if HERE.name in ('qwen-eval', 'hermes-eval'):
                command += ['--model', 'qwen25']
            elif HERE.name == 'internvl-eval':
                command += ['--model', 'internvl35']
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            if HERE.name.startswith('fluxmem_'):
                pred = json.loads(Path(str(output) + '.preview.jsonl').read_text().splitlines()[0])
            else:
                row = json.loads(output.read_text().splitlines()[0])
                key = 'response_g1_prediction' if HERE.name == 'response-g1-eval' else 'prediction'
                pred = row[key]
            self.assertEqual(pred['prompt_version'], prompt_version())
            self.assertIn('Select all correct options.', pred['prompt'])
            self.assertNotIn('SECRET', pred['prompt'])
            rejected = subprocess.run(command + ['--prompt-version', 'v2'],
                                      capture_output=True, text=True, timeout=30)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('unrecognized arguments', rejected.stderr)


if __name__ == '__main__':
    unittest.main()
