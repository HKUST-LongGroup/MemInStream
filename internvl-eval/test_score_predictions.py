import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import score_predictions as scoring


def choice(answer='B', **extra):
    return dict(question_type='multiple-choice', options=['A. Red', 'B. Blue'],
                correct_option='B', prediction=dict(status='completed', answer=answer), **extra)


class ScoringTests(unittest.TestCase):
    def test_choice_normalization_and_ambiguous_prose(self):
        for answer in ['b', '(B)', '**B**', 'Answer: B.', 'B. Blue', 'Blue']:
            self.assertEqual(scoring.score_row(choice(answer))[1], 'correct')
        for answer in ['A or B', 'I think B because blue', 'C', '', None]:
            self.assertEqual(scoring.score_row(choice(answer))[1], 'unparseable_prediction')
        self.assertEqual(scoring.score_row(choice('A'))[1], 'incorrect')

    def test_multi_select_exact_set_and_index_fallback(self):
        row = choice('B,A')
        row['correct_option'] = ['A', 'B']
        self.assertEqual(scoring.score_row(row)[1], 'correct')
        row['prediction']['answer'] = 'B'
        self.assertEqual(scoring.score_row(row)[1], 'incorrect')
        row['correct_option'] = None
        row['correct_index'] = 1
        self.assertEqual(scoring.score_row(row)[1], 'correct')
        row['correct_index'] = True
        self.assertEqual(scoring.score_row(row)[1], 'invalid_reference')
        del row['correct_index']
        row['answer'] = 'Blue'
        self.assertEqual(scoring.score_row(row)[1], 'correct')

    def test_binary_aliases_and_false_not_missing(self):
        for gold, answer in [('Yes', 'True'), ('No', 'False'), (False, 'no'), ('True', '是')]:
            row = dict(question_type='binary', answer=gold,
                       prediction=dict(status='completed', answer=answer))
            self.assertEqual(scoring.score_row(row)[1], 'correct')
        self.assertIsNone(scoring.parse_binary('Yes and no'))

    def test_denominators_failures_exclusion_and_empty(self):
        rows = [choice(), choice('A'), choice('uncertain'),
                dict(question_type='binary', answer='No', prediction={'status': 'error'}),
                dict(question_type='binary', answer='True', prediction={'status': 'completed', 'answer': 'yes'}),
                dict(question_type='binary', answer='invalid'),
                dict(question_type='open-ended', answer='free text')]
        report, details = scoring.summarize(rows)
        self.assertEqual(report['input_rows'], 7)
        self.assertEqual(report['excluded_open_ended_or_other'], 1)
        self.assertEqual(report['overall']['total'], 6)
        self.assertEqual(report['overall']['correct'], 2)
        self.assertEqual(report['overall']['wrong'], 4)
        self.assertEqual(report['overall']['accuracy'], 2/6)
        self.assertEqual(len(details), 7)
        self.assertIsNone(scoring.summarize([])[0]['overall']['accuracy'])
        with self.assertRaisesRegex(ValueError, 'dry_run'):
            scoring.summarize([dict(prediction={'status': 'dry_run'})])

    def test_count_parsing_exact_and_invalid_numbers(self):
        for answer in [3, '3', '03', '**3**', 'Answer: 3.', '3 times', '3次', '３']:
            self.assertEqual(scoring.parse_count(answer), 3)
        self.assertEqual(scoring.parse_count('0'), 0)
        for answer in [True, False, -1, '-1', 3.5, '3.5', '.5', '3 or 4', 'There are 3 and 4', '', None]:
            self.assertIsNone(scoring.parse_count(answer), repr(answer))

    def test_rec_counts_in_totals_and_other_open_questions_stay_excluded(self):
        def count(answer, status='completed'):
            return dict(question_type='open-ended', original_task='Repetition Event Count',
                        answer='3', prediction=dict(status=status, answer=answer))
        rows = [choice(), count('3 times'), count('4'), count('unknown'), count('', 'error'),
                dict(question_type='open-ended', answer='3')]
        report, details = scoring.summarize(rows)
        self.assertEqual(report['counting']['total'], 4)
        self.assertEqual(report['counting']['correct'], 1)
        self.assertEqual(report['counting']['wrong'], 3)
        self.assertEqual(report['overall']['total'], 5)
        self.assertEqual(report['overall']['correct'], 2)
        self.assertEqual(report['excluded_open_ended_or_other'], 1)
        self.assertEqual(details[1]['category'], 'counting')

    def test_cli_json_and_jsonl_reports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for suffix in ['.json', '.jsonl']:
                source, report, details = root / ('input'+suffix), root / 'report.json', root / 'details.jsonl'
                source.write_text(json.dumps([choice()]) if suffix == '.json' else json.dumps(choice())+'\n')
                run = subprocess.run([sys.executable, scoring.__file__, '--input', str(source),
                                      '--report', str(report), '--details', str(details)],
                                     capture_output=True, text=True, check=True)
                self.assertIn('总题数 1，答对 1，答错 0，准确率 100.00%', run.stdout)
                self.assertEqual(json.loads(report.read_text())['overall']['correct'], 1)
                self.assertEqual(json.loads(details.read_text())['result'], 'correct')


if __name__ == '__main__':
    unittest.main()
