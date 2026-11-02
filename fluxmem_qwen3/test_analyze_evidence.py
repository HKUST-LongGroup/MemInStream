import json
from pathlib import Path
import tempfile
import unittest

from analyze_evidence import align_rows, run


class AnalysisTests(unittest.TestCase):
    def test_cohorts_missing_old_prediction_and_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [dict(source_dataset='test', source_id=str(i), question='Color?',
                         question_type='multiple-choice', options=['A. Red', 'B. Blue'],
                         answer='Red', correct_option='A', query_time=100, evidence=[[10, 20], [15, 30]],
                         prediction=dict(status='completed', answer='A')) for i in range(5)]
            rows[4].update(question_type='open-ended', options=None, correct_option=None)
            predictions = [dict(rows[i], fluxmem_prediction=dict(input_index=i, status='completed',
                           response=response, video_start=20, video_end=100))
                           for i, response in [(0, 'A'), (1, 'B'), (2, 'unknown'), (4, 'Red')]]
            src, pred = root / 'data.json', root / 'pred.jsonl'
            src.write_text(json.dumps(rows))
            pred.write_text('\n'.join(json.dumps(row) for row in predictions))
            out = root / 'out'
            run(src, pred, out, plots=False)
            summary = json.loads((out / 'summary.json').read_text())
            self.assertEqual(summary['scoring']['overall']['accuracy'], .25)
            self.assertEqual(summary['coverage']['completed'], 4)
            self.assertEqual(summary['wrong_score_results'], dict(incorrect=1, unparseable_prediction=1, prediction_failed_or_missing=1))
            correct = json.loads((out / 'correct_samples.jsonl').read_text())
            self.assertEqual(correct['fluxmem_analysis']['evidence_union_s'], 20)
            self.assertEqual(correct['fluxmem_analysis']['evidence_coverage'], .5)
            self.assertEqual(correct['fluxmem_analysis']['window_visibility'], 'partial')
            self.assertEqual(len((out / 'unscored_samples.jsonl').read_text().splitlines()), 1)
            self.assertEqual(len((out / 'wrong_samples.jsonl').read_text().splitlines()), 3)
            # Export keeps the source record and original legacy prediction intact.
            self.assertEqual(correct['prediction'], rows[0]['prediction'])
            predictions[0]['evidence'] = [[0, 5]]
            pred.write_text('\n'.join(json.dumps(row) for row in predictions))
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                align_rows(src, pred)

    def test_duplicate_predictions_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = dict(source_dataset='test', source_id='1')
            pred = dict(row, fluxmem_prediction=dict(input_index=0, status='completed', response='A'))
            (root / 'input.json').write_text(json.dumps([row]))
            (root / 'pred.jsonl').write_text('\n'.join([json.dumps(pred)] * 2))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                align_rows(root / 'input.json', root / 'pred.jsonl')


if __name__ == '__main__':
    unittest.main()
