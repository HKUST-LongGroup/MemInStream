import tempfile
import csv
import json
import unittest
from pathlib import Path

from analyze_evidence import analyze_file, binned_stats, features
from qwen_eval_utils import atomic_write


def sample(**updates):
    row = dict(source_dataset='SYNTHETIC_TEST', question_type='multiple-choice',
               options=['A. Red', 'B. Blue'], correct_option='A', query_time=100,
               evidence=[[10, 20], [15, 25], [70, 80]],
               prediction=dict(status='completed', answer='A', window_start=60, window_end=100))
    row.update(updates)
    return row


class EvidenceTests(unittest.TestCase):
    def test_union_span_signed_lags_and_actual_window(self):
        result = features(sample(), 0)
        self.assertEqual(result['evidence_union_s'], 25)
        self.assertEqual(result['evidence_span_s'], 70)
        self.assertEqual(result['latest_evidence_lag_s'], 20)
        self.assertEqual(result['oldest_evidence_lag_s'], 90)
        self.assertEqual(result['window_visibility'], 'partial')
        self.assertEqual(result['evidence_coverage'], .4)
        future = features(sample(evidence=[[99, 110]]), 0)
        self.assertEqual(future['latest_evidence_lag_s'], -10)
        self.assertTrue(future['evidence_after_query'])

    def test_bad_missing_point_and_empty_evidence(self):
        for evidence in [None, [], [[3, 2]], [[float('nan'), 20]], 'bad']:
            result = features(sample(evidence=evidence), 0)
            self.assertNotIn('evidence_union_s', result)
        point = features(sample(evidence=[[60, 60]]), 0)
        self.assertEqual(point['evidence_union_s'], 0)
        self.assertIsNone(point['evidence_coverage'])
        self.assertEqual(point['window_visibility'], 'full')
        point = features(sample(evidence=[[100, 100]]), 0)
        self.assertEqual(point['window_visibility'], 'none')
        no_query = features(sample(query_time=None), 0)
        self.assertEqual(no_query['evidence_union_s'], 25)
        self.assertNotIn('latest_evidence_lag_s', no_query)
        stale = features(sample(evidence_annotation_status='error'), 0)
        self.assertNotIn('evidence_union_s', stale)

    def test_bins_include_failures_but_not_unscored_questions(self):
        correct = features(sample(), 0)
        failed = features(sample(prediction={'status': 'error'}), 1)
        unscored = features(sample(question_type='open-ended', options=None), 2)
        stats = binned_stats([correct, failed, unscored], 'evidence_union_s', [0, 25, 30])
        self.assertEqual(stats[2]['total'], 2)
        self.assertEqual(stats[2]['correct'], 1)
        self.assertEqual(stats[2]['accuracy'], .5)
        self.assertIsNone(stats[0]['accuracy'])

    def test_wrong_cohort_includes_failure_types_but_excludes_open_questions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'test.json'
            atomic_write(source, [sample(),
                                  sample(prediction=dict(status='completed', answer='B')),
                                  sample(prediction=dict(status='completed', answer='unknown')),
                                  sample(prediction=dict(status='error')),
                                  sample(question_type='open-ended', options=None)])
            result = analyze_file(source, root / 'out', plots=False)
            self.assertEqual(result['wrong'], 3)
            self.assertEqual(result['correct'], 1)
            self.assertEqual(result['wrong_median_evidence_union_s'], 25)
            with (root / 'out' / 'wrong_samples.csv').open(encoding='utf-8-sig') as stream:
                records = list(csv.DictReader(stream))
            self.assertEqual(len(records), 3)
            self.assertEqual({r['score_result'] for r in records},
                             {'incorrect', 'unparseable_prediction', 'prediction_failed_or_missing'})
            report = json.loads((root / 'out' / 'summary.json').read_text())
            self.assertEqual(sum(report['wrong_score_results'].values()), 3)
            self.assertEqual(report['distributions']['wrong']['evidence_span_s']['n'], 3)

    def test_json_csv_outputs_and_zero_correct(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'test.json'
            atomic_write(source, [sample(prediction={'status': 'error'})])
            result = analyze_file(source, root / 'out', plots=False)
            self.assertEqual(result['correct'], 0)
            self.assertEqual(result['scored'], 1)
            self.assertTrue((root / 'out' / 'correct_samples.csv').is_file())
            self.assertIsNone(result['correct_median_latest_evidence_lag_s'])
            atomic_write(source, [])
            self.assertEqual(analyze_file(source, root / 'empty', plots=False)['scored'], 0)


if __name__ == '__main__':
    unittest.main()
