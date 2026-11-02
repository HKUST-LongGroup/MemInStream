#!/usr/bin/env python3
"""Score HERMES output with the existing Qwen or FluxMem scoring functions."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from predict import digest
from qwen_eval_utils import atomic_write, read_rows
import score_predictions as qwen
import fluxmem_scoring as flux


def join_predictions(rows, predictions):
    joined = [dict(r, prediction={'status': 'missing'}) for r in rows]
    seen = set()
    for entry in predictions:
        pred = entry.get('prediction', {})
        index = pred.get('row_index')
        if type(index) is not int or not 0 <= index < len(rows) or index in seen:
            raise ValueError('Invalid or duplicate prediction.row_index')
        seen.add(index)
        if pred.get('status') == 'dry_run':
            raise ValueError('dry_run has no model answers; cannot compute accuracy')
        if {k: v for k, v in entry.items() if k != 'prediction'} != {k: v for k, v in rows[index].items() if k != 'prediction'}:
            raise ValueError(f'Prediction/input mismatch at row {index}')
        if pred.get('status') == 'completed' and not isinstance(pred.get('answer'), str):
            raise ValueError(f'Completed row {index} has no string answer')
        joined[index] = entry
    return joined


def evaluate_rows(rows, policy='qwen', open_mode='skip', reviews=None):
    reviews = reviews or {}
    if policy == 'qwen':
        report, details = qwen.summarize(rows)
        for field in ['source_dataset', 'question_type', 'original_task', 'level1', 'level2']:
            groups = defaultdict(list)
            for row in rows:
                value = row.get(field, 'unknown')
                groups[json.dumps(value, ensure_ascii=False) if isinstance(value, list) else str(value)].append(row)
            report['by_' + field] = {key: qwen.summarize(value)[0]['overall'] for key, value in sorted(groups.items())}
        return report, details
    details = []
    for i, row in enumerate(rows):
        pred = row['prediction']
        complete = pred.get('status') == 'completed'
        item = {k: row.get(k) for k in ['source_dataset', 'source_id', 'question_type', 'original_task', 'level1', 'level2']}
        item.update(input_index=i, objective=flux.objective(row), correct=None,
                    status='completed' if complete else 'missing_prediction')
        if complete:
            item.update(response=pred['answer'])
            item.update(flux.score(row, pred['answer'], open_mode, reviews.get(i)))
        details.append(item)
    report = dict(policy='fluxmem_prediction-1016/evaluate.py scoring functions', open_ended=open_mode,
                  overall=flux.stats(details), objective=flux.stats([r for r in details if r['objective']]))
    for field in ['source_dataset', 'question_type', 'original_task', 'level1', 'level2']:
        report['by_' + field] = flux.grouped(details, field)
    return report, details


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path, help='original benchmark; all rows form the denominator')
    p.add_argument('--predictions', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--policy', choices=['qwen', 'fluxmem'], default='qwen')
    p.add_argument('--open-ended', choices=['skip', 'exact'], default='skip')
    p.add_argument('--open-scores', type=Path, help='FluxMem policy: reviewed input_index/response/correct JSONL')
    args = p.parse_args(argv)
    outputs = [args.output_dir / name for name in ['report.json', 'details.jsonl', 'open_review.jsonl']]
    inputs = [args.input, args.predictions] + ([args.open_scores] if args.open_scores else [])
    if any(a.resolve() == b.resolve() for a in outputs for b in inputs):
        p.error('output paths must not overwrite inputs')
    if args.policy == 'qwen' and (args.open_scores or args.open_ended != 'skip'):
        p.error('Qwen policy excludes open questions; use --policy fluxmem for open scoring')
    manifest = Path(str(args.predictions) + '.manifest.json')
    if manifest.exists() and json.loads(manifest.read_text()).get('input_sha256') != digest(args.input):
        p.error('input differs from prediction manifest')
    rows = join_predictions(read_rows(args.input), read_rows(args.predictions))
    reviews = {}
    if args.open_scores:
        for entry in read_rows(args.open_scores):
            i = entry.get('input_index')
            if type(i) is not int or not 0 <= i < len(rows) or i in reviews or type(entry.get('correct')) is not bool:
                p.error('invalid/duplicate semantic review')
            pred = rows[i]['prediction']
            if flux.objective(rows[i]) or pred.get('status') != 'completed' or entry.get('response') != pred.get('answer'):
                p.error('semantic review must match a completed open-answer response')
            reviews[i] = entry['correct']
    report, details = evaluate_rows(rows, args.policy, args.open_ended, reviews)
    report['rates'] = '0..1; full input dataset denominator. Qwen: failures/unparseable/invalid labels count as wrong. FluxMem: accuracy_scored excludes missing/unscored.'
    report['inference_completed'] = sum(r['prediction'].get('status') == 'completed' for r in rows)
    report['input_rows'] = len(rows)
    review = [dict(input_index=i, question=r.get('question'), reference=r.get('answer'),
                   response=r['prediction'].get('answer'), correct=reviews.get(i)) for i, r in enumerate(rows)
              if (not flux.objective(r) if args.policy == 'fluxmem' else qwen.kind(r) == 'open_ended_or_other')
              and r['prediction'].get('status') == 'completed']
    atomic_write(outputs[0], report)
    atomic_write(outputs[1], details, jsonl=True)
    atomic_write(outputs[2], review, jsonl=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
