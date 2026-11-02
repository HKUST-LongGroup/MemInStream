#!/usr/bin/env python3
"""Full-input coverage scoring using the supplied Qwen scorer without changing its rules."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from qwen_eval_utils import atomic_write, read_rows
from score_predictions import summarize


def align_predictions(rows, predictions):
    merged = [dict(row, prediction={'status': 'missing'}) for row in rows]
    seen = set()
    for record in predictions:
        pred = record.get('flash_vstream_prediction')
        if not isinstance(pred, dict):
            raise ValueError('Missing flash_vstream_prediction; refusing to score an older prediction field')
        index = pred.get('input_index')
        if type(index) is not int or not 0 <= index < len(rows) or index in seen:
            raise ValueError(f'Invalid/duplicate prediction index: {index}')
        seen.add(index)
        for key, value in rows[index].items():
            if key not in ('prediction', 'input_prediction', 'flash_vstream_prediction') and record.get(key) != value:
                raise ValueError(f'Input/prediction mismatch at row {index}, field {key}')
        # Same response, both schemas: do not silently score a conflicting field.
        if pred.get('status') == 'completed' and pred.get('answer') != pred.get('response'):
            raise ValueError(f'Conflicting answer/response at row {index}')
        merged[index]['prediction'] = pred
    for row in merged:
        for alias, key in [('Options', 'options'), ('Question', 'question'), ('Question time', 'query_time')]:
            if key not in row and alias in row:
                row[key] = row[alias]
    return merged


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, help='Original benchmark JSON/JSONL (defines full denominator)')
    p.add_argument('--predictions', required=True)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args(argv)
    dest = Path(args.output_dir)
    targets = [dest / 'report.json', dest / 'details.jsonl']
    if any(t.resolve() == Path(v).resolve() for t in targets for v in (args.input, args.predictions)):
        p.error('Output must not overwrite input/predictions')
    manifest = Path(args.predictions + '.manifest.json')
    if manifest.exists():
        expected = json.loads(manifest.read_text())['input_sha256']
        if hashlib.sha256(Path(args.input).read_bytes()).hexdigest() != expected:
            p.error('Input differs from prediction manifest')
    rows = align_predictions(read_rows(args.input), read_rows(args.predictions))
    report, details = summarize(rows)
    report['coverage'] = dict(total=len(rows), completed=sum(r['prediction']['status'] == 'completed' for r in rows))
    for field in ('source_dataset', 'question_type', 'level1', 'level2'):
        groups = defaultdict(list)
        for row in rows:
            key = row.get(field, 'unknown')
            # Multi-label categories are reported under their original combined value.
            groups[json.dumps(key, ensure_ascii=False) if isinstance(key, (list, dict)) else str(key)].append(row)
        report['by_' + field] = {key: summarize(group)[0]['overall'] for key, group in sorted(groups.items())}
    atomic_write(targets[0], report)
    atomic_write(targets[1], details, jsonl=True)
    print(json.dumps({'overall': report['overall'], 'coverage': report['coverage']}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
