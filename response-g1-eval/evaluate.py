#!/usr/bin/env python3
"""Score unified Response-G1 outputs with the existing qwen-eval policy."""
import argparse
from pathlib import Path
import sys
from core import HERE
sys.path.insert(0, str(HERE / 'scoring_reference'))
from qwen_eval_utils import atomic_write, read_rows
from score_predictions import summarize


def evaluate(rows):
    latest = {}
    for row in rows:
        pred = row.get('response_g1_prediction')
        if not isinstance(pred, dict):
            raise ValueError('Missing response_g1_prediction; refusing to score old prediction fields')
        latest[pred['input_index']] = dict(row, prediction=pred)
    return summarize([latest[i] for i in sorted(latest)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output-dir', required=True)
    a = p.parse_args()
    out = Path(a.output_dir)
    if Path(a.input).resolve() in {(out/'report.json').resolve(), (out/'details.jsonl').resolve()}:
        p.error('Report paths must differ from predictions')
    report, details = evaluate(read_rows(a.input))
    atomic_write(out/'report.json', report)
    atomic_write(out/'details.jsonl', details, jsonl=True)
    print(__import__('json').dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
