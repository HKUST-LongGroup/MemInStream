#!/usr/bin/env python3
"""Analyze FluxMem correct/wrong cohorts using the bundled qwen-eval policy."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from analysis_reference.analyze_evidence import analyze_file, features, write_csv
from analysis_reference.qwen_eval_utils import atomic_write, read_rows
from analysis_reference.score_predictions import summarize


def align_rows(input_path, prediction_path):
    rows = read_rows(input_path)
    predictions = {}
    manifest_path = Path(str(prediction_path) + '.manifest.json')
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        expected = manifest.get('input_sha256')
        if expected and hashlib.sha256(Path(input_path).read_bytes()).hexdigest() != expected:
            raise ValueError('Input differs from the prediction run manifest')
    for record in read_rows(prediction_path):
        pred = record.get('fluxmem_prediction')
        if not isinstance(pred, dict):
            raise ValueError('Expected fluxmem_prediction, not another model prediction field')
        index = pred.get('input_index')
        if type(index) is not int or not 0 <= index < len(rows):
            raise ValueError(f'Invalid input_index: {index}')
        if index in predictions:
            raise ValueError(f'Duplicate input_index: {index}; use one merged run')
        for key in ('source_dataset', 'source_id', 'question', 'options', 'answer',
                    'correct_option', 'correct_index', 'query_time', 'evidence'):
            if record.get(key) != rows[index].get(key):
                raise ValueError(f'Input/prediction mismatch: index={index}, field={key}')
        if pred.get('status') != 'completed' or not isinstance(pred.get('response'), str):
            raise ValueError(f'Expected a completed FluxMem response at index {index}')
        predictions[index] = pred
    aligned, originals = [], []
    for index, row in enumerate(rows):
        pred = predictions.get(index)
        original = dict(row)
        # Remove an old FluxMem result if it happened to be in the input dataset.
        original.pop('fluxmem_prediction', None)
        if pred:
            original['fluxmem_prediction'] = pred
        originals.append(original)
        mapped = dict(original)
        # Always replace the legacy model answer, including when FluxMem is missing.
        mapped['prediction'] = dict(status='completed', answer=pred['response'],
                                    window_start=pred.get('video_start'),
                                    window_end=pred.get('video_end')) if pred else dict(status='missing')
        aligned.append(mapped)
    return aligned, originals


def run(input_path, prediction_path, output, plots=True):
    aligned, originals = align_rows(input_path, prediction_path)
    # qwen-eval's scoring, evidence definitions, bins and plotting code are reused.
    brief = analyze_file(Path(prediction_path), output, plots=plots, rows=aligned)
    items = [features(row, i) for i, row in enumerate(aligned)]
    cohorts = defaultdict(list)
    enriched = []
    for item, row, mapped in zip(items, originals, aligned):
        cohort = 'correct' if item['correct'] else 'wrong' if item['eligible'] else 'unscored'
        export = dict(row, fluxmem_analysis=dict(item, cohort=cohort))
        cohorts[cohort].append(export)
        enriched.append(dict(item, question=row.get('question'), reference_answer=row.get('answer'),
                             options=json.dumps(row.get('options'), ensure_ascii=False),
                             correct_option=json.dumps(row.get('correct_option'), ensure_ascii=False),
                             model_response=mapped['prediction'].get('answer'),
                             video_path=row.get('video_path'),
                             prediction_status=mapped['prediction']['status']))
    csv_fields = list(dict.fromkeys(key for row in enriched for key in row)) or ['row_index']
    for cohort in ('correct', 'wrong', 'unscored'):
        atomic_write(output / f'{cohort}_samples.jsonl', cohorts[cohort], jsonl=True)
        selected = [row for row in enriched if
                    ('correct' if row['correct'] else 'wrong' if row['eligible'] else 'unscored') == cohort]
        write_csv(output / f'{cohort}_samples.csv', selected, csv_fields)
    write_csv(output / 'sample_features.csv', enriched)
    report_path = output / 'summary.json'
    report = json.loads(report_path.read_text())
    completed = sum(row['prediction']['status'] == 'completed' for row in aligned)
    report.update(dataset_input=str(Path(input_path).resolve()),
                  prediction_field='fluxmem_prediction.response',
                  coverage=dict(total=len(aligned), completed=completed, missing=len(aligned)-completed,
                                rate=completed / len(aligned) if aligned else None),
                  notes=['Missing includes failed and not-yet-run rows; the success file cannot distinguish them.',
                         'Wrong includes failed/missing, unparseable and invalid-reference objective rows.',
                         'Ordinary open questions are unscored, not wrong.',
                         'Window coverage uses recorded video_start/video_end, not retained memory tokens.',
                         'This uses qwen-eval scoring; parsing and denominators differ from the earlier evaluate.py.'])
    by_dataset = defaultdict(list)
    for row in aligned:
        by_dataset[str(row.get('source_dataset'))].append(row)
    report['by_dataset'] = {key: summarize(value)[0] for key, value in sorted(by_dataset.items())}
    atomic_write(report_path, report)
    stats = report['scoring']['overall']
    accuracy = f"{stats['accuracy']:.2%}" if stats['accuracy'] is not None else 'N/A'
    print(f"Accuracy={accuracy}; correct={stats['correct']}; wrong={stats['wrong']}; "
          f"completed={completed}/{len(aligned)}; "
          f"unscored={report['scoring']['excluded_open_ended_or_other']}")
    return brief


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True, help='Original full dataset JSON/JSONL')
    p.add_argument('--predictions', type=Path, required=True, help='Merged FluxMem prediction JSONL')
    p.add_argument('--output-dir', type=Path, default=Path('results/fluxmem_analysis'))
    p.add_argument('--no-plots', action='store_true')
    args = p.parse_args()
    for path in (args.input, args.predictions):
        if path.resolve().is_relative_to(args.output_dir.resolve()):
            p.error('Keep input files outside output-dir to avoid overwriting them')
    try:
        run(args.input, args.predictions, args.output_dir, not args.no_plots)
    except ValueError as exc:
        p.error(str(exc))


if __name__ == '__main__':
    main()
