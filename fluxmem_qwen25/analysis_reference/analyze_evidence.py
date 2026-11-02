#!/usr/bin/env python3
"""Analyze scored predictions by evidence duration, temporal distance and input coverage."""
import argparse
import csv
import glob
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median

from .qwen_eval_utils import atomic_write, read_rows, seconds
from .score_predictions import score_row, summarize

METRICS = {
    'evidence_union_s': ('Evidence union duration (s)', [0, 2, 5, 10, 20, 30, 60, 120, 300]),
    'evidence_span_s': ('Evidence span (s)', [0, 2, 5, 10, 20, 30, 60, 120, 300]),
    'latest_evidence_lag_s': ('Query minus latest evidence end (s)', [0, 5, 10, 30, 60, 120, 300, 600]),
    'oldest_evidence_lag_s': ('Query minus earliest evidence start (s)', [0, 5, 10, 30, 60, 120, 300, 600]),
}


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def features(row, index):
    category, result = score_row(row)
    item = dict(row_index=index, source_dataset=row.get('source_dataset'), source_id=row.get('source_id'),
                question_type=category, original_task=row.get('original_task'), score_result=result,
                eligible=result != 'excluded', correct=result == 'correct', evidence_state='missing',
                window_visibility='unknown')
    try:
        item['query_time_s'] = seconds(row.get('query_time'))
    except (ValueError, TypeError):
        item['query_time_s'] = None
    raw = row.get('evidence')
    if row.get('evidence_annotation_status') not in (None, 'completed'):
        item['evidence_state'] = 'annotation_not_completed'
        return item
    if raw is None or raw == []:
        return item
    if not isinstance(raw, list):
        item['evidence_state'] = 'invalid'
        return item
    intervals = []
    try:
        for pair in raw:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError('expected [start, end]')
            start, end = map(seconds, pair)
            if end < start:
                raise ValueError('reversed interval')
            intervals.append((start, end))
    except (ValueError, TypeError):
        item['evidence_state'] = 'invalid'
        return item
    if row.get('evidence_localization_status') == 'not_found':
        item['evidence_state'] = 'inconsistent_not_found'
        return item
    merged = merge_intervals(intervals)
    union = sum(end - start for start, end in merged)
    item.update(evidence_state='valid', evidence_interval_count=len(intervals),
                merged_interval_count=len(merged), evidence_union_s=union,
                evidence_span_s=merged[-1][1] - merged[0][0],
                mean_interval_s=mean(end - start for start, end in intervals))
    if item['query_time_s'] is not None:
        item.update(latest_evidence_lag_s=item['query_time_s'] - merged[-1][1],
                    oldest_evidence_lag_s=item['query_time_s'] - merged[0][0],
                    evidence_after_query=merged[-1][1] > item['query_time_s'])
    pred = row.get('prediction') or {}
    try:
        start, end = seconds(pred.get('window_start')), seconds(pred.get('window_end'))
        if start >= end:
            raise ValueError('empty observation window')
    except (AttributeError, ValueError, TypeError):
        return item
    overlap = sum(max(0, min(b, end) - max(a, start)) for a, b in merged)
    full = all(a >= start and b <= end and a < end for a, b in merged)
    any_visible = overlap > 0 or any(a == b and start <= a < end for a, b in merged)
    item.update(window_start_s=start, window_end_s=end, visible_evidence_s=overlap,
                evidence_coverage=overlap / union if union > 0 else None,
                window_visibility='full' if full else 'partial' if any_visible else 'none')
    return item


def distribution(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return dict(n=0, mean=None, median=None, min=None, max=None)
    return dict(n=len(values), mean=mean(values), median=median(values), min=values[0], max=values[-1])


def bucket_index(value, edges):
    if value < 0:
        return 0
    for i, end in enumerate(edges[1:], 1):
        if value < end:
            return i
    return len(edges)


def binned_stats(items, metric, edges):
    labels = ['<0'] + [f'[{a:g},{b:g})' for a, b in zip(edges, edges[1:])] + [f'[{edges[-1]:g},inf)']
    stats = [dict(metric=metric, bin=label, total=0, correct=0) for label in labels]
    for item in items:
        value = item.get(metric)
        if not item['eligible'] or value is None:
            continue
        group = stats[bucket_index(value, edges)]
        group['total'] += 1
        group['correct'] += item['correct']
    for group in stats:
        group['accuracy'] = group['correct'] / group['total'] if group['total'] else None
    return stats


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(key for row in rows for key in row)) or ['row_index']
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_run(items, bins, output, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .18, 'pdf.fonttype': 42})
    correct = [item for item in items if item['correct']]
    wrong = [item for item in items if item['eligible'] and not item['correct']]
    cohorts = [('correct', correct, 'Correct', '#0072B2'),
               ('wrong', wrong, 'Wrong / failed', '#D55E00')]

    def save(fig, name):
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, .96))
        for suffix in ('png', 'pdf'):
            fig.savefig(output / f'{name}.{suffix}', dpi=300, bbox_inches='tight')
        plt.close(fig)

    for name, cohort, cohort_label, color in cohorts:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, (metric, (label, _)) in zip(axes.flat, METRICS.items()):
            values = [r[metric] for r in cohort if r.get(metric) is not None]
            pooled = [r[metric] for r in correct + wrong if r.get(metric) is not None]
            if values:
                # Both cohorts use the same edges so their histograms are comparable.
                edges = np.histogram_bin_edges(pooled, bins=min(25, max(1, int(math.sqrt(len(pooled))))))
                ax.hist(values, bins=edges, color=color, edgecolor='white')
                ax.axvline(median(values), color='#333333', linestyle='--', label=f'Median={median(values):.1f}s')
                ax.legend()
            else:
                ax.text(.5, .5, 'No samples with valid evidence', ha='center', transform=ax.transAxes)
            ax.set(xlabel=label, ylabel=f'{cohort_label} questions', title=f'n={len(values)}')
        save(fig, f'{name}_distributions')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (metric, (label, _)) in zip(axes.flat, METRICS.items()):
        plotted = False
        for _, cohort, cohort_label, color in cohorts:
            values = sorted(r[metric] for r in cohort if r.get(metric) is not None)
            if values:
                # Collapse ties so the ECDF includes all observations at a value.
                counts = Counter(values)
                x = sorted(counts)
                cumulative = np.cumsum([counts[v] for v in x]) / len(values)
                ax.step(x, cumulative, where='post', color=color,
                        label=f'{cohort_label} (n={len(values)})')
                ax.scatter(x, cumulative, color=color, s=8)
                plotted = True
        if plotted:
            ax.legend()
        else:
            ax.text(.5, .5, 'No valid samples', ha='center', transform=ax.transAxes)
        ax.set_xscale('symlog', linthresh=1)
        ax.set(xlabel=label + ' (symlog)', ylabel='Fraction within cohort (ECDF)', ylim=(0, 1.05))
    save(fig, 'correct_vs_wrong_ecdf')

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (metric, (label, _)) in zip(axes.flat, METRICS.items()):
        groups = [r for r in bins if r['metric'] == metric]
        for i, group in enumerate(groups):
            if group['total']:
                rate = 100 * group['accuracy']
                ax.bar(i, rate, color='#009E73', width=.75)
                ax.text(i, rate + 2, f'{group["correct"]}/{group["total"]}', ha='center', fontsize=8)
            else:
                ax.text(i, 2, 'N/A', ha='center', fontsize=7, color='gray')
        ax.set_xticks(range(len(groups)), [g['bin'] for g in groups], rotation=45, ha='right')
        ax.set(xlabel=label, ylabel='Accuracy (%)', ylim=(0, 115))
    save(fig, 'accuracy_by_evidence')

    for name, cohort, cohort_label, color in cohorts:
        fig, ax = plt.subplots(figsize=(8, 5))
        points = [r for r in cohort if r.get('latest_evidence_lag_s') is not None]
        if points:
            ax.scatter([r['evidence_union_s'] for r in points], [r['latest_evidence_lag_s'] for r in points],
                       s=20, alpha=.5, color=color, edgecolors='none')
        else:
            ax.text(.5, .5, 'No valid samples', ha='center', transform=ax.transAxes)
        ax.set_xscale('symlog', linthresh=1)
        ax.set_yscale('symlog', linthresh=1)
        ax.axhline(0, color='gray', linestyle='--', linewidth=.8)
        ax.set(xlabel='Evidence union duration (s; symlog)',
               ylabel='Query minus latest evidence end (s; symlog)', title=f'{cohort_label} questions: n={len(points)}')
        save(fig, f'{name}_duration_vs_lag')

    fig, ax = plt.subplots(figsize=(8, 5))
    states = ['full', 'partial', 'none', 'unknown']
    for i, state in enumerate(states):
        rows = [r for r in items if r['eligible'] and r['window_visibility'] == state]
        n, k = len(rows), sum(r['correct'] for r in rows)
        ax.bar(i, k, color='#0072B2', label='Correct' if i == 0 else None)
        ax.bar(i, n-k, bottom=k, color='#E69F00', label='Wrong / failed' if i == 0 else None)
        ax.text(i, n, f'{k}/{n}', va='bottom', ha='center')
    ax.set_xticks(range(len(states)), states)
    ax.set(xlabel='Annotated evidence inside supplied video window', ylabel='Scored questions')
    ax.margins(y=.15)
    ax.legend()
    save(fig, 'window_coverage')


def analyze_file(path, output, plots=True, rows=None):
    rows = read_rows(path) if rows is None else rows
    accuracy, _ = summarize(rows)  # Reject dry-run records and use the same scoring policy.
    items = [features(row, i) for i, row in enumerate(rows)]
    eligible = [r for r in items if r['eligible']]
    correct = [r for r in items if r['correct']]
    wrong = [r for r in eligible if not r['correct']]
    bins = [stat for metric, (_, edges) in METRICS.items() for stat in binned_stats(items, metric, edges)]
    stats = {key: {metric: distribution([r.get(metric) for r in subset]) for metric in METRICS}
             for key, subset in [('correct', correct), ('wrong', wrong) , ('all_scored', eligible)]}
    report = dict(input=str(path.resolve()), scoring=accuracy, distributions=stats,
                  evidence_states_scored=dict(Counter(r['evidence_state'] for r in eligible)),
                  wrong_score_results=dict(Counter(r['score_result'] for r in wrong)),
                  wrong_window_visibility=dict(Counter(r['window_visibility'] for r in wrong)),
                  wrong_evidence_states=dict(Counter(r['evidence_state'] for r in wrong)),
                  correct_window_visibility=dict(Counter(r['window_visibility'] for r in correct)),
                  scored_evidence_after_query=sum(r.get('evidence_after_query', False) for r in eligible),
                  metric_definitions={
                      'evidence_union_s': 'Total duration after merging overlapping intervals, no double counting.',
                      'evidence_span_s': 'Latest evidence end minus earliest evidence start, including gaps.',
                      'latest_evidence_lag_s': 'query_time - latest evidence end; negative means evidence ends after query.',
                      'oldest_evidence_lag_s': 'query_time - earliest evidence start.',
                      'evidence_coverage': 'Duration overlapping recorded prediction window divided by evidence union; null for point-only evidence.'})
    output.mkdir(parents=True, exist_ok=True)
    atomic_write(output / 'summary.json', report)
    write_csv(output / 'sample_features.csv', items)
    write_csv(output / 'correct_samples.csv', correct, list(dict.fromkeys(k for row in items for k in row)) or ['row_index'])
    write_csv(output / 'wrong_samples.csv', wrong, list(dict.fromkeys(k for row in items for k in row)) or ['row_index'])
    write_csv(output / 'accuracy_bins.csv', bins)
    if plots:
        plot_run(items, bins, output, path.stem)
    brief = dict(run=path.stem, scored=len(eligible), correct=len(correct), accuracy=accuracy['overall']['accuracy'],
                 correct_valid_evidence=sum(r['evidence_state'] == 'valid' for r in correct))
    brief.update(wrong=len(wrong), wrong_valid_evidence=sum(r['evidence_state'] == 'valid' for r in wrong))
    for cohort_name in ['correct', 'wrong']:
        for metric in METRICS:
            brief[f'{cohort_name}_median_{metric}'] = stats[cohort_name][metric]['median']
    print(f'{path.name}: scored={len(eligible)}, correct={len(correct)}, '
          f'correct_valid_evidence={brief["correct_valid_evidence"]}; output={output}')
    return brief


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs='+', required=True, help='prediction JSON/JSONL paths or quoted glob patterns')
    parser.add_argument('--output-dir', default='results/evidence_analysis')
    parser.add_argument('--no-plots', action='store_true', help='generate JSON/CSV without matplotlib')
    args = parser.parse_args(argv)
    paths = []
    for pattern in args.input:
        matches = sorted(glob.glob(str(Path(pattern).expanduser())))
        if not matches:
            parser.error(f'no input files matched: {pattern}')
        for match in matches:
            path = Path(match).resolve()
            if path not in paths:
                paths.append(path)
    if len({p.stem for p in paths}) != len(paths):
        parser.error('input filenames must have distinct stems to avoid output collisions')
    root = Path(args.output_dir).expanduser()
    results = [analyze_file(path, root / path.stem, not args.no_plots) for path in paths]
    write_csv(root / 'runs_summary.csv', results)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
