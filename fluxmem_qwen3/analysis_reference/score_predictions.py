#!/usr/bin/env python3
"""Score multiple-choice, binary and counting predictions without loading a model."""
import argparse
import json
import re
import unicodedata
from pathlib import Path

from .qwen_eval_utils import atomic_write, options, read_rows


def clean(value):
    if not isinstance(value, (str, bool)):
        return ''
    text = unicodedata.normalize('NFKC', str(value)).strip()
    text = text.replace('**', '').replace('`', '').strip()
    text = re.sub(r'^(?:the\s+answer\s+is|answer\s*[:：]|答案(?:是|为|[:：]))\s*', '', text, flags=re.I)
    return text.strip().strip(' .。!！').strip()


def parse_choice(value, choices):
    """Accept a complete letter/list response or exact option text; never guess from prose."""
    if isinstance(value, list):
        if not value:
            return None
        parsed = [parse_choice(item, choices) for item in value]
        return set().union(*parsed) if all(item is not None for item in parsed) else None
    text = clean(value)
    valid = {chr(65 + i) for i in range(len(choices))}
    letters = text.upper().strip('()[] ')
    if re.fullmatch(r'[A-Z](?:\s*[,、;/]\s*[A-Z])*', letters):
        result = set(re.split(r'\s*[,、;/]\s*', letters))
        return result if result <= valid else None
    # Also accept the exact full option, e.g. "B. Blue", or "Blue".
    matches = set()
    for i, option in enumerate(choices):
        label = chr(65 + i)
        if text.casefold() in {clean(option).casefold(), clean(f'{label}. {option}').casefold(),
                               clean(f'{label}) {option}').casefold()}:
            matches.add(label)
    return matches if len(matches) == 1 else None


def parse_binary(value):
    text = clean(value).casefold()
    if text in {'true', 'yes', '是', '对', '正确', '真'}:
        return True
    if text in {'false', 'no', '否', '不', '不对', '错误', '错', '假'}:
        return False
    return None


def parse_count(value):
    """Parse a nonnegative integer, optionally followed by times/次; reject ambiguous prose."""
    if type(value) is int:
        return value if value >= 0 else None
    if not isinstance(value, str):
        return None
    text = unicodedata.normalize('NFKC', value).strip().replace('**', '').replace('`', '').strip()
    text = re.sub(r'^(?:the\s+answer\s+is|answer\s*[:：]|答案(?:是|为|[:：]))\s*', '', text, flags=re.I)
    match = re.fullmatch(r'([0-9]+)(?:\s*(?:times?|次))?\s*[.。!！]?', text.strip(), flags=re.I)
    return int(match.group(1)) if match else None


def kind(row):
    question_type = str(row.get('question_type', '')).lower()
    if question_type == 'binary':
        return 'binary'
    if row.get('options') or question_type in {'multiple-choice', 'multiple-select'}:
        return 'multiple_choice'
    if str(row.get('original_task', '')).strip().casefold() in {'repetition event count', 'rec'} or question_type in {'counting', 'count'}:
        return 'counting'
    return 'open_ended_or_other'


def choice_gold(row, choices):
    if row.get('correct_option') is not None:
        return parse_choice(row['correct_option'], choices)
    if row.get('correct_index') is not None:
        indices = row['correct_index']
        if not isinstance(indices, list):
            indices = [indices]
        if indices and all(type(i) is int and 0 <= i < len(choices) for i in indices):
            return {chr(65 + i) for i in indices}
        return None
    return parse_choice(row.get('answer'), choices)


def score_row(row):
    category = kind(row)
    if category == 'open_ended_or_other':
        return category, 'excluded'
    choices = options(row)
    parser = parse_count if category == 'counting' else parse_binary
    gold = choice_gold(row, choices) if category == 'multiple_choice' else parser(row.get('answer'))
    if gold is None:
        return category, 'invalid_reference'
    prediction = row.get('prediction')
    if not isinstance(prediction, dict) or prediction.get('status') != 'completed':
        return category, 'prediction_failed_or_missing'
    predicted = (parse_choice(prediction.get('answer'), choices) if category == 'multiple_choice'
                 else parser(prediction.get('answer')))
    if predicted is None:
        return category, 'unparseable_prediction'
    return category, 'correct' if predicted == gold else 'incorrect'


def summarize(rows):
    if any(isinstance(row.get('prediction'), dict) and row['prediction'].get('status') == 'dry_run'
           for row in rows):
        raise ValueError('输入包含 dry_run 记录，没有真实预测答案，不能统计准确率。请使用实际推理输出。')
    names = ['multiple_choice', 'binary', 'counting', 'overall']
    counts = {name: dict(total=0, correct=0, incorrect=0, prediction_failed_or_missing=0,
                         unparseable_prediction=0, invalid_reference=0) for name in names}
    details, excluded = [], 0
    for index, row in enumerate(rows):
        category, reason = score_row(row)
        details.append(dict(input_index=index, source_dataset=row.get('source_dataset'),
                            source_id=row.get('source_id'), category=category, result=reason))
        if reason == 'excluded':
            excluded += 1
            continue
        for name in [category, 'overall']:
            counts[name]['total'] += 1
            counts[name][reason] += 1
    for stats in counts.values():
        stats['wrong'] = stats['total'] - stats['correct']
        stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] else None
    return dict(input_rows=len(rows), excluded_open_ended_or_other=excluded,
                policy='All choice/binary/counting rows count in denominator; failed, unparseable and invalid-reference rows count as wrong.',
                **counts), details


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='prediction JSON array or JSONL')
    parser.add_argument('--report', help='optional summary JSON output')
    parser.add_argument('--details', help='optional per-sample scoring JSONL output')
    args = parser.parse_args(argv)
    # Never overwrite the input predictions with a summary or details file.
    paths = [Path(value).expanduser().resolve() for value in [args.input, args.report, args.details] if value]
    if len(set(paths)) != len(paths):
        parser.error('input、report、details 必须使用不同路径')
    try:
        report, details = summarize(read_rows(Path(args.input).expanduser()))
    except ValueError as exc:
        parser.error(str(exc))
    print(f'输入记录：{report["input_rows"]}；开放题/其他（不计分）：{report["excluded_open_ended_or_other"]}')
    for key, title in [('multiple_choice', '选择题'), ('binary', '判断题'), ('counting', '计数题'), ('overall', '合计')]:
        stats = report[key]
        accuracy = f'{stats["accuracy"]:.2%}' if stats['accuracy'] is not None else 'N/A'
        print(f'{title}：总题数 {stats["total"]}，答对 {stats["correct"]}，答错 {stats["wrong"]}，准确率 {accuracy}')
    stats = report['overall']
    print(f'计入答错：预测失败/缺失 {stats["prediction_failed_or_missing"]}，'
          f'答案无法解析 {stats["unparseable_prediction"]}，标准答案无效 {stats["invalid_reference"]}')
    if args.report:
        atomic_write(Path(args.report).expanduser(), report)
    if args.details:
        atomic_write(Path(args.details).expanduser(), details, jsonl=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
