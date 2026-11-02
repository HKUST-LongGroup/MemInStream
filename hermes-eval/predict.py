#!/usr/bin/env python3
"""HERMES on compact query-time benchmarks, with the existing aligned prompt."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

from hermes_adapter import HERE, HermesPredictor, verify_upstream
from prompt_templates import make_prompt, prompt_version
from qwen_eval_utils import atomic_write, read_rows
from shared_runtime import parallel_results
from official_video import load_query_video, query_window


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--model', required=True, choices=['qwen25', 'llavaov'])
    p.add_argument('--model-path', required=True)
    p.add_argument('--hermes-root', type=Path, default=HERE / 'upstream/HERMES')

    p.add_argument('--system-prompt', choices=['official'], default='official')
    p.add_argument('--visual-preprocessing', choices=['official'], default='official')
    p.add_argument('--history-seconds', '--time-window', type=float, choices=[0], default=0, help='Only 0: official full history; no lookback truncation')
    p.add_argument('--fps', '--sample-fps', type=float, default=0.5, help='Official run_infer.sh example: 0.5; upstream Python CLI default is 1')
    p.add_argument('--max-frames', type=int, default=0, help='Optional error-only safety cap; 0 = unlimited as upstream')
    p.add_argument('--kv-size', type=int, default=6000)
    p.add_argument('--chunk-frames', type=int, default=16)
    p.add_argument('--max-new-tokens', type=int, default=None, help='Unset: official choice=16, no choices=256; an explicit value is an experiment override')
    p.add_argument('--repetition-penalty', type=float, default=1.1, help='Official HERMES default: 1.1')
    p.add_argument('--video-root', type=Path, default=Path('.'))
    p.add_argument('--path-prefix', nargs=2, action='append', default=[], metavar=('OLD', 'NEW'))
    p.add_argument('--num-gpus', type=int, default=1, help='one isolated model replica per GPU')
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=2024)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)
    for name in ['fps', 'kv_size', 'chunk_frames', 'repetition_penalty', 'num_gpus', 'cpu_threads']:
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            p.error(f'{name} must be finite and positive')
    if not math.isfinite(args.history_seconds) or args.history_seconds < 0 or args.limit < 0:
        p.error('history-seconds/limit must be nonnegative')
    if args.max_frames < 0 or args.kv_size < 64 or (args.max_new_tokens is not None and args.max_new_tokens <= 0):
        p.error('max-frames>=0, kv-size>=64, explicit max-new-tokens>0 required')
    if args.output.suffix not in {'.json', '.jsonl'}:
        p.error('output must end in .json or .jsonl')
    for key in ['input', 'output', 'hermes_root', 'video_root']:
        setattr(args, key, getattr(args, key).expanduser().resolve())
    args.model_path = str(Path(args.model_path).expanduser().resolve())
    if args.input in artifact_paths(args.output):
        p.error('Output artifacts must not overwrite the input')
    if any(not old or not new for old, new in args.path_prefix):
        p.error('path-prefix OLD NEW must be nonempty')
    return args


def artifact_paths(output):
    return [output] + [Path(str(output) + suffix) for suffix in
                       ['.progress.jsonl', '.manifest.json', '.summary.json']]


def resolve_video(value, args):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('missing video_path; use the compact benchmark used by the existing predictors')
    for old, new in args.path_prefix:
        old = old.rstrip('/')
        if value == old or value.startswith(old + '/'):
            value = new.rstrip('/') + value[len(old):]
            break
    path = Path(value).expanduser()
    return path if path.is_absolute() else args.video_root / path


def prepare_rows(rows, args):
    plans = []
    for index, row in enumerate(rows[:args.limit or None]):
        if row.get('evaluation_protocol', 'backward') != 'backward':
            raise ValueError(f'Row {index}: only backward/query-time QA is supported')
        path = resolve_video(row.get('video_path'), args)
        start, end = query_window(row)
        prompt = make_prompt(row, start, end)
        plans.append(dict(input_index=index, video_path=str(path), window_start=start,
                          window_end=end, prompt=prompt, video_exists=path.exists()))
    if not plans:
        raise ValueError('No input samples')
    return plans


def generation_budget(row, args):
    if args.max_new_tokens is not None:
        return args.max_new_tokens
    return 16 if row.get('options', row.get('Options')) else 256


def predict_row(index, row, args, engine, worker_index=None):
    result = dict(row)
    pred = dict(status='pending', row_index=index, input_index=index, method='HERMES',
                model=args.model, prompt_version=prompt_version(), worker_index=worker_index)
    result['prediction'] = pred
    started = time.monotonic()
    try:
        path = resolve_video(row['video_path'], args)
        start, end = query_window(row)
        prompt = make_prompt(row, start, end)
        pred.update(video_path=str(path), window_start=start, window_end=end,
                    prompt=prompt, requested_fps=args.fps)
        if args.dry_run:
            pred.update(status='dry_run', video_exists=path.exists())
        else:
            frames, stats = load_query_video(path, end, args, row)
            pred.update(stats)
            answer, diagnostics = engine.generate(prompt, frames, max_new_tokens=generation_budget(row, args))
            pred.update(status='completed', answer=answer, diagnostics=diagnostics)
    except Exception as exc:
        pred.update(status='error', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    pred['wall_seconds'] = time.monotonic() - started
    return result


def gpu_worker(connection, args, worker_index):
    try:
        engine = None if args.dry_run else HermesPredictor(args, worker_index)
        connection.send(('ready', None))
        while True:
            task = connection.recv()
            if task is None:
                break
            dispatch_index, payload = task
            index, row = payload
            result = predict_row(index, row, args, engine, worker_index)
            result['prediction']['row_index'] = dispatch_index
            connection.send(('result', result))
    except BaseException as exc:
        try:
            connection.send(('fatal', f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}'))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        connection.close()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_manifest(args, plans, source):
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in {'resume', 'output'}}
    return dict(config=config, input_sha256=digest(args.input), selected_indices=[p['input_index'] for p in plans],
                upstream=source, adapter_sha256={p.name: digest(p) for p in sorted(HERE.glob('*.py'))},
                checkpoint_config_sha256=digest(Path(args.model_path) / 'config.json')
                if (Path(args.model_path) / 'config.json').is_file() else None)


def recover(rows, output, journal):
    records = {}
    # Journal is append-only; last attempt wins. A truncated last JSONL line is
    # deliberately rejected, not silently dropped (repair it explicitly first).
    for path in [output, journal]:
        if not path.exists():
            continue
        for entry in read_rows(path):
            pred = entry.get('prediction', {})
            i = pred.get('row_index')
            if type(i) is not int or not 0 <= i < len(rows):
                raise ValueError(f'Invalid resume index in {path}')
            if {k: v for k, v in entry.items() if k != 'prediction'} != {k: v for k, v in rows[i].items() if k != 'prediction'}:
                raise ValueError(f'Resume row {i} differs from input')
            records[i] = entry
    return records


def main(argv=None):
    args = parse_args(argv)
    source = verify_upstream(args.hermes_root)
    rows = read_rows(args.input)
    plans = prepare_rows(rows, args)
    output, journal, manifest_path, summary_path = artifact_paths(args.output)
    manifest = run_manifest(args, plans, source)
    if args.resume:
        if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Resume manifest differs/missing; use original settings or a new output')
    elif any(path.exists() for path in artifact_paths(output)):
        raise FileExistsError('Output artifacts already exist; use --resume or a new output')
    if not args.dry_run:
        missing = [p['video_path'] for p in plans if not p['video_exists']]
        if missing:
            raise FileNotFoundError(f'{len(missing)} missing videos; first: {missing[0]}; use --path-prefix OLD NEW')
    records = recover(rows, output, journal) if args.resume else {}
    selected = {p['input_index'] for p in plans}
    if set(records) - selected:
        raise ValueError('Resume contains records outside selection')
    success = 'dry_run' if args.dry_run else 'completed'
    pending = [(i, rows[i]) for i in sorted(selected) if records.get(i, {}).get('prediction', {}).get('status') != success]
    atomic_write(manifest_path, manifest)
    print(json.dumps(dict(selected=len(selected), pending=len(pending), missing_videos=sum(not p['video_exists'] for p in plans),
                          model=args.model, upstream_revision=source['revision'], dry_run=args.dry_run)), flush=True)
    # Spawn even one GPU worker so device visibility is set before torch import.
    events = parallel_results(pending, args, worker_target=gpu_worker)
    try:
        with journal.open('a', encoding='utf-8') as stream:
            for result in events:
                pred = result['prediction']
                i = pred['input_index']
                pred['row_index'] = i
                if i not in selected:
                    raise RuntimeError('Worker returned an unexpected index')
                records[i] = result
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                print(f'[{sum(r["prediction"]["status"] == success for r in records.values())}/{len(selected)}] row={i} {pred["status"]}', flush=True)
                if pred['status'] == 'error':
                    print(pred['error'], file=sys.stderr, flush=True)
    finally:
        events.close()
        final = [records.get(i, dict(rows[i], prediction=dict(status='missing', row_index=i, method='HERMES')))
                 for i in sorted(selected)]
        atomic_write(output, final, jsonl=output.suffix == '.jsonl')
        atomic_write(summary_path, dict(selected=len(selected), statuses=dict(Counter(r['prediction']['status'] for r in final))))
    return 0 if all(r['prediction']['status'] == success for r in final) else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('Interrupted; completed predictions saved. Resume with the same flags plus --resume.', file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)
