#!/usr/bin/env python3
"""Flash-VStream-Qwen inference on the user's unified benchmark, with resumable output."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import traceback

from backend import Predictor, decode_window
from prompt_templates import make_prompt, prompt_version
from qwen_eval_utils import atomic_write, read_rows, seconds
from scheduling import parallel_results

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_upstream(root):
    manifest = json.loads((HERE / 'upstream_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path = Path(root) / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f'Official source differs from pinned snapshot: {path}')
    return manifest['commit']


def window(row, history_seconds, duration=None):
    query = seconds(row.get('query_time', row.get('Question time')))
    start = max(0., query - history_seconds) if history_seconds else 0.
    end = min(query, duration) if duration is not None else query
    if end <= start:
        raise ValueError(f'Empty video window [{start}, {end})')
    return start, end


def resolve_video(value, args):
    path = str(value)
    for old, new in args.path_prefix:
        old = old.rstrip('/')
        if path == old or path.startswith(old + '/'):
            path = new.rstrip('/') + path[len(old):]
            break
    path = Path(path).expanduser()
    return path if path.is_absolute() else Path(args.video_root).expanduser() / path


def predict_row(index, row, args, engine, worker_index=None):
    result = dict(row)
    if 'prediction' in result:
        result['input_prediction'] = result['prediction']
    pred = dict(status='pending', row_index=index, input_index=index, model='flash_vstream_qwen',
                prompt_version=prompt_version(), worker_index=worker_index,
                config={k: v for k, v in vars(args).items() if k not in ('input', 'output', 'resume')})
    result['prediction'] = pred
    result['flash_vstream_prediction'] = pred
    try:
        for field in ('source_dataset', 'source_id', 'video_path'):
            if row.get(field) is None:
                raise ValueError(f'Missing {field}')
        # This adapter implements the same per-question backward observation protocol.
        if row.get('evaluation_protocol') in ('proactive', 'forward'):
            raise ValueError('Future/proactive observation protocols are not implemented')
        path = resolve_video(row['video_path'], args)
        start, end = window(row, args.history_seconds)
        if not args.dry_run:
            if not path.is_file():
                raise FileNotFoundError(path)
            if args.seed is not None:
                import numpy as np
                import torch
                random.seed(args.seed + index)
                np.random.seed((args.seed + index) % (2**32))
                torch.manual_seed(args.seed + index)
            images, end, stats = decode_window(path, start, end, args)
            pred.update(stats)
        prompt = make_prompt(row, start, end)
        pred.update(video_path=str(path), window_start=start, window_end=end, prompt=prompt,
                    chat_system=args.chat_system)
        if args.dry_run:
            pred.update(status='dry_run', video_exists=path.is_file())
        else:
            answer, diagnostics = engine.generate(prompt, images)
            pred.update(status='completed', answer=answer, response=answer, **diagnostics)
    except Exception as exc:
        pred.update(status='error', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    return result


def gpu_worker(connection, args, worker_index):
    try:
        if not args.dry_run:
            import torch
            torch.cuda.set_device(worker_index)
        engine = None if args.dry_run else Predictor(args)
        connection.send(('ready', None))
        consecutive = 0
        while True:
            task = connection.recv()
            if task is None:
                break
            index, row = task
            result = predict_row(index, row, args, engine, worker_index)
            failed = result['prediction']['status'] == 'error'
            consecutive = consecutive + 1 if failed else 0
            connection.send(('result', result))
            if failed and not args.dry_run:
                gc.collect()
                torch.cuda.empty_cache()
            if args.max_consecutive_errors and consecutive >= args.max_consecutive_errors:
                raise RuntimeError(f'Stopped after {consecutive} consecutive errors; inspect saved traceback')
    except BaseException as exc:
        try:
            connection.send(('fatal', f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}'))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model-path', default=str(HERE / 'models/Flash-VStream-Qwen-7b'))
    p.add_argument('--processor-path', default=str(HERE / 'models/Qwen2-VL-7B-Instruct'))
    p.add_argument('--upstream-root', default=str(HERE / 'upstream'))
    p.add_argument('--history-seconds', '--time-window', type=float, default=0,
                   help='0 = all past video; 256 matches default FluxMem observation horizon')

    p.add_argument('--chat-system', choices=['official', 'qwen'], default='official',
                   help='official/FluxMem user-only messages; qwen adds the supplied Qwen system prompt')
    p.add_argument('--extract-fps', type=float, default=1,
                   help='Official candidate extraction FPS, before uniform max-frames sampling')
    p.add_argument('--max-frames', type=int, default=240)
    p.add_argument('--max-pixels', type=int, default=200704)
    p.add_argument('--decoder-threads', type=int, default=2)
    p.add_argument('--num-gpus', type=int, default=1)
    p.add_argument('--seed', type=int, default=None, help='Optional per-row seed; official script does not set one')
    p.add_argument('--video-root', default='.')
    p.add_argument('--path-prefix', nargs=2, action='append', default=[], metavar=('OLD', 'NEW'))
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--max-consecutive-errors', type=int, default=5)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args(argv)
    if not math.isfinite(args.history_seconds) or args.history_seconds < 0:
        p.error('--history-seconds must be finite and nonnegative')
    if not math.isfinite(args.extract_fps) or args.extract_fps <= 0:
        p.error('--extract-fps must be finite and positive')
    if args.max_frames < 2 or args.max_frames % 2 or args.max_pixels < 128*28*28:
        p.error('--max-frames must be even and >=2; --max-pixels must be >=100352 (official VIDEO_MIN_PIXELS)')
    if min(args.num_gpus, args.decoder_threads) < 1 or min(args.limit, args.max_consecutive_errors) < 0:
        p.error('Invalid worker/thread/limit/error count')
    if args.seed is not None and not 0 <= args.seed < 2**32:
        p.error('--seed must be in [0, 2**32)')
    if Path(args.output).suffix not in ('.json', '.jsonl'):
        p.error('--output must end in .json or .jsonl')
    return args


def run_manifest(args):
    versions = {}
    for package in ('torch', 'torchvision', 'transformers', 'flash-attn', 'accelerate', 'decord', 'numpy', 'pillow'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    config = {k: v for k, v in vars(args).items() if k not in ('resume', 'output')}
    return dict(input_sha256=sha(args.input), config=config, versions=versions,
                upstream_commit=verify_upstream(args.upstream_root),
                adapter_files={p.name: sha(p) for p in sorted(HERE.glob('*.py'))},
                checkpoint_metadata={str(p): sha(p) for folder in (args.model_path, args.processor_path)
                                     for p in sorted(Path(folder).glob('*.json'))},
                prompt_version=prompt_version())


def load_progress(path, rows):
    records = {}
    if not path.exists():
        return records
    # A partial final line is a recoverable interrupted append; rewrite journal atomically.
    lines = path.read_text().splitlines()
    valid = []
    for n, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if n != len(lines) - 1:
                raise
            atomic_write(path, valid, jsonl=True)
            break
        index = record['prediction']['row_index']
        if type(index) is not int or not 0 <= index < len(rows):
            raise ValueError('Journal contains invalid row index')
        if any(record.get(key) != rows[index].get(key) for key in ('source_dataset', 'source_id', 'question')):
            raise ValueError('Journal/input identity mismatch')
        valid.append(record)
        records[index] = record
    return records


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    journal = Path(str(output) + '.progress.jsonl')
    manifest_path = Path(str(output) + '.manifest.json')
    summary_path = Path(str(output) + '.summary.json')
    if Path(args.input).expanduser().resolve() in (output, journal, manifest_path, summary_path):
        raise ValueError('Output must not overwrite input')
    rows = read_rows(args.input)
    if args.limit:
        rows = rows[:args.limit]
    manifest = run_manifest(args)
    if args.resume:
        if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Resume requires identical input, prompt, code, environment and configuration')
    elif any(p.exists() for p in (output, journal, manifest_path, summary_path)):
        raise FileExistsError('Use a new --output or --resume with the identical configuration')
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(manifest_path, manifest)
    records = load_progress(journal, rows) if args.resume else {}
    done_status = 'dry_run' if args.dry_run else 'completed'
    pending = [(i, row) for i, row in enumerate(rows) if records.get(i, {}).get('prediction', {}).get('status') != done_status]
    fatal = None
    events = None
    try:
        if pending and not args.dry_run:
            import torch
            if torch.cuda.device_count() < args.num_gpus:
                raise ValueError(f'Requested {args.num_gpus} GPUs, visible {torch.cuda.device_count()}')
        events = parallel_results(pending, args, gpu_worker)
        with journal.open('a', encoding='utf-8') as stream:
            for result in events:
                pred = result['prediction']
                index = pred['row_index']
                records[index] = result
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                print(f'row={index} status={pred["status"]} {pred.get("error", "")}', flush=True)
    except BaseException as exc:
        fatal = f'{type(exc).__name__}: {exc}'
        print(fatal, file=sys.stderr, flush=True)
    finally:
        if events is not None:
            events.close()
        results = []
        for i, row in enumerate(rows):
            if i in records:
                results.append(records[i])
            else:
                record = dict(row)
                if 'prediction' in record:
                    record['input_prediction'] = record['prediction']
                pred = dict(status='missing', row_index=i, input_index=i, model='flash_vstream_qwen')
                record.update(prediction=pred, flash_vstream_prediction=pred)
                results.append(record)
        atomic_write(output, results, jsonl=output.suffix == '.jsonl')
        completed = sum(r['prediction']['status'] == done_status for r in results)
        summary = dict(total=len(rows), completed=completed, remaining=len(rows)-completed, fatal=fatal,
                       dry_run=args.dry_run, output=str(output))
        atomic_write(summary_path, summary)
        print(json.dumps(summary), flush=True)
    return int(bool(fatal) or completed != len(rows))


if __name__ == '__main__':
    raise SystemExit(main())
