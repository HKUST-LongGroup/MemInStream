#!/usr/bin/env python3
"""Run official StreamForest checkpoints on query-time benchmark JSON/JSONL."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
import multiprocessing as mp
from multiprocessing.connection import wait
from pathlib import Path
import sys
import time
import traceback

from qwen_eval_utils import atomic_write, probe_video, read_rows
from prompt_templates import make_prompt, prompt_version
from video_input import window, decode_video, resolve_video
from streamforest_backend import Predictor, verify_upstream, decode_official
from score_predictions import summarize

HERE = Path(__file__).resolve().parent

def gpu_worker(connection, args, worker_index):
    """Spawned process: one checkpoint on one logical CUDA device."""
    try:
        engine = None if args.dry_run else Predictor(args, device_index=worker_index)
        connection.send(('ready', None))
        while True:
            task = connection.recv()
            if task is None:
                break
            index, row = task
            result = predict_row(index, row, args, engine, worker_index)
            connection.send(('result', result))
            if result['prediction']['status'] == 'error' and not args.dry_run:
                gc.collect()
                engine.torch.cuda.empty_cache()
    except BaseException as exc:
        try:
            connection.send(('fatal', traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def parallel_results(rows, args, worker_target=gpu_worker):
    """Keep one outstanding sample per worker; yield completed rows immediately.

    Pipes bound queued work; process sentinels detect crashes without waiting
    forever for a result. Only the parent process writes output files.
    """
    if not rows:
        return
    ctx = mp.get_context('spawn')
    workers, active, assigned = [], {}, {}
    next_index = 0
    try:
        for rank in range(min(args.num_gpus, len(rows))):
            parent, child = ctx.Pipe()
            process = ctx.Process(target=worker_target, args=(child, args, rank), name=f'streamforest-gpu-{rank}')
            try:
                process.start()
            except BaseException:
                parent.close()
                child.close()
                raise
            child.close()
            workers.append((process, parent))
            active[parent] = (process, rank)
            assigned[parent] = None
        while active:
            ready = wait(list(active) + [process.sentinel for process, _ in active.values()])
            # Drain messages before checking sentinels so fatal errors retain their details.
            for connection in list(active):
                if connection not in ready:
                    continue
                process, rank = active[connection]
                try:
                    kind, payload = connection.recv()
                except (EOFError, OSError) as exc:
                    raise RuntimeError(f'GPU worker {rank} exited without a result') from exc
                if kind == 'fatal':
                    raise RuntimeError(f'GPU worker {rank} failed: {payload}')
                if kind == 'result':
                    if payload['prediction']['row_index'] != assigned[connection]:
                        raise RuntimeError(f'GPU worker {rank} returned an unexpected row')
                    yield payload
                elif kind != 'ready' or assigned[connection] is not None:
                    raise RuntimeError(f'GPU worker {rank} sent unexpected message: {kind}')
                if next_index < len(rows):
                    assigned[connection] = next_index
                    connection.send((next_index, rows[next_index]))
                    next_index += 1
                else:
                    connection.send(None)
                    del active[connection]
            for connection, (process, rank) in active.items():
                if process.sentinel in ready:
                    # A message may have arrived just after wait returned.
                    if connection.poll():
                        continue
                    raise RuntimeError(f'GPU worker {rank} exited unexpectedly (exitcode={process.exitcode})')
    finally:
        # Also runs on Ctrl-C, write failures and worker initialization failures.
        for process, connection in workers:
            connection.close()
            if process.is_alive():
                process.terminate()
        for process, _ in workers:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)


def validate_gpu_count(args):
    if args.num_gpus > 1 and not args.dry_run:
        import torch
        available = torch.cuda.device_count()
        if args.num_gpus > available:
            raise ValueError(f'--num-gpus={args.num_gpus}, but only {available} CUDA GPUs are visible; '
                             'check CUDA_VISIBLE_DEVICES')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True, help='new .json or .jsonl output; existing outputs are protected')
    p.add_argument('--streamforest-root', default=str(HERE / 'StreamForest'))
    p.add_argument('--model-path', default='MCG-NJU/StreamForest-Qwen2-7B')
    p.add_argument('--vision-tower', default='google/siglip-so400m-patch14-384',
                   help='HF ID or local SigLIP directory (local basename must contain siglip)')

    p.add_argument('--system-prompt', choices=['aligned', 'official'], default='aligned',
                   help='aligned uses the Qwen baseline system message inside official qwen_2 chat format')
    p.add_argument('--history-seconds', '--time-window', dest='history_seconds', type=float, default=600,
                   help='lookback seconds; 0 = all video before the question')
    p.add_argument('--sampling', choices=['aligned', 'official'], default='aligned',
                   help='aligned=Qwen FFmpeg; official=StreamForest dynamic_fps1/decord')
    p.add_argument('--fps', type=float, default=1)
    p.add_argument('--max-frames', type=int, default=2048)
    p.add_argument('--max-pixels', type=int, default=153664, help='aligned decoder pixels before native SigLIP preprocessing')
    p.add_argument('--resize-factor', type=int, choices=[28, 32], default=28,
                   help='28 matches qwen25 input resizing; 32 matches qwen3')
    p.add_argument('--max-input-tokens', type=int, default=32768)
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--num-gpus', type=int, default=1, help='one independent model per visible GPU')
    p.add_argument('--attn-implementation', choices=['sdpa', 'flash_attention_2', 'eager'], default='sdpa')
    p.add_argument('--video-root', default='.')
    p.add_argument('--path-prefix', nargs=2, action='append', default=[], metavar=('OLD', 'NEW'))
    p.add_argument('--video-timeout', type=float, default=600)
    p.add_argument('--ffmpeg-threads', type=int, default=2)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--dry-run', action='store_true', help='validate prompts/windows without model, CUDA or video decoding')
    args = p.parse_args(argv)
    for key in ['fps', 'max_frames', 'max_pixels', 'max_input_tokens', 'max_new_tokens',
                'num_gpus', 'video_timeout', 'ffmpeg_threads']:
        value = getattr(args, key)
        if not math.isfinite(value) or value <= 0:
            p.error(f'--{key.replace("_", "-")} must be finite and positive')
    if not math.isfinite(args.history_seconds) or args.history_seconds < 0 or args.limit < 0:
        p.error('history-seconds and limit must be nonnegative')
    if args.max_frames < 4 or args.max_pixels < 1024:
        p.error('max-frames must be >=4 and max-pixels >=1024')
    if args.sampling == 'official' and args.fps != 1:
        p.error('official dynamic_fps1 does not support configurable FPS; use --sampling aligned')
    if args.max_new_tokens >= args.max_input_tokens:
        p.error('max-new-tokens must be smaller than max-input-tokens')
    if not 0 <= args.seed < 2**32:
        p.error('seed must be in [0, 2**32)')
    if Path(args.output).suffix not in {'.json', '.jsonl'}:
        p.error('output must end in .json or .jsonl')
    return args


def predict_row(index, row, args, engine, worker_index=None):
    result = dict(row)
    # Input may itself be a baseline prediction file. Never retain its answers
    # as a new model's predictions, or use them in the prompt.
    result.pop('prediction', None)
    prediction = dict(status='pending', row_index=index, model=args.model_path,
                      prompt_version=prompt_version(),
                      sampling=args.sampling, worker_index=worker_index, audio_used=False)
    result['prediction'] = prediction
    started = time.monotonic()
    try:
        if row.get('evaluation_protocol', 'backward') != 'backward':
            raise ValueError('Only backward query-time QA is supported')
        if not isinstance(row.get('video_path'), str) or not row['video_path'].strip():
            raise ValueError('video_path must be a nonempty string')
        path = resolve_video(row['video_path'], args)
        start, end = window(row, args.history_seconds)
        if not args.dry_run:
            if not path.is_file():
                raise FileNotFoundError(path)
            metadata = probe_video(str(path), args.video_timeout)
            start, end = window(row, args.history_seconds, metadata[2])
        prompt = make_prompt(row, start, end)
        prediction.update(video_path=str(path), window_start=start, window_end=end,
                          requested_fps=args.fps, prompt=prompt, system_prompt=args.system_prompt)
        if args.sampling == 'aligned':
            bound = math.ceil((end-start) * args.fps)
            if bound + bound % 2 > args.max_frames:
                raise ValueError('Frame budget exceeded; increase max-frames or explicitly reduce history/FPS')
        if args.dry_run:
            prediction.update(status='dry_run', video_exists=path.is_file())
        else:
            frames, stats = (decode_official(path, start, end, args) if args.sampling == 'official'
                             else decode_video(path, start, end, metadata, args))
            prediction.update(stats)
            answer, tokens = engine.generate(prompt, frames)
            prediction.update(status='completed', answer=answer, input_tokens=tokens,
                              visual_tokens=engine.visual_tokens, backend=engine.info)
    except Exception as exc:
        prediction.update(status='error', error=f'{type(exc).__name__}: {exc}')
    prediction['wall_seconds'] = time.monotonic() - started
    return result


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    journal = output.with_suffix('.progress.jsonl')
    manifest_path = Path(str(output) + '.manifest.json')
    report_path = Path(str(output) + '.accuracy.json')
    details_path = Path(str(output) + '.scores.jsonl')
    all_paths = [Path(args.input).expanduser().resolve(), output, journal,
                 manifest_path, report_path, details_path]
    if len(set(all_paths)) != len(all_paths):
        raise ValueError('Input, output and derived output paths must all differ')
    for path in all_paths[1:]:
        if path.exists():
            raise FileExistsError(f'Use a new output name; file already exists: {path}')
    source = verify_upstream(args.streamforest_root)
    rows = read_rows(Path(args.input).expanduser())
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError('No input samples')
    # Normalize legacy aliases for both prompt generation AND scoring.
    rows = [dict(row) for row in rows]
    for row in rows:
        for key, alias in [('question', 'Question'), ('query_time', 'Question time'), ('options', 'Options')]:
            if key not in row and alias in row:
                row[key] = row[alias]
    validate_gpu_count(args)
    config = vars(args).copy()
    manifest = dict(config=config, input_sha256=hashlib.sha256(Path(args.input).expanduser().read_bytes()).hexdigest(),
                    upstream_revision=source['revision'],
                    source_snapshot_sha256=hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
                    adapter_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in HERE.glob('*.py') if not p.name.startswith('test_')})
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(manifest_path, manifest)
    if args.num_gpus > 1:
        events = parallel_results(rows, args)
    else:
        engine = None if args.dry_run else Predictor(args)
        events = (predict_row(index, row, args, engine) for index, row in enumerate(rows))
    results, failures = [None] * len(rows), 0
    try:
        with journal.open('x', encoding='utf-8') as stream:
            for completed, result in enumerate(events, 1):
                prediction = result['prediction']
                index = prediction['row_index']
                if not 0 <= index < len(rows) or results[index] is not None:
                    raise RuntimeError(f'Duplicate/invalid result index: {index}')
                results[index] = result
                failures += prediction['status'] == 'error'
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                print(f'[{completed}/{len(rows)}] row={index} {prediction["status"]} '
                      f'{prediction.get("error", "")}', flush=True)
                if prediction['status'] == 'error' and not args.dry_run and args.num_gpus == 1:
                    gc.collect()
                    engine.torch.cuda.empty_cache()
    finally:
        events.close()
    if any(result is None for result in results):
        raise RuntimeError('Missing results; inspect progress JSONL')
    atomic_write(output, results, jsonl=output.suffix == '.jsonl')
    if not args.dry_run:
        report, details = summarize(results)
        atomic_write(report_path, report)
        atomic_write(details_path, details, jsonl=True)
        stats = report['overall']
        accuracy = f'{stats["accuracy"]:.2%}' if stats['accuracy'] is not None else 'N/A'
        print(f'Accuracy: {stats["correct"]}/{stats["total"]} = {accuracy}; report={report_path}', flush=True)
    print(f'Output: {output}; rows={len(rows)}; errors={failures}', flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
