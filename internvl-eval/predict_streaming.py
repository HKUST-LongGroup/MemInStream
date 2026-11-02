#!/usr/bin/env python3
"""Shared streaming benchmark runner for local InternVL3.5 and LLaVA-OneVision."""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
from multiprocessing.connection import wait
import subprocess
from pathlib import Path

from qwen_eval_utils import atomic_write, probe_video, read_rows, seconds


from prompt_templates import make_prompt, prompt_version


def window(row, history_seconds, duration=None):
    query = seconds(row.get('query_time', row.get('Question time')))
    start = max(0.0, query - history_seconds) if history_seconds else 0.0
    end = min(query, duration) if duration is not None else query
    if end <= start:
        raise ValueError(f'empty video window [{start}, {end}); query_time={query}')
    return start, end


def frame_size(width, height, max_pixels, factor):
    scale = min(1.0, math.sqrt(max_pixels / (width * height)))
    w = max(factor, int(width * scale / factor) * factor)
    h = max(factor, int(height * scale / factor) * factor)
    if w * h > max_pixels:
        if w >= h:
            w = max(factor, (max_pixels // (h * factor)) * factor)
        else:
            h = max(factor, (max_pixels // (w * factor)) * factor)
    return w, h


def decode_video(path, start, end, metadata, args):
    import numpy as np
    count_bound = math.ceil((end - start) * args.fps)
    # A finite requested history determines the input frame count even if the
    # available video is shorter. Zero history means all available past video.
    target_frames = math.ceil((args.history_seconds or (end - start)) * args.fps)
    if max(count_bound, target_frames) > args.max_frames:
        raise ValueError('frame budget exceeded; increase --max-frames or explicitly lower --fps / --history-seconds')
    width, height = (frame_size(*metadata[:2], args.max_pixels, 1)
                     if args.max_pixels else metadata[:2])
    duration = end - start
    # Trim BEFORE resampling. Frames at/after the query boundary cannot enter the model.
    filters = (f'trim=duration={duration:.9f},setpts=PTS-STARTPTS,'
               f'fps={args.fps:.9f}:round=down,scale={width}:{height}')
    command = [
        'ffmpeg', '-nostdin', '-v', 'error', '-threads', str(args.ffmpeg_threads),
        '-ss', f'{start:.9f}', '-t', f'{duration:.9f}', '-i', str(path),
        '-map', '0:v:0', '-an', '-sn', '-dn', '-vf', filters,
        '-filter_threads', str(args.ffmpeg_threads), '-threads', str(args.ffmpeg_threads),
        '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1',
    ]
    result = subprocess.run(command, capture_output=True, timeout=args.video_timeout)
    if result.returncode:
        raise ValueError(result.stderr.decode(errors='replace')[-2000:])
    if not result.stdout:
        # A sub-second clip can have source frames but no output from fps with
        # round=down. Recover one frame from the SAME trimmed past-only window.
        fallback = list(command)
        fallback[fallback.index('-vf') + 1] = (
            f'trim=duration={duration:.9f},setpts=PTS-STARTPTS,scale={width}:{height}')
        fallback[-1:-1] = ['-frames:v', '1']
        result = subprocess.run(fallback, capture_output=True, timeout=args.video_timeout)
        if result.returncode:
            raise ValueError(result.stderr.decode(errors='replace')[-2000:])
        if not result.stdout:
            raise ValueError('no source frames in requested window; cannot pad an empty video')
    frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, height, width, 3).copy()
    decoded = len(frames)
    if decoded > args.max_frames:
        raise ValueError('decoded video exceeds --max-frames')
    if decoded > target_frames:
        raise ValueError('decoded frame count exceeds requested window budget')
    padding = target_frames - decoded
    if padding:
        frames = np.concatenate([frames, np.repeat(frames[-1:], padding, axis=0)], axis=0)
    return frames, {'decoded_frames': decoded, 'input_frames': len(frames),
                    'target_frames': target_frames, 'padding_frames': padding,
                    'padding_policy': 'repeat_last', 'sampling_policy': 'fixed_window_frames_v1',
                    'width': width, 'height': height}


def Predictor(args, device_index=None):
    if args.model == 'internvl35':
        from predict_filtered_internvl35 import InternVL35Predictor
        return InternVL35Predictor(args, device_index)
    from predict_filtered_llava_onevision import LlavaOnevisionPredictor
    return LlavaOnevisionPredictor(args, device_index)


def resolve_video(value, args):
    path = str(value)
    for old, new in args.path_prefix:
        if path == old or path.startswith(old.rstrip('/') + '/'):
            path = new.rstrip('/') + path[len(old.rstrip('/')):]
            break
    path = Path(path).expanduser()
    return path if path.is_absolute() else Path(args.video_root).expanduser() / path


def parse_args(argv=None, model=None):
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument('--input', required=True, help='compact JSON array or JSONL')
    p.add_argument('--output', required=True, help='.json (array + progress JSONL) or .jsonl')
    p.add_argument('--model', required=model is None, default=model,
                   choices=[model] if model else ['internvl35', 'llava_onevision'])
    p.add_argument('--model-path', required=True, help='local checkpoint directory')
    p.add_argument('--history-seconds', type=float, default=60, help='requested history; pad short video to ceil(history*fps) frames; 0 = all past video')
    p.add_argument('--fps', type=float, default=1, help='sampling FPS, including fractional values such as 0.5')
    p.add_argument('--video-root', default='.', help='base directory for relative video paths')
    p.add_argument('--path-prefix', nargs=2, action='append', default=[], metavar=('OLD', 'NEW'))
    p.add_argument('--max-frames', type=int, default=2048, help='hard cap; never silently reduces FPS')
    p.add_argument('--max-pixels', type=int, default=0, help='optional decode pixel cap; 0 keeps source resolution')
    p.add_argument('--max-input-tokens', type=int, default=32768, help='total input + generation budget')
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--num-gpus', type=int, default=1, help='>1: one model replica per visible GPU with dynamic sample scheduling')
    p.add_argument('--device-map', default='auto', help='auto permits sharding across visible GPUs')
    p.add_argument('--attn-implementation', choices=['sdpa', 'flash_attention_2', 'eager'], default=None, help='default: InternVL eager, LLaVA sdpa')
    p.add_argument('--dtype', choices=['auto', 'bfloat16', 'float16'], default='auto')
    p.add_argument('--vision-batch-size', type=int, default=8, help='InternVL vision encoder chunk size')
    p.add_argument('--video-timeout', type=float, default=600)
    p.add_argument('--ffmpeg-threads', type=int, default=2)
    p.add_argument('--limit', type=int, default=0, help='0 = all supplied samples, with no additional filtering')
    p.add_argument('--dry-run', action='store_true', help='validate rows/windows/prompts without model or videos')
    args = p.parse_args(argv)
    for key in ['fps', 'max_frames', 'vision_batch_size', 'max_input_tokens', 'max_new_tokens', 'video_timeout', 'ffmpeg_threads']:
        value = getattr(args, key)
        if not math.isfinite(value) or value <= 0:
            p.error(f'--{key.replace("_", "-")} must be finite and positive')
    if not math.isfinite(args.history_seconds) or args.history_seconds < 0 or args.limit < 0:
        p.error('--history-seconds and --limit must be nonnegative')
    if args.num_gpus < 1:
        p.error('--num-gpus must be >=1')
    if args.num_gpus > 1 and args.device_map != 'auto':
        p.error('--device-map is only configurable with --num-gpus 1; parallel workers use one GPU each')
    if args.max_pixels < 0 or 0 < args.max_pixels < 1024:
        p.error('--max-pixels must be 0 or >=1024')
    if args.attn_implementation is None:
        args.attn_implementation = 'eager' if args.model == 'internvl35' else 'sdpa'
    if args.model == 'internvl35' and args.attn_implementation == 'sdpa':
        p.error('InternVL custom code supports eager or flash_attention_2, not sdpa')
    if Path(args.output).suffix not in {'.json', '.jsonl'}:
        p.error('--output must end in .json or .jsonl')
    return args


def predict_row(index, row, args, engine, worker_index=None):
    result = dict(row)
    config = {k: v for k, v in vars(args).items() if k not in {'input', 'output', 'limit'}}
    prediction = {'status': 'pending', 'row_index': index, 'config': config,
                  'prompt_version': prompt_version()}
    if worker_index is not None:
        prediction.update(worker_index=worker_index,
                          cuda_device=None if args.dry_run else worker_index)
    result['prediction'] = prediction
    try:
        path = resolve_video(row['video_path'], args)
        start, end = window(row, args.history_seconds)
        if not args.dry_run:
            metadata = probe_video(str(path), args.video_timeout)
            start, end = window(row, args.history_seconds, metadata[2])
        prompt = make_prompt(row, start, end)
        prediction.update(video_path=str(path), window_start=start, window_end=end,
                          requested_fps=args.fps)
        if args.dry_run:
            prediction.update(status='dry_run', prompt=prompt, video_exists=path.is_file())
        else:
            frames, stats = decode_video(path, start, end, metadata, args)
            prediction.update(stats)
            if stats['padding_frames']:
                prompt += (f'\nThe final {stats["padding_frames"]} frame(s) repeat the last available '
                           'frame to fill the requested input length; they do not show additional events '
                           'or extend the observed video time.')
            answer, tokens = engine.generate(prompt, frames)
            prediction.update(status='completed', answer=answer, input_tokens=tokens)
    except Exception as exc:
        prediction.update(status='error', error=f'{type(exc).__name__}: {exc}')
    return result


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
            connection.send(('result', predict_row(index, row, args, engine, worker_index)))
    except BaseException as exc:
        try:
            connection.send(('fatal', f'{type(exc).__name__}: {exc}'))
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
            process = ctx.Process(target=worker_target, args=(child, args, rank), name=f'{args.model}-gpu-{rank}')
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


def main(argv=None, model=None):
    args = parse_args(argv, model=model)
    output = Path(args.output).expanduser().resolve()
    journal = output.with_suffix('.progress.jsonl')
    if output.exists() or journal.exists():
        raise FileExistsError(f'output/progress file already exists; use a new --output: {output}')
    rows = read_rows(Path(args.input).expanduser())
    if args.limit:
        rows = rows[:args.limit]
    validate_gpu_count(args)
    if args.num_gpus > 1:
        events = parallel_results(rows, args)
    else:
        engine = None if args.dry_run or not rows else Predictor(args)
        events = (predict_row(index, row, args, engine) for index, row in enumerate(rows))
    output.parent.mkdir(parents=True, exist_ok=True)
    results, failures = [None] * len(rows), 0
    try:
        with journal.open('x', encoding='utf-8') as stream:
            for completed, result in enumerate(events, 1):
                prediction = result['prediction']
                index = prediction['row_index']
                if not 0 <= index < len(rows) or results[index] is not None:
                    raise RuntimeError(f'duplicate or invalid result index: {index}')
                results[index] = result
                failures += prediction['status'] == 'error'
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                print(f'[{completed}/{len(rows)}] row={index} '
                      f'{result.get("source_dataset")}:{result.get("source_id")} {prediction["status"]}', flush=True)
    finally:
        events.close()
    if any(result is None for result in results):
        raise RuntimeError('workers did not return all samples; inspect progress JSONL')
    atomic_write(output, results, jsonl=output.suffix == '.jsonl')
    print(f'Output: {output}; rows={len(rows)}; errors={failures}', flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
