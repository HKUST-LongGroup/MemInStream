#!/usr/bin/env python3
"""Run TimeChat-Online DTD on compact unified benchmark samples."""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
from multiprocessing.connection import wait
import subprocess
import tempfile
import sys
import hashlib
import functools
import os
import inspect
import importlib.metadata
from pathlib import Path

from qwen_eval_utils import atomic_write, options, probe_video, read_rows, seconds


from prompt_templates import dialogue_prompt, make_prompt, prompt_version


def window(row, history_seconds, duration=None):
    query = seconds(row.get('query_time', row.get('Question time')))
    start = max(0.0, query - history_seconds) if history_seconds else 0.0
    end = min(query, duration) if duration is not None else query
    if end <= start:
        raise ValueError(f'empty video window [{start}, {end}); query_time={query}')
    return start, end


HERE = Path(__file__).resolve().parent


@functools.lru_cache(maxsize=1)
def verify_upstream():
    manifest = json.loads((HERE / 'upstream_manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        actual = hashlib.sha256((HERE / 'upstream' / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Official source changed: {relative}')
    return manifest['commit']


def sampling_fps(args, query_time):
    if args.fps is not None:
        return args.fps
    return 1.0 if query_time <= 300 else 0.5 if query_time <= 600 else 0.2


def build_messages(prompt, video, args, fps):
    messages = [{'role': 'user', 'content': [
        {'type': 'video', 'video': str(video), 'min_pixels': args.min_pixels,
         'max_pixels': args.max_pixels, 'min_frames': args.min_frames,
         'max_frames': args.max_frames, 'fps': fps},
        {'type': 'text', 'text': prompt}]}]
    if args.system_prompt == 'qwen':
        messages.insert(0, {'role': 'system', 'content':
            'Answer the question using the supplied video and any provided dialogue history.'})
    return messages


def clip_video(path, start, end, target, args):
    # Re-encode a bounded clip BEFORE official sampling. Do not pass the full
    # source to qwen-vl-utils: 0.0.10 does not implement arbitrary clip windows.
    duration = end - start
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-threads', str(args.ffmpeg_threads),
        '-ss', f'{start:.9f}', '-i', str(path), '-t', f'{duration:.9f}',
        '-map', '0:v:0', '-an', '-sn', '-dn',
        '-vf', f'trim=duration={duration:.9f},setpts=PTS-STARTPTS',
        '-filter_threads', str(args.ffmpeg_threads), '-c:v', 'libx264',
        '-threads', str(args.ffmpeg_threads), '-y', str(target)]
    result = subprocess.run(command, capture_output=True, timeout=args.video_timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace')[-2000:])
    if not target.is_file() or not target.stat().st_size:
        raise ValueError('empty clipped video')


class Predictor:
    def __init__(self, args, device_index=None):
        import torch
        from transformers import AutoProcessor
        from qwen_vl_utils import process_vision_info
        self.commit = verify_upstream()
        for package, expected in [('transformers', '4.50.0'), ('qwen-vl-utils', '0.0.10'),
                                  ('accelerate', '1.5.2')]:
            actual = importlib.metadata.version(package)
            if actual != expected:
                raise RuntimeError(f'{package}=={expected} required, found {actual}; use setup.sh')
        root = HERE / 'upstream' / 'eval'
        sys.path.insert(0, str(root))
        # Deliberately NOT transformers.Qwen2_5_VLForConditionalGeneration.
        from qwen2_5_vl.modeling_qwen2_5_vl_DTD import Qwen2_5_VLForConditionalGeneration
        if not Path(inspect.getfile(Qwen2_5_VLForConditionalGeneration)).resolve().is_relative_to(root):
            raise RuntimeError('Wrong TimeChat model module imported')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; use --dry-run for CPU-only validation')
        if device_index is not None:
            torch.cuda.set_device(device_index)
        torch.manual_seed(1234)
        path = Path(args.model_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f'Checkpoint missing: {path}; run download_model.py')
        if json.loads((path / 'config.json').read_text()).get('model_type') != 'qwen2_5_vl':
            raise ValueError('Expected TimeChatOnline-7B (qwen2_5_vl) checkpoint')
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(path), local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
            device_map=args.device_map if device_index is None else {'': device_index}).eval()
        self.processor = AutoProcessor.from_pretrained(str(path), local_files_only=True,
            min_pixels=args.min_pixels, max_pixels=args.max_pixels)
        self.args, self.torch, self.process_vision_info = args, torch, process_vision_info
        self.generation_config = self.model.generation_config.to_dict()
        provenance = path / 'timechat_download.json'
        self.model_provenance = (json.loads(provenance.read_text()) if provenance.exists()
                                 else {'local_path': str(path), 'revision': 'unknown'})

    def generate(self, prompt, plan):
        args = self.args
        # Temporary clips are per sample/worker and always removed, even on OOM.
        with tempfile.TemporaryDirectory(prefix='timechat-', dir=args.temp_dir) as temp:
            clip = Path(temp) / 'window.mp4'
            clip_video(plan['path'], plan['start'], plan['end'], clip, args)
            messages = build_messages(prompt, clip, args, plan['fps'])
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos = self.process_vision_info(messages)
            inputs = self.processor(text=[text], images=images, videos=videos,
                                    padding=True, return_tensors='pt')
            # Official eval does not forward fps/video_kwargs to the processor.
            # Preserve this behavior (processor's default temporal scale).
            input_tokens = inputs.input_ids.shape[1]
            diagnostics = {'input_tokens': input_tokens,
                'input_frames': int(videos[0].shape[0]),
                'video_grid_thw': inputs.video_grid_thw.tolist(),
                'generation_config': self.generation_config,
                'model_provenance': self.model_provenance,
                'official_commit': self.commit}
            inputs = inputs.to(self.model.device)
            with self.torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                    drop_method=args.drop_method, drop_threshold=args.drop_threshold,
                    drop_absolute=not args.drop_relative, dr_save_path=plan['drop_path'])
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output)]
            answer = self.processor.batch_decode(trimmed, skip_special_tokens=True,
                clean_up_tokenization_spaces=False)[0].strip()
            if not answer:
                raise ValueError('model returned an empty answer')
            return answer, diagnostics


def resolve_video(value, args):
    path = str(value)
    for old, new in args.path_prefix:
        if path == old or path.startswith(old.rstrip('/') + '/'):
            path = new.rstrip('/') + path[len(old.rstrip('/')):]
            break
    path = Path(path).expanduser()
    return path if path.is_absolute() else Path(args.video_root).expanduser() / path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument('--input', required=True, help='compact JSON array or JSONL')
    p.add_argument('--output', required=True, help='.json (array + progress JSONL) or .jsonl')
    p.add_argument('--model-path', required=True, help='local checkpoint directory')
    p.add_argument('--history-seconds', type=float, default=0, help='seconds before query_time; 0 = all past video')
    p.add_argument('--fps', type=float, default=None, help='override official adaptive FPS (1 / 0.5 / 0.2 by query time)')
    p.add_argument('--video-root', default='.', help='base directory for relative video paths')
    p.add_argument('--path-prefix', nargs=2, action='append', default=[], metavar=('OLD', 'NEW'))
    p.add_argument('--max-frames', type=int, default=1016, help='official sampling cap; qwen-vl-utils uniformly resamples within budget')
    p.add_argument('--max-pixels', type=int, default=448*448)
    p.add_argument('--max-new-tokens', type=int, default=128)
    p.add_argument('--num-gpus', type=int, default=1, help='>1: one model replica per visible GPU with dynamic sample scheduling')
    p.add_argument('--device-map', default='auto', help='auto permits sharding across visible GPUs')
    p.add_argument('--attn-implementation', choices=['sdpa', 'flash_attention_2', 'eager'], default='flash_attention_2')
    p.add_argument('--video-timeout', type=float, default=600)
    p.add_argument('--ffmpeg-threads', type=int, default=2)
    p.add_argument('--limit', type=int, default=0, help='0 = all supplied samples, with no additional filtering')
    p.add_argument('--dry-run', action='store_true', help='validate rows/windows/prompts without model or videos')
    p.add_argument('--min-pixels', type=int, default=448*448)
    p.add_argument('--min-frames', type=int, default=4)
    p.add_argument('--drop-method', choices=['feature', 'pixel', 'none'], default='feature')
    p.add_argument('--drop-threshold', type=float, default=0.5)
    p.add_argument('--drop-relative', action='store_true')
    p.add_argument('--system-prompt', choices=['official', 'qwen'], default='official',
                   help='official/FluxMem user-only template, or exact Qwen system message')
    p.add_argument('--temp-dir', help='existing directory for temporary clips (default system temp)')
    p.add_argument('--resume', action='store_true', help='Continue the same output from its progress journal')
    args = p.parse_args(argv)
    for key in ['min_frames', 'min_pixels', 'max_frames', 'max_pixels', 'max_new_tokens', 'video_timeout', 'ffmpeg_threads']:
        value = getattr(args, key)
        if not math.isfinite(value) or value <= 0:
            p.error(f'--{key.replace("_", "-")} must be finite and positive')
    if not math.isfinite(args.history_seconds) or args.history_seconds < 0 or args.limit < 0:
        p.error('--history-seconds and --limit must be nonnegative')
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        p.error('--fps must be finite and positive')
    if not math.isfinite(args.drop_threshold):
        p.error('--drop-threshold must be finite')
    lower = -1 if args.drop_method == 'feature' and not args.drop_relative else 0
    if args.drop_method != 'none' and not lower <= args.drop_threshold <= 1:
        p.error('--drop-threshold outside method range')
    if args.min_frames > args.max_frames or args.min_pixels > args.max_pixels:
        p.error('min-frames/min-pixels must not exceed corresponding maximum')
    if args.min_frames % 2 or args.max_frames % 2:
        p.error('frame limits must be multiples of temporal patch size 2')
    if args.num_gpus < 1:
        p.error('--num-gpus must be >=1')
    if args.num_gpus > 1 and args.device_map != 'auto':
        p.error('--device-map is only configurable with --num-gpus 1; parallel workers use one GPU each')
    if args.max_pixels < 32 * 32 or args.max_frames < 2:
        p.error('--max-pixels must be >=1024 and --max-frames >=2')
    if Path(args.output).suffix not in {'.json', '.jsonl'}:
        p.error('--output must end in .json or .jsonl')
    return args


def predict_row(index, row, args, engine, worker_index=None):
    result = dict(row)
    config = {k: v for k, v in vars(args).items() if k not in {'input', 'output', 'limit', 'resume'}}
    prediction = {'status': 'pending', 'row_index': index, 'config': config,
                  'prompt_version': prompt_version()}
    if worker_index is not None:
        prediction.update(worker_index=worker_index,
                          cuda_device=None if args.dry_run else worker_index)
    result['prediction'] = prediction
    try:
        if row.get('evaluation_protocol', 'backward') != 'backward':
            raise ValueError('Only backward query-time QA is supported')
        path = resolve_video(row['video_path'], args)
        start, end = window(row, args.history_seconds)
        if not args.dry_run:
            metadata = probe_video(str(path), args.video_timeout)
            start, end = window(row, args.history_seconds, metadata[2])
        prompt = make_prompt(row, start, end)
        prediction.update(video_path=str(path), window_start=start, window_end=end,
                          requested_fps=sampling_fps(args, seconds(row.get('query_time', row.get('Question time')))),
                          prompt=prompt, official_commit=verify_upstream())
        if args.dry_run:
            prediction.update(status='dry_run', video_exists=path.is_file())
        else:
            drop_dir = Path(args.output).expanduser().resolve().with_suffix('.drop')
            drop_dir.mkdir(parents=True, exist_ok=True)
            drop_path = drop_dir / f'{index:08d}.jsonl'
            # An interrupted generation may have left DTD diagnostics without a result.
            drop_path.unlink(missing_ok=True)
            answer, stats = engine.generate(prompt, dict(path=path, start=start, end=end,
                fps=prediction['requested_fps'], drop_path=str(drop_path)))
            prediction.update(stats)
            prediction.update(status='completed', answer=answer, drop_stats_path=str(drop_path))
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


def parallel_results(rows, args, worker_target=gpu_worker, indices=None):
    """Keep one outstanding sample per worker; yield completed rows immediately.

    Pipes bound queued work; process sentinels detect crashes without waiting
    forever for a result. Only the parent process writes output files.
    """
    if not rows:
        return
    indices = list(range(len(rows))) if indices is None else indices
    ctx = mp.get_context('spawn')
    workers, active, assigned = [], {}, {}
    next_index = 0
    try:
        for rank in range(min(args.num_gpus, len(rows))):
            parent, child = ctx.Pipe()
            process = ctx.Process(target=worker_target, args=(child, args, rank), name=f'timechat-gpu-{rank}')
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
                    assigned[connection] = indices[next_index]
                    connection.send((indices[next_index], rows[next_index]))
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


def resume_config(config):
    # Paths for temporary clips and worker placement do not define the sample prompt.
    return {k: v for k, v in config.items()
            if k not in {'input', 'output', 'limit', 'resume', 'temp_dir'}}


def read_progress(journal, rows, args):
    results = [None] * len(rows)
    raw = journal.read_bytes()
    lines = raw.splitlines(keepends=True)
    valid_bytes = 0
    for n, line in enumerate(lines):
        try:
            result = json.loads(line)
        except (ValueError, UnicodeError):
            if n == len(lines) - 1 and not line.endswith(b'\n'):
                # Only an incomplete final line is recoverable, never interior corruption.
                break
            raise ValueError(f'Corrupt progress JSON at line {n + 1}')
        pred = result.get('prediction', {})
        index = pred.get('row_index')
        if type(index) is not int or not 0 <= index < len(rows) or results[index] is not None:
            raise ValueError(f'Invalid/duplicate progress row index at line {n + 1}')
        if {k: v for k, v in result.items() if k != 'prediction'} != {
                k: v for k, v in rows[index].items() if k != 'prediction'}:
            raise ValueError(f'Input changed at row {index}; use original input to resume')
        if resume_config(pred.get('config', {})) != resume_config(vars(args)):
            raise ValueError(f'Prediction configuration changed at row {index}')
        if pred.get('official_commit') not in (None, verify_upstream()):
            raise ValueError('Official source revision changed')
        allowed = {'dry_run', 'error'} if args.dry_run else {'completed', 'error'}
        if pred.get('status') not in allowed:
            raise ValueError(f'Incompatible progress status at row {index}')
        results[index] = result
        valid_bytes += len(line)
    return results, raw, valid_bytes


def main(argv=None):
    args = parse_args(argv)
    verify_upstream()
    output = Path(args.output).expanduser().resolve()
    journal = output.with_suffix('.progress.jsonl')
    rows = read_rows(Path(args.input).expanduser())
    if args.limit:
        rows = rows[:args.limit]
    if args.resume:
        if not journal.is_file():
            raise FileNotFoundError(f'Resume needs the original progress journal: {journal}')
        results, raw, valid_bytes = read_progress(journal, rows, args)
        if valid_bytes != len(raw):
            backup = journal.with_name(journal.name + '.interrupted-backup')
            with backup.open('xb') as f:
                f.write(raw)
            journal.write_bytes(raw[:valid_bytes])
            print(f'Recovered incomplete last line; backup: {backup}', flush=True)
        if valid_bytes and not raw[:valid_bytes].endswith(b'\n'):
            with journal.open('ab') as f:
                f.write(b'\n')
    else:
        if output.exists() or journal.exists() or output.with_suffix('.drop').exists():
            raise FileExistsError(f'output/progress exists; use --resume or a new --output: {output}')
        results = [None] * len(rows)
    indices = [i for i, result in enumerate(results) if result is None]
    pending = [rows[i] for i in indices]
    finished = len(rows) - len(pending)
    failures = sum(r is not None and r['prediction']['status'] == 'error' for r in results)
    print(f'Already recorded: {finished}; pending: {len(pending)}; recorded errors: {failures}', flush=True)
    if pending:
        validate_gpu_count(args)
    if args.num_gpus > 1:
        events = parallel_results(pending, args, indices=indices)
    else:
        engine = None if args.dry_run or not pending else Predictor(args)
        events = (predict_row(i, rows[i], args, engine) for i in indices)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with journal.open('a' if args.resume else 'x', encoding='utf-8') as stream:
            for completed, result in enumerate(events, finished + 1):
                prediction = result['prediction']
                index = prediction['row_index']
                if not 0 <= index < len(rows) or results[index] is not None:
                    raise RuntimeError(f'duplicate or invalid result index: {index}')
                results[index] = result
                failures += prediction['status'] == 'error'
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
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
