#!/usr/bin/env python3
"""Pinned Response-G1 reactive inference on unified JSON/JSONL records."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

from core import HERE, build_core, verify_upstream
sys.path.insert(0, str(HERE / 'scoring_reference'))
from qwen_eval_utils import atomic_write, options, read_rows, seconds


from prompt_templates import dialogue_prompt, make_prompt, prompt_version


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(row, args):
    prior_eval = row.get('unified_eval') or {}
    protocol = row.get('evaluation_protocol', prior_eval.get('protocol', 'backward') if isinstance(prior_eval, dict) else 'backward')
    if protocol not in ('backward', 'reactive'):
        raise ValueError('Unified adapter accepts backward/reactive only; proactive needs a native benchmark protocol')
    q = seconds(row.get('query_time', row.get('Question time')))
    if args.profile == 'streamingbench' and (not q.is_integer() or q >= 86400):
        raise ValueError('Official StreamingBench timestamp parser requires integer seconds below 24 hours; use ovobench for fractional times')
    question = row.get('question', row.get('Question'))
    if not isinstance(question, str) or not question.strip():
        raise ValueError('question must be a nonempty string')
    raw_options = row.get('options', row.get('Options'))
    if raw_options is not None and (not isinstance(raw_options, list) or not all(isinstance(x, str) for x in raw_options)):
        raise ValueError('options must be a list of strings')
    choices = options({'options': raw_options})
    multi = row.get('QA_type') == 'mc_multi' or row.get('question_type') == 'multiple-select'
    if len(choices) > 26:
        raise ValueError('at most 26 choices supported')
    path = row.get('video_path')
    if not isinstance(path, str) or not path:
        raise ValueError('video_path must be a nonempty string')
    for old, new in args.path_prefix:
        old = old.rstrip('/')
        if path == old or path.startswith(old + '/'):
            path = new.rstrip('/') + path[len(old):]
            break
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = Path(args.video_root).expanduser() / path
    return dict(video_path=str(path.resolve()), question=question, options=choices, realtime=q,
                time_stamp=f'{int(q)//3600:02}:{int(q)%3600//60:02}:{int(q)%60:02}', task='STU',
                prompt_adapted=True, multi=multi, question_type=row.get('question_type'),
                prompt=make_prompt(row, 0, q))


def format_prompt(plan, graphs, ns):
    text = ''
    if graphs is not None:
        text = 'The relative scene graph shows:\n'
        for timestamp, graph in graphs:
            text += f'<{timestamp:.1f} seconds>:' + ns['format_scene_graphs_for_prompt_offline']([graph]) + '\n'
    text += ('\n' if text else '') + plan['prompt']
    return text


class Predictor:
    def __init__(self, args):
        import torch
        import transformers
        if transformers.__version__ != '4.57.1':
            raise RuntimeError('This snapshot requires transformers==4.57.1')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required for inference; use --dry-run for CPU validation')
        self.torch, self.args = torch, args
        self.ns = build_core(args.profile, args.model)
        self.original_prompt = self.ns['build_prompt']
        torch.manual_seed(1234)
        path = str(Path(args.model_path).expanduser().resolve())
        cfg = json.loads((Path(path) / 'config.json').read_text())
        expected = 'qwen3_vl' if args.model == 'qwen3' else 'qwen2_5_vl'
        if cfg.get('model_type') != expected:
            raise ValueError(f'Expected dense {expected} checkpoint')
        cls = transformers.Qwen3VLForConditionalGeneration if args.model == 'qwen3' else transformers.Qwen2_5_VLForConditionalGeneration
        self.model = cls.from_pretrained(path, dtype=torch.bfloat16, attn_implementation=args.attn_implementation,
                                         device_map='auto', local_files_only=True).eval()
        self.processor = transformers.AutoProcessor.from_pretrained(path, min_pixels=args.min_pixels,
                                      max_pixels=args.max_pixels, local_files_only=True)

    def predict(self, plan, effective_interval):
        ns = self.ns
        ns['build_prompt'] = self.original_prompt
        if plan['prompt_adapted']:
            if self.args.profile == 'ovobench':
                ns['build_prompt'] = lambda *a, **kw: format_prompt(plan, kw['context_Scene_Graphs'], ns)
            else:
                ns['build_prompt'] = lambda question, options, graphs: format_prompt(plan, graphs, ns)
        call = {k: plan[k] for k in ('video_path', 'question', 'options', 'realtime', 'time_stamp', 'task')}
        if self.args.profile == 'streamingbench':
            # Upstream eval(options) receives only our serialized list, never input code.
            call['options'] = repr([f'{chr(65+i)}. {v}' for i, v in enumerate(plan['options'])])
        original_extract = ns['extract_frame']
        attempts = []
        cap = None
        cap_path = None
        def extract(path, timestamp):
            nonlocal cap, cap_path
            # Reuse one decoder for all sampled timestamps in this video. The original
            # implementation opened and closed the same file up to ~500 times.
            if 'cv2' not in ns or 'Image' not in ns:
                frame = original_extract(path, timestamp)
            else:
                if cap is None or cap_path != path:
                    if cap is not None:
                        cap.release()
                    cap = ns['cv2'].VideoCapture(path)
                    cap_path = path
                frame = None
                if cap.isOpened() and timestamp >= 0:
                    video_fps = cap.get(ns['cv2'].CAP_PROP_FPS)
                    cap.set(ns['cv2'].CAP_PROP_POS_FRAMES, int(timestamp * video_fps))
                    ok, decoded = cap.read()
                    if ok:
                        rgb = ns['cv2'].cvtColor(decoded, ns['cv2'].COLOR_BGR2RGB)
                        frame = ns['Image'].fromarray(rgb)
            attempts.append({'requested_seconds': timestamp, 'decoded': frame is not None})
            return frame
        ns['extract_frame'] = extract
        try:
            with self.torch.no_grad():
                result = ns['infer'](SimpleNamespace(fps=self.args.fps, frame_interval=effective_interval,
                                                     vision_batch_size=self.args.vision_batch_size),
                                    self.model, self.processor, **call)
            result['frame_attempts'] = attempts
            return result
        finally:
            if cap is not None:
                cap.release()
            ns['extract_frame'] = original_extract


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True, help='JSONL results; original fields + response_g1_prediction')
    p.add_argument('--model', choices=['qwen3', 'qwen25'], default='qwen3')
    p.add_argument('--model-path', required=True)
    p.add_argument('--profile', choices=['ovobench', 'streamingbench'], default='ovobench')
    p.add_argument('--attn-implementation', choices=['flash_attention_2', 'sdpa'], default='flash_attention_2')
    p.add_argument('--fps', type=int, default=1)
    p.add_argument('--frame-interval', type=int, default=1)
    p.add_argument('--min-pixels', type=int, default=448*448)
    p.add_argument('--max-pixels', type=int, default=448*448)
    p.add_argument('--vision-batch-size', type=int, default=8,
                   help='number of sampled frames encoded together; lower this on CUDA OOM')
    p.add_argument('--shard-strategy', choices=['balanced', 'round_robin'], default='balanced',
                   help='balanced greedily equalizes estimated sampled frames across workers')
    p.add_argument('--fsync-interval', type=int, default=32,
                   help='durably sync output every N rows; 0 syncs only at normal completion')
    p.add_argument('--video-root', default='.')
    p.add_argument('--path-prefix', nargs=2, action='append', default=[], metavar=('OLD', 'NEW'))
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)
    if min(args.fps, args.frame_interval, args.min_pixels, args.num_shards, args.vision_batch_size) <= 0 or args.max_pixels < args.min_pixels:
        p.error('Invalid sampling/pixel/shard parameters')
    if args.limit < 0 or args.fsync_interval < 0 or not 0 <= args.shard_index < args.num_shards:
        p.error('Invalid limit/shard index')
    if Path(args.output).suffix.lower() not in ('.jsonl', '.ndjson'):
        p.error('Output must use .jsonl or .ndjson')
    if Path(args.input).expanduser().resolve() == Path(args.output).expanduser().resolve():
        p.error('Input and output must differ')
    return args


def run(args):
    upstream = verify_upstream()
    rows = read_rows(Path(args.input).expanduser())
    if args.limit:
        rows = rows[:args.limit]
    # Compute StreamingBench's stateful interval in original input order BEFORE sharding/resume.
    plans, interval = [], args.frame_interval
    for index, row in enumerate(rows):
        try:
            plan = prepare(row, args)
            end = plan['realtime'] * args.fps
            if args.profile == 'streamingbench':
                if 300 < end <= 600: interval = 2
                elif end > 600: interval = 5
            effective = (int(end)+500)//500 if args.profile == 'ovobench' else interval
            plans.append((index, plan, effective, None))
        except Exception as exc:
            plans.append((index, None, interval, str(exc)))
    if args.shard_strategy == 'round_robin':
        selected = [p for p in plans if p[0] % args.num_shards == args.shard_index]
    else:
        # Long videos cost approximately in proportion to sampled frames. Greedy LPT
        # assignment avoids leaving one GPU with a much longer tail than its peers.
        assignments, loads = {}, [0] * args.num_shards
        weighted = []
        for item in plans:
            index, plan, effective, error = item
            weight = 0 if error else int(plan['realtime'] * args.fps) // effective + 1
            weighted.append((weight, index, item))
        for weight, index, item in sorted(weighted, key=lambda x: (-x[0], x[1])):
            shard = min(range(args.num_shards), key=lambda i: (loads[i], i))
            assignments[index] = shard
            loads[shard] += weight
        selected = [p for p in plans if assignments[p[0]] == args.shard_index]
    out = Path(args.output).expanduser().resolve()
    manifest_path = Path(str(out)+'.manifest.json')
    config = {k:v for k,v in vars(args).items() if k not in ('resume','output')}
    config['model_path'] = str(Path(args.model_path).expanduser().resolve())
    versions = {}
    if not args.dry_run:
        for package in ('torch','transformers','accelerate','flash-attn','opencv-python','Pillow','qwen-vl-utils'):
            try: versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError: versions[package] = None
    manifest = dict(prompt_version=prompt_version(), config=config, input_sha256=digest(Path(args.input).expanduser()), upstream_commit=upstream['commit'],
                    adapter_sha256={p.name:digest(p) for p in HERE.glob('*.py')},
                    scoring_sha256={p.name:digest(p) for p in (HERE/'scoring_reference').glob('*.py')}, versions=versions)
    done = {}
    if out.exists() or manifest_path.exists():
        if not args.resume:
            raise ValueError('Output/manifest exists; use --resume with identical configuration or a new output')
        if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Resume manifest mismatch')
        if out.exists():
            for record in read_rows(out):
                pred = record['response_g1_prediction']
                done[pred['input_index']] = record
    atomic_write(manifest_path, manifest)
    engine = None
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('a', encoding='utf-8') as stream:
        rows_since_sync = 0
        for index, plan, effective, error in selected:
            if index in done and done[index]['response_g1_prediction']['status'] in ('completed','dry_run'):
                continue
            started = time.monotonic()
            initializing = False
            pred = dict(input_index=index, status='error', model=args.model, profile=args.profile,
                        prompt_version=prompt_version())
            try:
                if error: raise ValueError(error)
                pred.update(video_path=plan['video_path'], window_start=0, window_end=plan['realtime'],
                            prompt_adapted=plan['prompt_adapted'], model_adapted=args.model=='qwen25',
                            frame_interval=effective)
                if args.dry_run:
                    pred.update(status='dry_run', video_exists=Path(plan['video_path']).is_file(),
                                scheduled_frames=int(plan['realtime']*args.fps)//effective+1)
                    if plan['prompt_adapted']:
                        pred['prompt'] = format_prompt(plan, None, {})
                else:
                    if not Path(plan['video_path']).is_file(): raise FileNotFoundError(plan['video_path'])
                    if engine is None:
                        # Initialization errors are fatal, not repeated for every sample.
                        initializing = True
                        engine = Predictor(args)
                        initializing = False
                    pred.update(engine.predict(plan, effective), status='completed')
            except Exception as exc:
                pred.update(error=str(exc), traceback=traceback.format_exc())
            pred['elapsed_seconds'] = time.monotonic()-started
            record = dict(rows[index], response_g1_prediction=pred)
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
            stream.flush()
            rows_since_sync += 1
            if args.fsync_interval and rows_since_sync >= args.fsync_interval:
                os.fsync(stream.fileno())
                rows_since_sync = 0
            done[index] = record
            print(f'[{index+1}/{len(rows)}] {pred["status"]}', flush=True)
            if pred['status']=='error': print(pred['error'], file=sys.stderr, flush=True)
            if initializing:
                raise RuntimeError('Model initialization failed; see saved traceback in output')
        if rows_since_sync:
            os.fsync(stream.fileno())
    # Remove retry history from final result while preserving original row order.
    atomic_write(out, [done[i] for i, *_ in selected], jsonl=True)
    failed = sum(done[i]['response_g1_prediction']['status']=='error' for i, *_ in selected)
    atomic_write(str(out)+'.summary.json', dict(rows=len(selected), failed=failed, completed=len(selected)-failed,
                                                dry_run=args.dry_run))
    return int(failed > 0)


if __name__ == '__main__':
    raise SystemExit(run(parse_args()))
