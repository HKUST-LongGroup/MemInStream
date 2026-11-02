#!/usr/bin/env python3
"""One model per GPU, independent resumable shards, ordered merge."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from core import HERE
sys.path.insert(0, str(HERE/'scoring_reference'))
from qwen_eval_utils import atomic_write, read_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus', required=True, help='physical CUDA ids, e.g. 0,1,2,3')
    p.add_argument('--output', required=True)
    a, rest = p.parse_known_args()
    if any(x.split('=')[0] in ('--num-shards','--shard-index','--output') for x in rest):
        p.error('Sharding/output options are controlled by this launcher')
    gpu_ids = a.gpus.split(',')
    if len(set(gpu_ids)) != len(gpu_ids) or not all(x.isdigit() for x in gpu_ids):
        p.error('gpus must be unique comma-separated nonnegative integers')
    from predict import parse_args
    checked = parse_args([*rest, '--output', a.output])
    out = Path(a.output).expanduser().resolve()
    if Path(checked.input).expanduser().resolve()==out:
        p.error('Input and output must differ')
    if out.exists() and '--resume' not in rest:
        p.error('Output exists; use --resume or a new output')
    folder = Path(str(out)+'.shards'); folder.mkdir(parents=True, exist_ok=True)
    processes, handles = [], []
    try:
        for i, gpu in enumerate(gpu_ids):
            # A startup failure must not reuse a completion summary from an older run.
            (folder/f'part{i}.jsonl.summary.json').unlink(missing_ok=True)
            log = (folder/f'part{i}.log').open('a'); handles.append(log)
            cmd = [sys.executable,str(HERE/'predict.py'),*rest,'--output',str(folder/f'part{i}.jsonl'),
                   '--num-shards',str(len(gpu_ids)),'--shard-index',str(i)]
            processes.append(subprocess.Popen(cmd, env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),
                                               stdout=log,stderr=subprocess.STDOUT))
        codes = [proc.wait() for proc in processes]
    finally:
        for proc in processes:
            if proc.poll() is None: proc.terminate()
        for proc in processes:
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()
        for log in handles: log.close()
    if any(code not in (0,1) for code in codes):
        raise RuntimeError(f'Worker startup/termination failure: {codes}; inspect {folder}')
    rows = {}
    expected = 0
    for i in range(len(gpu_ids)):
        summary = folder/f'part{i}.jsonl.summary.json'
        if not summary.exists(): raise RuntimeError(f'Missing worker completion summary: {summary}')
        expected += json.loads(summary.read_text())['rows']
        local = {}
        for row in read_rows(folder/f'part{i}.jsonl'):
            local[row['response_g1_prediction']['input_index']] = row
        if rows.keys() & local.keys(): raise ValueError('Duplicate indices across shards')
        rows.update(local)
    if len(rows)!=expected: raise ValueError('Incomplete merge')
    atomic_write(out,[rows[i] for i in sorted(rows)],jsonl=True)
    print(f'Merged {len(rows)} rows to {out}')
    return int(any(codes))


if __name__ == '__main__': raise SystemExit(main())
