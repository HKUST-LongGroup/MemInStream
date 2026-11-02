#!/usr/bin/env python3
"""Qwen3-VL-8B adaptation of the local FluxMem memory and evaluation pipeline."""
from __future__ import annotations

import argparse
import gc
from collections import Counter
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import math
import os
import re
import signal
import subprocess
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
EVALUATOR = "evaluation/ovobench/ovobench.py"
ADAPTER_VERSION = 9


from prompt_templates import dialogue_prompt, make_prompt, prompt_version


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify_upstream(root):
    manifest = json.loads((HERE / "upstream_manifest.json").read_text())
    for relative, expected in manifest["sha256"].items():
        path = root / relative
        if not path.is_file() or digest(path.read_bytes()) != expected:
            raise ValueError(f"Reference source verification failed: {path}. "
                             "Use the bundled local FluxMem snapshot or its identical source tree.")
    for directory in ("models/qwen2-5-vl/src", "models/qwen-vl-utils/src"):
        for path in (root / directory).rglob("*.py"):
            if path.relative_to(root).as_posix() not in manifest["sha256"]:
                raise ValueError(f"Unexpected Python source in reference package: {path}")
    # Also verify that the algorithm port and Qwen3 adapter match this release.
    adapter = json.loads((HERE / 'adapter_manifest.json').read_text())
    for relative, expected in adapter['sha256'].items():
        if digest((HERE / relative).read_bytes()) != expected:
            raise ValueError(f'Qwen3 adapter source changed: {relative}')
    return manifest


# Kept verbatim in sync with predict_filtered_qwen.make_prompt and its helpers.
def seconds(value):
    if value is None or isinstance(value, bool):
        raise ValueError(f"invalid time: {value!r}")
    parts = str(value).strip().replace("：", ":").split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"invalid time: {value!r}")
    number = sum(float(x) * (60 ** i) for i, x in enumerate(reversed(parts)))
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"invalid time: {value!r}")
    return float(number)


def options(row):
    return [re.sub(r"^[A-Z][.)]\s*", "", str(x)).strip() for x in row.get("options") or []]


def read_rows(path):
    with path.open(encoding="utf-8") as f:
        value = [json.loads(line) for line in f if line.strip()] if path.suffix.lower() in {
            ".jsonl", ".ndjson"} else json.load(f)
    if not isinstance(value, list) or not all(isinstance(r, dict) for r in value):
        raise ValueError("Input must be a JSON array or JSONL objects")
    return value


def remap_path(value, mappings):
    if not isinstance(value, str) or not value:
        raise ValueError("video_path must be a nonempty string")
    for old, new in sorted(mappings, key=lambda pair: len(pair[0]), reverse=True):
        old = old.rstrip("/")
        if value == old or value.startswith(old + "/"):
            value = new.rstrip("/") + value[len(old):]
            break
    return str(Path(value).expanduser().resolve())


def prepare_rows(rows, args):
    plans, seen = [], set()
    for index, row in enumerate(rows):
        if args.limit and index >= args.limit:
            break
        if index % args.num_shards != args.shard_index:
            continue
        try:
            identity = (str(row["source_dataset"]), str(row["source_id"]))
            if identity in seen:
                raise ValueError(f"duplicate source identity {identity}")
            seen.add(identity)
            if row.get("evaluation_protocol", "backward") != "backward":
                raise ValueError("This runner supports backward query-time QA only")
            q = seconds(row.get("query_time", row.get("Question time")))
            if q <= 0:
                raise ValueError("query_time must be positive finite clip-relative seconds")
            path = remap_path(row.get("video_path"), args.path_prefix)
            start = max(0.0, q - args.time_window) if args.time_window > 0 else 0.0
            prompt = make_prompt(row, start, q)
            plans.append(dict(input_index=index, source_dataset=identity[0], source_id=identity[1],
                              video_path=path, video_exists=Path(path).is_file(),
                              query_time=q, video_start=start, video_end=q,
                              anchor_end=args.time_window > 0, prompt=prompt,
                              prompt_version=prompt_version()))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid input row {index}: {exc}") from exc
    return plans


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def official_args(args, stats_path):
    return SimpleNamespace(ckpt_path=args.model_path, log_path=None,
                           attn_implementation=getattr(args, "attn_implementation", "flash_attention_2"),
                           min_pixels=args.min_pixels, max_pixels=args.max_pixels,
                           min_frames=args.min_frames, max_frames=args.max_frames,
                           fps=args.fps, time_window_size=args.time_window,
                           use_fluxmem=True, short_frames=args.short_frames,
                           medium_frames=args.medium_frames, video_chunk_t=args.video_chunk_t,
                           generation="Qwen3-VL FluxMem: max_new_tokens=128; num_beams=1; other sampling settings from checkpoint",
                           save_path=str(stats_path))


def load_official(root, args, stats_path):
    # Retain the caller interface; runtime now uses native Qwen3-VL plus FluxMem.
    from qwen3_fluxmem.runtime import Qwen3FluxMemPredictor
    return Qwen3FluxMemPredictor(args, stats_path)


def environment_versions(backend="sdpa"):
    versions = {}
    for name in ('torch', 'torchvision', 'transformers', 'accelerate', 'av',
                 'requests', 'tqdm', 'huggingface-hub', 'flash-attn'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    for name in ('torch', 'torchvision', 'transformers', 'accelerate', 'av', 'requests', 'tqdm', 'huggingface-hub'):
        if versions[name] is None:
            raise RuntimeError(f'Missing dependency {name}; install requirements.txt with this Python: {sys.executable}')
    if versions['transformers'] != '4.57.1':
        raise RuntimeError('Qwen3 FluxMem adapter requires transformers==4.57.1')
    if backend == 'flash_attention_2' and not (versions['flash-attn'] or '').startswith('2.'):
        raise RuntimeError('Install a matching flash-attn 2 wheel, or use --attn-implementation sdpa')
    return versions


def completed_indices(output, selected):
    done = set()
    if not output.exists():
        return done
    # Refuse malformed output instead of silently discarding possibly completed work.
    with output.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            index = record["fluxmem_prediction"]["input_index"]
            if index not in selected or index in done:
                raise ValueError(f"Unexpected/duplicate output index at line {line_no}: {index}")
            done.add(index)
    # A valid JSON record may survive interruption without its final newline.
    if output.stat().st_size:
        with output.open("rb+") as f:
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                f.seek(0, 2)
                f.write(b"\n")
    return done


def select_gpus(count, dry_run=False):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible is not None else [str(i) for i in range(count)]
    devices = [x.strip() for x in devices if x.strip()]
    if len(devices) < count or "-1" in devices or len(set(devices)) != len(devices):
        raise ValueError(f"--num-gpus {count} exceeds or conflicts with CUDA_VISIBLE_DEVICES={visible!r}")
    if not dry_run:
        import torch
        if torch.cuda.device_count() < count:
            raise RuntimeError(f"Requested {count} GPUs but PyTorch sees {torch.cuda.device_count()}")
    return devices[:count]


def worker_command(args, rank, output):
    command = [sys.executable, "-u", str(Path(__file__).resolve()),
               "--input", str(args.input.resolve()), "--output", str(output),

               "--fluxmem-root", str(args.fluxmem_root.resolve()), "--model-path", args.model_path,
               "--attn-implementation", getattr(args, "attn_implementation", "flash_attention_2"),
               "--num-shards", str(args.num_gpus), "--shard-index", str(rank)]
    for name in ("limit", "time_window", "fps", "min_frames", "max_frames", "min_pixels", "max_pixels",
                 "short_frames", "medium_frames", "max_consecutive_errors"):
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    for name in ("video_chunk_t", "cpu_threads"):
        if getattr(args, name, None) is not None:
            command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    for old, new in args.path_prefix:
        command.extend(["--path-prefix", f"{old}={new}"])
    return command


def merge_shards(output, plans, shard_outputs, num_gpus):
    expected = {plan["input_index"]: plan for plan in plans}
    records = {}
    for rank, path in shard_outputs.items():
        if not path.exists():
            continue
        # Also validates JSON and repairs a missing trailing newline after interruption.
        completed_indices(path, {i for i in expected if i % num_gpus == rank})
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                prediction = record["fluxmem_prediction"]
                index = prediction["input_index"]
                if index in records or prediction.get("status") != "completed":
                    raise ValueError(f"Invalid or duplicate shard prediction for index {index}")
                plan = expected[index]
                if any(str(record.get(k)) != plan[k] for k in ("source_dataset", "source_id")):
                    raise ValueError(f"Shard prediction identity does not match input index {index}")
                records[index] = record
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        for index in sorted(records):
            f.write(json.dumps(records[index], ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(output)
    return len(records)


def run_multi_gpu(args, output, plans, audit, devices):
    shard_dir = Path(str(output) + ".shards")
    shard_dir.mkdir(parents=True, exist_ok=True)
    owner_path = shard_dir / "launcher.manifest.json"
    owner = json.loads(Path(str(output) + ".manifest.json").read_text())
    if owner_path.exists():
        if json.loads(owner_path.read_text()) != owner:
            raise ValueError("Shard directory belongs to a different run; choose a new --output")
    elif any(shard_dir.glob("part*.jsonl*")):
        raise ValueError("Existing shard results lack launcher provenance; choose a new --output")
    else:
        write_json(owner_path, owner)
    active_ranks = sorted({plan["input_index"] % args.num_gpus for plan in plans})
    outputs = {rank: shard_dir / f"part{rank}.jsonl" for rank in active_ranks}
    processes, logs, returncodes = {}, [], {}
    try:
        for rank in active_ranks:
            log_path = shard_dir / f"part{rank}.log"
            log = log_path.open("a", encoding="utf-8")
            logs.append(log)
            log.write(f"\n--- launch {time.strftime('%Y-%m-%d %H:%M:%S')} GPU={devices[rank]} ---\n")
            log.flush()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[rank])
            processes[rank] = subprocess.Popen(worker_command(args, rank, outputs[rank]),
                env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            print(f"Started worker {rank} on GPU {devices[rank]}; log: {log_path}", flush=True)
        pending = set(processes)
        while pending:
            for rank in list(pending):
                code = processes[rank].poll()
                if code is not None:
                    returncodes[str(rank)] = code
                    pending.remove(rank)
                    print(f"Worker {rank} exited with code {code}", flush=True)
            if pending:
                time.sleep(1)
    finally:
        # Interrupting the parent must not leave GPU workers/decoder descendants running.
        for process in processes.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process in processes.values():
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        for log in logs:
            log.close()
    completed = merge_shards(output, plans, outputs, args.num_gpus)
    remaining = len(plans) - completed
    summary = dict(audit, completed=completed, remaining=remaining,
                   worker_exit_codes=returncodes, shard_directory=str(shard_dir))
    write_json(Path(str(output) + ".summary.json"), summary)
    print(f"Merged {completed}/{len(plans)} predictions in input order into {output}", flush=True)
    return 1 if remaining or any(returncodes.values()) else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="Resumable prediction JSONL; unique per shard/config")
    p.add_argument("--fluxmem-root", type=Path, default=HERE / "upstream")
    p.add_argument("--model-path", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--attn-implementation", choices=["flash_attention_2", "sdpa"],
                   default="sdpa", help="SDPA uses PyTorch and does not require flash-attn")
    p.add_argument("--limit", type=int, default=0, help="First N input rows; 0=all")
    p.add_argument("--dry-run", action="store_true", help="Audit data/source without GPU imports or decoding")
    p.add_argument("--path-prefix", action="append", default=[], metavar="OLD=NEW")
    # Match the user's local OVO shell recipe and Python preprocessing defaults.
    p.add_argument("--time-window", type=float, default=256, help="Lookback seconds; 0=official full-history branch")
    p.add_argument("--fps", type=float, default=1)
    p.add_argument("--min-frames", type=int, default=4)
    p.add_argument("--max-frames", type=int, default=256)
    p.add_argument("--min-pixels", type=int, default=16 * 32 * 32)
    p.add_argument("--max-pixels", type=int, default=512 * 32 * 32)
    p.add_argument("--short-frames", type=int, default=8)
    p.add_argument("--medium-frames", type=int, default=64)
    p.add_argument("--video-chunk-t", type=int, default=os.environ.get("VIDEO_CHUNK_T"),
                   help="Forward local FluxMem's visual encoder chunk size (temporal grids); unset=disabled")
    p.add_argument("--cpu-threads", type=int, default=None,
                   help="Optional PyTorch/BLAS thread limit; does not directly set PyAV decoder threads")
    p.add_argument("--max-consecutive-errors", type=int, default=5,
                   help="Stop this worker after N consecutive failures; 0=unlimited")
    p.add_argument("--num-gpus", type=int, default=1,
                   help="Launch N GPU workers and merge predictions automatically")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    args = p.parse_args(argv)
    if args.num_gpus < 1:
        p.error("num-gpus must be positive")
    if args.num_gpus > 1 and (args.num_shards != 1 or args.shard_index != 0):
        p.error("--num-gpus automatically shards data; do not combine with manual shard flags")
    if args.limit < 0 or args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        p.error("Invalid limit or shard selection")
    if not math.isfinite(args.time_window) or args.time_window < 0 or not math.isfinite(args.fps) or args.fps <= 0:
        p.error("time-window must be finite/nonnegative, fps finite/positive")
    if args.min_pixels <= 0 or args.max_pixels < args.min_pixels:
        p.error("Require 0 < min-pixels <= max-pixels")
    if args.max_consecutive_errors < 0:
        p.error("max-consecutive-errors must be nonnegative")
    for name in ("video_chunk_t", "cpu_threads"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            p.error(f"{name.replace('_', '-')} must be positive")
    if any(v < minimum or v % 2 for v, minimum in [
            (args.min_frames, 2), (args.max_frames, 2), (args.short_frames, 2), (args.medium_frames, 0)]):
        p.error("Frame counts must be even; min/max/short >= 2, medium >= 0")
    if args.min_frames > args.max_frames:
        p.error("min-frames cannot exceed max-frames")
    if args.cpu_threads is not None:
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = str(args.cpu_threads)
    mappings = []
    for value in args.path_prefix:
        if "=" not in value or not all(value.split("=", 1)):
            p.error("path-prefix must be OLD=NEW with nonempty paths")
        mappings.append(value.split("=", 1))
    args.path_prefix = mappings
    root, output = args.fluxmem_root.resolve(), args.output.resolve()
    if output == args.input.resolve():
        p.error("output must differ from input")
    source = verify_upstream(root)
    rows = read_rows(args.input)
    plans = prepare_rows(rows, args)
    audit = dict(model_family='qwen3_vl', input_rows=len(rows), selected_rows=len(plans),
                 source_dataset=dict(Counter(x["source_dataset"] for x in plans)),
                 prompt_version=prompt_version(),
                 missing_videos=sum(not x["video_exists"] for x in plans),
                 evidence_before_window=sum(any(float(a) < plan["video_start"] for a, b in
                                               (rows[plan["input_index"]].get("evidence") or [])) for plan in plans),
                 upstream_revision=source["revision"], official_args=vars(official_args(args, "memory.jsonl")))
    audit["source_kind"] = source["source_kind"]
    audit["source_snapshot_sha256"] = digest(json.dumps(source["sha256"], sort_keys=True).encode())
    devices = None
    if args.num_gpus > 1:
        devices = select_gpus(args.num_gpus, args.dry_run)
        counts = Counter(plan["input_index"] % args.num_gpus for plan in plans)
        audit["gpu_workers"] = [dict(rank=rank, gpu=device, samples=counts[rank])
                                for rank, device in enumerate(devices)]
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        write_json(Path(str(output) + ".audit.json"), audit)
        preview = Path(str(output) + ".preview.jsonl")
        preview.parent.mkdir(parents=True, exist_ok=True)
        preview.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in plans))
        return 0
    if not plans:
        raise ValueError("No selected rows")
    if audit["missing_videos"]:
        example = next(x["video_path"] for x in plans if not x["video_exists"])
        raise FileNotFoundError(f"{audit['missing_videos']} video paths unavailable; e.g. {example}. "
                                "Run where the videos exist or use --path-prefix OLD=NEW.")
    versions = environment_versions(args.attn_implementation)
    manifest = dict(model_family='qwen3_vl',
                    adapter_sources=json.loads((HERE/'adapter_manifest.json').read_text()),
                    prompt_version=prompt_version(),
                    prompt_sha256=digest((HERE / "prompt_templates.py").read_bytes()),
                    adapter_version=ADAPTER_VERSION, adapter_sha256=digest(Path(__file__).read_bytes()),
                    input_sha256=digest(args.input.read_bytes()), upstream=source,
                    inference=vars(official_args(args, "memory.jsonl")), versions=versions,
                    num_gpus=args.num_gpus,
                    cpu_threads=args.cpu_threads,
                    video_reader_override=os.environ.get("FORCE_QWENVL_VIDEO_READER"),
                    plans_sha256=digest(json.dumps(plans, sort_keys=True).encode()))
    manifest_path = Path(str(output) + ".manifest.json")
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Run settings/input/source/environment changed; choose a new --output")
    elif output.exists() and output.stat().st_size:
        raise ValueError("Existing predictions lack a manifest; choose a new --output")
    else:
        write_json(manifest_path, manifest)
    if args.num_gpus > 1:
        return run_multi_gpu(args, output, plans, audit, devices)
    done = completed_indices(output, {x["input_index"] for x in plans})
    pending = [x for x in plans if x["input_index"] not in done]
    if not pending:
        print(f"All {len(done)} selected predictions are already complete.", flush=True)
        return 0
    evaluator = load_official(root, args, Path(str(output) + ".memory.jsonl"))
    failures, consecutive_errors = 0, 0
    needs_cleanup = False
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as f:
        for attempt, plan in enumerate(pending, 1):
            if needs_cleanup:
                # Previous exception has left scope, so traceback-held tensors can be collected.
                evaluator.release_after_error()
                needs_cleanup = False
            started = time.monotonic()
            index = plan["input_index"]
            print(f"[attempt {attempt}/{len(pending)}; completed {len(done)}/{len(plans)}; failed {failures}] "
                  f"{plan['source_dataset']}:{plan['source_id']}", flush=True)
            try:
                response = evaluator.predict(plan)
            except Exception as exc:
                failures += 1
                consecutive_errors += 1
                needs_cleanup = True
                error = dict(input_index=index, source_dataset=plan["source_dataset"],
                             source_id=plan["source_id"], error=f"{type(exc).__name__}: {exc}",
                             traceback=traceback.format_exc(), diagnostics=evaluator.diagnostics)
                with Path(str(output) + ".errors.jsonl").open("a", encoding="utf-8") as ef:
                    ef.write(json.dumps(error, ensure_ascii=False) + "\n")
                print(error["traceback"], file=sys.stderr, flush=True)
                if args.max_consecutive_errors and consecutive_errors >= args.max_consecutive_errors:
                    print(f"Stopping worker after {consecutive_errors} consecutive failures. "
                          "Resolve the error and resume with the same command.", file=sys.stderr, flush=True)
                    break
                continue
            consecutive_errors = 0
            record = dict(rows[index])
            record["fluxmem_prediction"] = dict(plan, status="completed", response=response,
                wall_seconds=time.monotonic()-started, upstream_revision=source["revision"],
                source_snapshot_sha256=audit["source_snapshot_sha256"], diagnostics=evaluator.diagnostics,
                model=args.model_path, input_mode="query_time_window", audio_used=False)
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            done.add(index)
    write_json(Path(str(output) + ".summary.json"), dict(audit, completed=len(done),
               failed_this_run=failures, remaining=len(plans)-len(done)))
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; completed shard predictions are saved for resume.", file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
