"""Qwen-aligned FFmpeg sampling, extracted from the user's 2026-10-20 runner."""
import math
import subprocess
from pathlib import Path
from qwen_eval_utils import seconds

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
    if count_bound + count_bound % 2 > args.max_frames:
        raise ValueError('frame budget exceeded; increase --max-frames or explicitly lower --fps / --history-seconds')
    width, height = frame_size(*metadata[:2], args.max_pixels, args.resize_factor)
    duration = end - start
    # Trim BEFORE resampling. Frames at/after the query boundary cannot enter the model.
    filters = (f'trim=duration={duration:.9f},setpts=PTS-STARTPTS,'
               f'fps={args.fps:.9f}:round=down,scale={width}:{height}')
    result = subprocess.run([
        'ffmpeg', '-nostdin', '-v', 'error', '-threads', str(args.ffmpeg_threads),
        '-ss', f'{start:.9f}', '-t', f'{duration:.9f}', '-i', str(path),
        '-map', '0:v:0', '-an', '-sn', '-dn', '-vf', filters,
        '-filter_threads', str(args.ffmpeg_threads), '-threads', str(args.ffmpeg_threads),
        '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1',
    ], capture_output=True, timeout=args.video_timeout)
    if result.returncode:
        raise ValueError(result.stderr.decode(errors='replace')[-2000:])
    if not result.stdout:
        raise ValueError('no frames in requested window; increase FPS or history duration')
    frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, height, width, 3).copy()
    decoded = len(frames)
    if decoded + decoded % 2 > args.max_frames:
        raise ValueError('decoded video exceeds --max-frames')
    if decoded % 2:
        frames = np.concatenate([frames, frames[-1:]], axis=0)
    return frames, {'decoded_frames': decoded, 'input_frames': len(frames),
                    'padding_frames': decoded % 2, 'width': width, 'height': height}


def resolve_video(value, args):
    path = str(value)
    for old, new in args.path_prefix:
        if path == old or path.startswith(old.rstrip('/') + '/'):
            path = new.rstrip('/') + path[len(old.rstrip('/')):]
            break
    path = Path(path).expanduser()
    return path if path.is_absolute() else Path(args.video_root).expanduser() / path
