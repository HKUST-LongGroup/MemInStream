"""HERMES BaseVQA sampling + HermesVQA query prefix, without Qwen preprocessing.

Indices match BaseVQA.load_video/load_video_frames followed by
video[:ceil(end_time * sample_fps)]. Decode only that prefix to avoid loading
unused future frames. The official rounded-FPS/index convention is preserved.
"""
import math
from pathlib import Path
from prompt_templates import seconds


def query_window(row):
    query = seconds(row.get('query_time', row.get('Question time')))
    if query <= 0:
        raise ValueError('query_time must be positive')
    return 0.0, query


def sample_indices(total_frames, native_fps, sample_fps, query_time, directory=False):
    fps = native_fps if directory else round(native_fps)
    step = max(1, int(fps / sample_fps)) if directory else int(fps / sample_fps)
    if step <= 0:
        raise ValueError('Official sampler requires sample_fps <= rounded video FPS')
    return list(range(0, total_frames, step))[:math.ceil(query_time * sample_fps)]


def load_query_video(path, query_time, args, row):
    import numpy as np
    path = Path(path)
    limit = math.ceil(query_time * args.fps)
    native_fps = None
    if path.suffix == '.npy':
        video = np.load(path, mmap_mode='r')
        total = len(video)
        indices = np.linspace(0, total - 1, int(total * args.fps), dtype=int).tolist()[:limit]
        getter = lambda: np.asarray(video[indices]).copy()
    elif path.is_dir():
        from PIL import Image
        native_fps = float(row['fps'])
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
        indices = sample_indices(len(files), native_fps, args.fps, query_time, directory=True)
        def getter():
            frames = []
            for i in indices:
                with Image.open(files[i]) as im:
                    frames.append(np.array(im.convert('RGB')))
            return np.stack(frames)
    else:
        from decord import VideoReader
        vr = VideoReader(str(path), num_threads=1)
        native_fps = vr.get_avg_fps()
        indices = sample_indices(len(vr), native_fps, args.fps, query_time)
        getter = lambda: vr.get_batch(indices).asnumpy()
    if not indices:
        raise ValueError('No sampled frames before the query index')
    if args.max_frames and len(indices) > args.max_frames:
        raise ValueError('Frame safety limit exceeded; --max-frames 0 disables it (official default)')
    frames = getter()
    return frames, dict(sampling='official_hermes_index_prefix', frame_indices=indices,
                        native_fps=native_fps, decoded_frames=len(frames), input_frames=len(frames),
                        padding_frames=0, height=frames.shape[1], width=frames.shape[2])
