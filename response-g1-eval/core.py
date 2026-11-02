"""Execute pinned upstream reactive statements without rewriting their inference logic."""
import ast
import hashlib
import importlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / 'upstream'


def encode_video_frames(args, model, processor, video_path, start_frame, end_frame,
                        memories, extract_frame, encode_images,
                        whole_current_frame_tokens, retry_decode=False):
    """Decode on the CPU, then encode independent frames in small GPU batches."""
    pending_frames, pending_timestamps = [], []
    last_frame_idx = start_frame
    last_timestamp = start_frame / args.fps

    def flush():
        if not pending_frames:
            return
        batch = processor.image_processor(pending_frames)
        pixel_values = batch['pixel_values'].to(model.device)
        image_grid_thw = batch['image_grid_thw'].to(model.device)
        frame_features, _, _ = encode_images(model, pixel_values, image_grid_thw)
        if len(frame_features) != len(pending_frames):
            raise RuntimeError(
                f'Vision encoder returned {len(frame_features)} frame features for '
                f'{len(pending_frames)} input frames')
        for timestamp, feature in zip(pending_timestamps, frame_features):
            # The upstream wrapper consumes a per-image tuple and adds timestamp tokens.
            tokens = whole_current_frame_tokens(processor, model, timestamp, (feature,))
            if tokens is not None:
                memories.update(tokens.unsqueeze(0).to(model.device))
        pending_frames.clear()
        pending_timestamps.clear()
        del batch, pixel_values, image_grid_thw, frame_features

    for frame_idx in range(start_frame, int(end_frame) + 1, args.frame_interval):
        last_frame_idx = frame_idx
        timestamp = frame_idx / args.fps
        last_timestamp = timestamp
        frame = extract_frame(video_path, timestamp)
        if frame is None and retry_decode:
            frame = extract_frame(video_path, timestamp - 1)
            if frame is None:
                frame = extract_frame(video_path, timestamp - 2)
        if frame is None:
            continue
        pending_frames.append(frame)
        pending_timestamps.append(timestamp)
        if len(pending_frames) >= args.vision_batch_size:
            flush()
    flush()
    return last_timestamp, last_frame_idx


def verify_upstream():
    manifest = json.loads((HERE / 'upstream_manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        if hashlib.sha256((UPSTREAM / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Upstream source changed: {name}')
    return manifest


def source_tree(profile):
    name = f'eval_{profile}_reactive.py'
    return ast.parse((UPSTREAM / name).read_text()), name


def inference_nodes(profile):
    tree, _ = source_tree(profile)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'eval_r')
    body = next(n for n in ast.walk(fn) if isinstance(n, ast.Try)).body
    # Exclude only benchmark row/path unpacking, scoring, and output writing.
    return body[2:34] if profile == 'ovobench' else [body[9], *body[13:44]]


def encode_qwen25(model, pixel_values, image_grid_thw):
    # Transformers 4.57.1 returns a tuple of per-image tensors (no DeepStack tuple).
    images = model.get_image_features(pixel_values, image_grid_thw)
    return images, None, images[0].shape[0]


def whole_qwen25_frame_tokens(processor, model, timestamp, frame_tokens):
    """Supply missing processor aliases, then execute the unchanged upstream wrapper."""
    aliases = {'vision_start_token': '<|vision_start|>',
               'vision_end_token': '<|vision_end|>'}
    missing = {name: token for name, token in aliases.items()
               if not getattr(processor, name, None)}
    if missing:
        vocab = processor.tokenizer.get_vocab()
        for token in missing.values():
            if token not in vocab:
                raise ValueError(f'Qwen2.5 tokenizer is missing required token {token}')
        for name, token in missing.items():
            setattr(processor, name, token)
    utils = importlib.import_module('qwen_online.utils')
    return utils.whole_current_frame_tokens(processor, model, timestamp, frame_tokens)


def build_core(profile, model_family):
    verify_upstream()
    import torch
    import cv2
    from PIL import Image
    from collections import deque
    from datetime import datetime
    sys.path.insert(0, str(UPSTREAM))
    utils = importlib.import_module('qwen_online.utils')
    graph_name = 'response_graph.Scene_Graph_ovo' if profile == 'ovobench' else 'response_graph.Scene_Graph'
    graphs = importlib.import_module(graph_name)
    ns = dict(torch=torch, cv2=cv2, Image=Image, deque=deque, datetime=datetime)
    ns.update({name: getattr(graphs, name) for name in dir(graphs) if not name.startswith('_')})
    ns.update(encode_images=utils.encode_images if model_family == 'qwen3' else encode_qwen25,
              whole_current_frame_tokens=(utils.whole_current_frame_tokens
                                          if model_family == 'qwen3' else whole_qwen25_frame_tokens),
              encode_video_frames=encode_video_frames)
    tree, name = source_tree(profile)
    names = {'Memory_Bank_Naive', 'build_prompt', 'extract_frame', 'time_to_seconds',
             'text_kwargs', 'generate_trigger_kwargs', 'prompt', 'prompt_graph_context'}
    nodes = [n for n in tree.body if
             isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names or
             isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(UPSTREAM / name), 'exec'), ns)
    wrapper = ast.parse('def infer(args, model, processor, video_path, question, options, realtime, time_stamp, task):\n pass').body[0]
    body = inference_nodes(profile)
    loop_index = 5 if profile == 'ovobench' else 6
    end_name = 'end_frame' if profile == 'ovobench' else 'max_frame'
    replacement = ast.parse(
        f"timestamp, frame_idx = encode_video_frames(args, model, processor, video_path, "
        f"start_frame, {end_name}, memories, extract_frame, encode_images, "
        f"whole_current_frame_tokens, retry_decode={profile == 'ovobench'})").body[0]
    if not isinstance(body[loop_index], ast.For):
        raise RuntimeError(f'Pinned {profile} source frame loop changed unexpectedly')
    body[loop_index] = replacement
    wrapper.body = body + ast.parse(
        "return dict(answer=response, scene_graph=current_Scene_Graph, context_graphs=context_Scene_Graphs, "
        "prompt=prompt_response, last_scheduled_timestamp=timestamp, frame_interval=args.frame_interval, "
        "encoded_frames=len(memories.memory_bank))").body
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), str(UPSTREAM / name), 'exec'), ns)
    return ns
