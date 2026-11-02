#!/usr/bin/env python3
"""Download complete inference snapshots to explicit local checkpoint folders."""
import argparse
import json
from pathlib import Path

MODELS = {
    'internvl35': ('OpenGVLab/InternVL3_5-8B', 'InternVL3_5-8B', 'internvl_chat'),
    'llava_onevision': ('llava-hf/llava-onevision-qwen2-7b-ov-hf',
                       'llava-onevision-qwen2-7b-ov-hf', 'llava_onevision'),
}
# Include custom Python code, tokenizer/processor files and every safetensors
# shard. Skip duplicate PyTorch weights, demo videos and training artifacts.
PATTERNS = ['*.json', '*.safetensors', '*.py', '*.model', '*.txt', '*.tiktoken',
            '*.jinja', '*.md', 'LICENSE*', '*.vocab', '*.merges']


def validate_checkpoint(path, expected_type):
    config = json.loads((path / 'config.json').read_text())
    if config.get('model_type') != expected_type:
        raise ValueError(f'Unexpected model_type in {path}')
    indices = list(path.glob('*.safetensors.index.json'))
    if indices:
        required = {name for index in indices
                    for name in json.loads(index.read_text())['weight_map'].values()}
        missing = [name for name in required if not (path / name).is_file()]
        if missing:
            raise FileNotFoundError(f'Missing weight shards: {missing}')
    elif not list(path.glob('*.safetensors')):
        raise FileNotFoundError(f'No safetensors weights in {path}')
    if expected_type == 'internvl_chat' and not (path / 'modeling_internvl_chat.py').is_file():
        raise FileNotFoundError('InternVL custom modeling code was not downloaded')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=[*MODELS, 'all'], default='all')
    parser.add_argument('--model-root', default='./models', help='parent directory for model folders')
    parser.add_argument('--local-dir', help='exact destination; only with a single --model')
    parser.add_argument('--revision', default='main', help='HF branch/tag/commit; resolved to an immutable commit')
    parser.add_argument('--max-workers', type=int, default=4)
    args = parser.parse_args(argv)
    if args.local_dir and args.model == 'all':
        parser.error('--local-dir requires a single --model')
    if args.max_workers < 1:
        parser.error('--max-workers must be positive')
    from huggingface_hub import HfApi, snapshot_download
    for key in MODELS if args.model == 'all' else [args.model]:
        repo, folder, expected = MODELS[key]
        path = (Path(args.local_dir) if args.local_dir else Path(args.model_root) / folder).expanduser().resolve()
        # Resolve once, so concurrent updates to main cannot mix revisions.
        revision = HfApi().model_info(repo, revision=args.revision).sha
        manifest_path = path / 'download_manifest.json'
        manifest = dict(repo_id=repo, requested_revision=args.revision, resolved_revision=revision)
        if path.exists() and any(path.iterdir()):
            if not manifest_path.exists():
                raise FileExistsError(f'{path} is not an empty or script-managed directory; choose a new destination')
            previous = json.loads(manifest_path.read_text())
            if any(previous.get(k) != manifest[k] for k in ('repo_id', 'resolved_revision')):
                raise FileExistsError(f'{path} contains another checkpoint/revision; use a new directory')
        path.mkdir(parents=True, exist_ok=True)
        # Save before downloading so interrupted downloads can be resumed.
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        print(f'Downloading {repo}@{revision} -> {path}', flush=True)
        snapshot_download(repo_id=repo, revision=revision, local_dir=str(path),
                          allow_patterns=PATTERNS, max_workers=args.max_workers)
        validate_checkpoint(path, expected)
        print(f'Ready: {path}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
