#!/usr/bin/env python3
"""Resolve an immutable HF revision, download full TimeChat checkpoint, record provenance."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--revision', default='main', help='HF commit/ref, independent of GitHub source commit')
    args = p.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    repo = 'wyccccc/TimeChatOnline-7B'
    revision = HfApi().model_info(repo, revision=args.revision).sha
    root = Path(args.output).expanduser().resolve()
    provenance = root / 'timechat_download.json'
    if provenance.exists():
        old = json.loads(provenance.read_text())
        if old['revision'] != revision:
            raise RuntimeError('Destination contains a different revision; use a new directory')
    snapshot_download(repo_id=repo, revision=revision, local_dir=str(root))
    for name in ['config.json', 'preprocessor_config.json', 'tokenizer_config.json', 'generation_config.json']:
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    index = root / 'model.safetensors.index.json'
    if index.exists():
        for name in set(json.loads(index.read_text())['weight_map'].values()):
            if not (root / name).is_file():
                raise FileNotFoundError(root / name)
    elif not (root / 'model.safetensors').is_file():
        raise FileNotFoundError('No safetensors checkpoint found')
    provenance.write_text(json.dumps({'repo_id': repo, 'revision': revision}, indent=2) + '\n')
    print(f'Downloaded {repo}@{revision} to {root}')


if __name__ == '__main__':
    main()
