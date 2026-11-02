#!/usr/bin/env python3
"""Download the official trained model and base-Qwen processor assets (not base weights)."""
import argparse
import json
from pathlib import Path


def main():
    from huggingface_hub import HfApi, snapshot_download
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'models')
    p.add_argument('--model-revision', default='main')
    p.add_argument('--processor-revision', default='main')
    args = p.parse_args()
    api = HfApi()
    downloads = {}
    specs = [
        ('zhang9302002/Flash-VStream-Qwen-7b', args.model_revision,
         ['*.json', '*.safetensors', '*.bin', '*.txt', '*.model', '*.tiktoken']),
        ('Qwen/Qwen2-VL-7B-Instruct', args.processor_revision,
         ['config.json', 'preprocessor_config.json', 'processor_config.json', 'tokenizer*.json',
          'vocab.json', 'merges.txt', 'special_tokens_map.json', 'added_tokens.json', 'chat_template*']),
    ]
    for repo, revision, patterns in specs:
        resolved = api.model_info(repo, revision=revision).sha
        folder = args.output_dir / repo.split('/')[-1]
        print(f'Downloading {repo}@{resolved} -> {folder}', flush=True)
        snapshot_download(repo_id=repo, revision=resolved, local_dir=str(folder), allow_patterns=patterns)
        downloads[repo] = dict(revision=resolved, path=str(folder.resolve()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'download_manifest.json').write_text(json.dumps(downloads, indent=2) + '\n')
    print('Saved immutable revisions to download_manifest.json. Base Qwen weights are not needed.')


if __name__ == '__main__':
    main()
