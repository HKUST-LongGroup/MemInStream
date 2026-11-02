# Flash-VStream evaluation

[Flash-VStream-Qwen](https://github.com/IVGSZ/Flash-VStream) offline evaluation,
source revision `8f6bde2f397f4846505df4d03364d97617fc02ee`. This method uses the
trained Flash-VStream model and its native Flash Memory implementation.

Run all commands from this method directory. Replace `DATASET_PATH` and
`YOUR_MODEL` with your own locations.

## Environment

Linux x86_64, a BF16/FlashAttention-capable NVIDIA GPU, Python 3.10, and FFmpeg.

```bash
conda create -n meminstream-flash-vstream python=3.10.14 -y
conda activate meminstream-flash-vstream
conda install -c conda-forge ffmpeg -y
bash setup_env.sh
```

The setup pins Torch 2.6.0 / torchvision 0.21.0 (CUDA 12.4), Transformers 4.45.0
and a matching FlashAttention 2.7.4.post1 wheel. Do not substitute the pip
`qwen-vl-utils` package for the bundled custom implementation.
`python check_environment.py` checks the model imports and visible GPU.

## Models

Two sets of assets are required:

- [Flash-VStream-Qwen-7b](https://huggingface.co/zhang9302002/Flash-VStream-Qwen-7b): trained weights.
- [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct): processor and tokenizer only.

```bash
python download_models.py --output-dir YOUR_MODEL
```

The downloader creates `YOUR_MODEL/Flash-VStream-Qwen-7b` and
`YOUR_MODEL/Qwen2-VL-7B-Instruct`, and records resolved revisions. It does not
download a second copy of base-model weights. Flash-VStream is based on Qwen2-VL;
Qwen2.5-VL weights cannot replace this trained checkpoint.

## Run

Set `INPUT`, `MODEL_PATH` and `PROCESSOR_PATH` in `predict.sh`, then run
`bash predict.sh`, or:

```bash
CUDA_VISIBLE_DEVICES=0 python predict.py \
  --input DATASET_PATH/annotations.json --output results/flash_vstream.jsonl \
  --model-path YOUR_MODEL/Flash-VStream-Qwen-7b \
  --processor-path YOUR_MODEL/Qwen2-VL-7B-Instruct \
  --history-seconds 256 --num-gpus 1
```

The script uses a 256-second window; the Python default of 0 uses full history.
The native final frame budget is 240. `--extract-fps` controls candidate sampling,
not the effective final FPS for long videos. Each GPU holds one independent model
replica. Relative video paths can use `--video-root DATASET_PATH`.

## Evaluate

```bash
python evaluate.py --input DATASET_PATH/annotations.json \
  --predictions results/flash_vstream.jsonl --output-dir results/scores
```

This validates input coverage and uses exact option-set matching. Failed/missing
objective questions count as incorrect; open-ended questions are excluded.
The optional `evaluate_fluxmem_style.py` uses a different denominator and reports
coverage separately.

Append `--resume` to the identical prediction command to continue. Preserve output
sidecars and use new output paths after changing configuration or source. The
manifest verifies the bundled source snapshot. Licenses remain with upstream code.

## Quick check

Add `--limit 2 --dry-run` to the prediction command to inspect input parsing and
prompts without loading a model. Use a separate output path, then run `--limit 2`
on a CUDA GPU before a full evaluation. Dry-run outputs are not accuracy results.
Full checkpoint inference requires the corresponding Linux/CUDA environment.

Source manifests describe the bundled files. Where path examples were sanitized,
`original_upstream_sha256` preserves their previous hashes.
