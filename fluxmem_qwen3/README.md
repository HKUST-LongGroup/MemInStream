# FluxMem with Qwen3-VL

[FluxMem](https://github.com/ShareLab-SII/FluxMem) evaluation using Qwen3-VL-8B-Instruct.
The bundled source snapshot includes local algorithm changes relative to upstream
revision `6fdbf65b6f2af1a94ca41d542be5be86a565d338`; use the bundled files for reproduction.
The Qwen3 adapter preserves FluxMem token selection and adapts positions and DeepStack features to Qwen3.

Run all commands from this method directory. Replace `DATASET_PATH` and
`YOUR_MODEL` with your own locations.

## Environment

Linux, NVIDIA CUDA, Python 3.11, and FFmpeg video libraries.

```bash
conda create -n meminstream-fluxmem-qwen3 python=3.11 -y
conda activate meminstream-fluxmem-qwen3
conda install -c conda-forge ffmpeg -y
bash setup.sh
```

The setup uses Torch 2.6.0, torchvision 0.21.0 and Transformers 4.57.1. SDPA is the default; FlashAttention is optional. The bundled Qwen3 adapter needs no editable Qwen2.5 package installation.

## Models

Download [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), including tokenizer and processor files:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-VL-8B-Instruct', local_dir='YOUR_MODEL/Qwen3-VL-8B-Instruct')"
```

Use the matching dense VL checkpoint; weights are not included.

## Run

Set paths in `predict.sh` and run `bash predict.sh`, or run:

```bash
CUDA_VISIBLE_DEVICES=0 python predict.py \
  --input DATASET_PATH/annotations.json \
  --model-path YOUR_MODEL/Qwen3-VL-8B-Instruct \
  --output results/fluxmem.jsonl \
  --time-window 256 --fps 1 --num-gpus 1 \
  --video-chunk-t 4 --cpu-threads 4 --attn-implementation sdpa
```

`--time-window 0` selects full history. The default memory keeps 8 short-term and
64 medium-term frames; the video budget is 256 frames. Multiple GPUs use independent
model replicas. Use video paths valid from this directory; path remapping accepts
`--path-prefix OLD_DATASET=DATASET_PATH`.

## Evaluate

```bash
python evaluate.py --input DATASET_PATH/annotations.json \
  --predictions results/fluxmem.jsonl --output-dir results/scores
```

The scorer checks that predictions match the input. Choices require exact set
matching; open-ended questions are skipped unless reviewed scores are provided.
`accuracy_scored` excludes unscored/missing samples, so also inspect coverage and
the lower-bound accuracy. It is not interchangeable with another scorer's denominator.

Predictions use `fluxmem_prediction`. Re-running the identical command resumes
completed work automatically; there is no separate resume flag. Preserve manifests
and shards, and choose a new output for any changed input, source or configuration.
Dry-run writes `OUTPUT.audit.json` and `OUTPUT.preview.jsonl`.
Optional evidence plots: install `requirements-analysis.txt`, then run `bash analysis.sh`.
Upstream source hashes are checked at startup.

## Quick check

Add `--limit 2 --dry-run` to the prediction command to inspect input parsing and
prompts without loading a model. Use a separate output path, then run `--limit 2`
on a CUDA GPU before a full evaluation. Dry-run outputs are not accuracy results.
Full checkpoint inference requires the corresponding Linux/CUDA environment.

Source manifests describe the bundled files. Where path examples were sanitized,
`original_upstream_sha256` preserves their previous hashes.
