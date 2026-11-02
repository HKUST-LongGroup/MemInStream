# HERMES evaluation

[HERMES](https://github.com/haowei-freesky/HERMES), with Qwen2.5-VL or Hugging Face
LLaVA-OneVision checkpoints. Source revision: `8d699b16a6bedb9086c1b39ec4253c6a1d1ce789`.
The adapter uses the official KV compression and full video history for each question.

Run all commands from this method directory. Replace `DATASET_PATH` and
`YOUR_MODEL` with your own locations.

## Environment

Linux, NVIDIA CUDA, Python 3.12. Use separate environments for the two backbones:

```bash
conda create -n meminstream-hermes-qwen python=3.12 -y
conda activate meminstream-hermes-qwen
python -m pip install -r requirements-qwen.txt
python -m pip install flash-attn --no-build-isolation
```

For LLaVA, create another Python 3.12 environment and install `requirements-llava.txt`
instead, then FlashAttention. Both require Torch 2.9.1 / torchvision 0.24.1.
Qwen uses Transformers 4.57.1; LLaVA uses the pinned Transformers Git revision in
its requirements file. Decord performs video decoding. FlashAttention needs a
matching CUDA toolkit or prebuilt wheel.

## Models

- [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [LLaVA-OneVision-Qwen2-7B-OV (HF)](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf)

Download only the backbone you intend to run:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct', local_dir='YOUR_MODEL/Qwen2.5-VL-7B-Instruct')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('llava-hf/llava-onevision-qwen2-7b-ov-hf', local_dir='YOUR_MODEL/llava-onevision-qwen2-7b-ov-hf')"
```

## Run

Set paths in `predict.sh`, then run `bash predict.sh`, or:

```bash
CUDA_VISIBLE_DEVICES=0 python predict.py \
  --input DATASET_PATH/annotations.json --output results/hermes.jsonl \
  --model qwen25 --model-path YOUR_MODEL/Qwen2.5-VL-7B-Instruct \
  --fps 0.5 --kv-size 6000 --chunk-frames 16 --num-gpus 1
```

For LLaVA, activate its environment and replace the model arguments with
`--model llavaov --model-path YOUR_MODEL/llava-onevision-qwen2-7b-ov-hf`.
`--history-seconds` only accepts 0: this method uses the full query-time prefix.
Each question starts with independent memory. The official loader uses FP16;
choice answers default to 16 new tokens, other answers to 256. Set
`--max-new-tokens` explicitly if a different limit is required.
Relative video paths can use `--video-root DATASET_PATH`.

## Evaluate

```bash
python evaluate.py --input DATASET_PATH/annotations.json \
  --predictions results/hermes.jsonl --output-dir results/scores
```

The default Qwen scoring policy includes missing/failed objective questions in the
denominator and excludes open-ended questions. Choices use exact set matching.
The optional `--policy fluxmem` has a different denominator; report coverage with it.

Append `--resume` to the same command to resume. Input, source and settings must
match the saved manifest. Each GPU holds a full model replica. Upstream hashes are
checked at startup; licenses remain in `upstream/HERMES/`.

## Quick check

Add `--limit 2 --dry-run` to the prediction command to inspect input parsing and
prompts without loading a model. Use a separate output path, then run `--limit 2`
on a CUDA GPU before a full evaluation. Dry-run outputs are not accuracy results.
Full checkpoint inference requires the corresponding Linux/CUDA environment.

Source manifests describe the bundled files. Where path examples were sanitized,
`original_upstream_sha256` preserves their previous hashes.
