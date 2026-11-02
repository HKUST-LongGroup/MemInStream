# Response-G1 evaluation

[Response-G1](https://github.com/PolyX-Research/Response-G1) reactive evaluation,
source revision `5af0abe438155d33aba69ca59f25b8b3ea9b81d7`.
Qwen3 uses the upstream backbone; Qwen2.5 is a backbone adaptation. The unified
benchmark entry point always uses the final question prompt, with upstream scene
graph construction and memory processing preserved.

Run all commands from this method directory. Replace `DATASET_PATH` and
`YOUR_MODEL` with your own locations.

## Environment

Linux, NVIDIA CUDA, Python 3.10. The setup selects the CUDA 12.6 wheel build:

```bash
conda create -n meminstream-response-g1 python=3.10 -y
conda activate meminstream-response-g1
bash setup.sh
```

Torch 2.8.0 / torchvision 0.23.0, Transformers 4.57.1 and FlashAttention 2.8.3
are used. FlashAttention installation needs a compatible CUDA toolkit or wheel.
For SDPA, install the Torch pair and `requirements.txt` without FlashAttention,
then pass `--attn-implementation sdpa`. OpenCV handles decoding.

## Models

Download a complete local dense checkpoint:

- [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-VL-8B-Instruct', local_dir='YOUR_MODEL/Qwen3-VL-8B-Instruct')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct', local_dir='YOUR_MODEL/Qwen2.5-VL-7B-Instruct')"
```

The prediction loader reads local weights only.

## Run

Set paths in `predict.sh` (Qwen3) or `predict_qwen25.sh`, then run it with Bash.
Direct single-GPU command:

```bash
CUDA_VISIBLE_DEVICES=0 python predict.py \
  --input DATASET_PATH/annotations.json --output results/response_g1_qwen3.jsonl \
  --model qwen3 --model-path YOUR_MODEL/Qwen3-VL-8B-Instruct \
  --profile ovobench --fps 1 --vision-batch-size 8
```

For Qwen2.5 use `--model qwen25` and its checkpoint. For multiple independent GPU
replicas, replace the prefix `CUDA_VISIBLE_DEVICES=0 python predict.py` with
`python run_multigpu.py --gpus 0,1,2,3` and keep the remaining arguments.
The default OVO profile observes the full query-time prefix with its original
approximately 500-frame sampling budget. `--profile streamingbench` retains its
own sampling schedule and requires integer query times below 24 hours.
Relative video paths can use `--video-root DATASET_PATH`.

## Evaluate

```bash
python evaluate.py --input results/response_g1_qwen3.jsonl --output-dir results/scores
```

The scorer reads `response_g1_prediction`, uses exact option-set matching and
counts failed/unparseable objective answers as incorrect. Open-ended questions
are excluded. Score the complete merged output, since the report covers only
rows present in that file.

Append `--resume` to the identical prediction command to continue. Preserve the
manifest and, for multiple GPUs, the `.shards` directory. Changed input, source or
settings require a new output. Each worker holds one model replica. Source hashes
are checked before evaluation. `native.py` remains an optional upstream benchmark
runner with its original benchmark-specific schema; MemInStream uses `predict.py`.

## Quick check

Add `--limit 2 --dry-run` to the prediction command to inspect input parsing and
prompts without loading a model. Use a separate output path, then run `--limit 2`
on a CUDA GPU before a full evaluation. Dry-run outputs are not accuracy results.
Full checkpoint inference requires the corresponding Linux/CUDA environment.

Source manifests describe the bundled files. Where path examples were sanitized,
`original_upstream_sha256` preserves their previous hashes.
