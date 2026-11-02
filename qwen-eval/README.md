# Qwen-VL evaluation

Qwen2.5-VL and dense Qwen3-VL baselines with query-time video windows.

Run all commands from this method directory. Replace `DATASET_PATH` and
`YOUR_MODEL` with your own locations.

## Environment

Linux, NVIDIA CUDA, Python 3.10 or 3.11, and system `ffmpeg` / `ffprobe`.

```bash
conda create -n meminstream-qwen python=3.11 -y
conda activate meminstream-qwen
conda install -c conda-forge ffmpeg -y
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-predict-filtered-qwen.txt
```

Transformers is pinned to 4.57.1. The default SDPA backend does not require FlashAttention.

## Models

- [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)

Download the complete checkpoint for the backbone you use:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct', local_dir='YOUR_MODEL/Qwen2.5-VL-7B-Instruct')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-VL-8B-Instruct', local_dir='YOUR_MODEL/Qwen3-VL-8B-Instruct')"
```

## Run

Edit `INPUT` and `MODEL_PATH` in `predict.sh`, then run `bash predict.sh`.
The model-specific alternatives are `qwen25_predict.sh` and `qwen3_predict.sh`.
Equivalent direct command:

```bash
CUDA_VISIBLE_DEVICES=0 python predict.py \
  --input DATASET_PATH/annotations.json \
  --model qwen25 --model-path YOUR_MODEL/Qwen2.5-VL-7B-Instruct \
  --output results/predictions.jsonl \
  --history-seconds 600 --fps 1 --num-gpus 1
```

For Qwen3, set `--model qwen3` and its checkpoint path. Set `--num-gpus N` for
one model replica per visible GPU. `--history-seconds 0` uses all prior video;
choose the same window explicitly when comparing methods. Relative video paths
can be resolved with `--video-root DATASET_PATH`.

## Evaluate

```bash
python score_predictions.py --input results/predictions.jsonl \
  --report results/accuracy.json --details results/scores.jsonl
```

Choice, binary and supported counting questions are scored automatically. Open-ended
questions are excluded from objective accuracy. Errors and unparseable answers
count as incorrect. Score the completed main output, not a partial progress file.

Predictions preserve the input rows and add `prediction`; a progress JSONL is also
written. Existing outputs are protected; use a new output path for a new run.
Optional evidence plots: install `requirements-analysis.txt`, then run
`bash run_evidence_analysis.sh`.

## Quick check

Add `--limit 2 --dry-run` to the prediction command to inspect input parsing and
prompts without loading a model. Use a separate output path, then run `--limit 2`
on a CUDA GPU before a full evaluation. Dry-run outputs are not accuracy results.
Full checkpoint inference requires the corresponding Linux/CUDA environment.
