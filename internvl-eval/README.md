# InternVL3.5 and LLaVA-OneVision evaluation

Query-time video evaluation for InternVL3.5-8B and LLaVA-OneVision-7B, using each
model's native visual preprocessing and the shared final question prompt.

Run all commands from this method directory. Replace `DATASET_PATH` and
`YOUR_MODEL` with your own locations.

## Environment

Linux, NVIDIA CUDA, Python 3.10 or 3.11, and `ffmpeg` / `ffprobe`.

```bash
conda create -n meminstream-internvl python=3.11 -y
conda activate meminstream-internvl
conda install -c conda-forge ffmpeg -y
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-internvl35.txt
```

For LLaVA use `requirements-llava-onevision.txt`. Both pin Transformers 4.52.4.
InternVL defaults to eager attention and loads the model's custom Python code;
LLaVA defaults to SDPA. FlashAttention is optional.

## Models

- [InternVL3.5-8B](https://huggingface.co/OpenGVLab/InternVL3_5-8B)
- [LLaVA-OneVision-Qwen2-7B-OV (HF)](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf)

```bash
python download_models.py --model all --model-root YOUR_MODEL
```

This creates `YOUR_MODEL/InternVL3_5-8B` and
`YOUR_MODEL/llava-onevision-qwen2-7b-ov-hf`, including all weight shards and custom
model code. Use `--model internvl35` or `--model llava_onevision` to download only
one model. The downloader records the resolved Hugging Face revision.

## Run

Set paths in `internvl35_predict.sh` or `llava_onevision_predict.sh`, then run the
chosen script with Bash. `predict.sh` defaults to InternVL. Direct commands:

```bash
CUDA_VISIBLE_DEVICES=0 python predict.py \
  --input DATASET_PATH/annotations.json --output results/internvl35.jsonl \
  --model internvl35 --model-path YOUR_MODEL/InternVL3_5-8B \
  --history-seconds 60 --fps 1 --num-gpus 1

CUDA_VISIBLE_DEVICES=0 python predict.py \
  --input DATASET_PATH/annotations.json --output results/llava_onevision.jsonl \
  --model llava_onevision --model-path YOUR_MODEL/llava-onevision-qwen2-7b-ov-hf \
  --history-seconds 60 --fps 1 --num-gpus 1
```

Relative video paths can use `--video-root DATASET_PATH`. Multiple GPUs use one
model replica per device. A positive history window pads short clips to the
requested frame count; `--history-seconds 0` uses all prior video.

## Evaluate

```bash
python score_predictions.py --input results/internvl35.jsonl \
  --report results/accuracy.json --details results/scores.jsonl
```

Use the corresponding output for LLaVA. Choices require exact set matching;
failed/unparseable objective answers count as incorrect. Open-ended questions are
excluded from objective accuracy.

Outputs preserve input fields and add `prediction`. The progress JSONL records
completed questions; score the completed main output. Existing outputs are protected,
so use a fresh path for each run.

## Quick check

Add `--limit 2 --dry-run` to the prediction command to inspect input parsing and
prompts without loading a model. Use a separate output path, then run `--limit 2`
on a CUDA GPU before a full evaluation. Dry-run outputs are not accuracy results.
Full checkpoint inference requires the corresponding Linux/CUDA environment.
