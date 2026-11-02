# StreamForest evaluation

[StreamForest](https://github.com/MCG-NJU/StreamForest) query-time evaluation with
its native SigLIP vision tower, FSTW and PEMF memory. Source revision:
`f9d14670ff451ff5865638e17c9958e99bf2fec5`.

Run all commands from this method directory. Replace `DATASET_PATH` and
`YOUR_MODEL` with your own locations.

## Environment

Linux, NVIDIA CUDA, Python 3.10, and system `ffmpeg` / `ffprobe`.

```bash
conda create -n meminstream-streamforest python=3.10 -y
conda activate meminstream-streamforest
conda install -c conda-forge ffmpeg -y
python -m pip install --upgrade pip wheel packaging ninja
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install flash-attn==2.6.3 --no-build-isolation
python -m pip install -r requirements.install.txt
```

The bundled environment pins Transformers 4.46.0 and Accelerate 0.29.3.
FlashAttention needs a compatible CUDA toolkit or wheel. Upstream eagerly imports
FlashAttention and DeepSpeed even when inference uses SDPA. `requirements.install.txt`
retains the original environment pins while omitting two unused optional packages;
`StreamForest/requirements.txt` preserves the upstream dependency list.

## Models

- [StreamForest-Qwen2-7B](https://huggingface.co/MCG-NJU/StreamForest-Qwen2-7B): general model.
- [StreamForest-Drive-Qwen2-7B](https://huggingface.co/MCG-NJU/StreamForest-Drive-Qwen2-7B): optional driving model.
- [SigLIP SO400M patch14 384](https://huggingface.co/google/siglip-so400m-patch14-384): required vision tower.

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('MCG-NJU/StreamForest-Qwen2-7B', local_dir='YOUR_MODEL/StreamForest-Qwen2-7B')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('google/siglip-so400m-patch14-384', local_dir='YOUR_MODEL/siglip-so400m-patch14-384')"
```

Keep `siglip` in the local vision-tower directory name because the upstream builder
uses the name to select its implementation. Use StreamForest checkpoints, not
plain Qwen-VL weights. The adapter overrides the checkpoint's original vision-tower
location with the path supplied below.

## Run

Set `INPUT`, `MODEL_PATH` and `VISION_TOWER` in `predict.sh`, then run
`bash predict.sh`, or:

```bash
CUDA_VISIBLE_DEVICES=0 python predict.py \
  --input DATASET_PATH/annotations.json --output results/streamforest.jsonl \
  --model-path YOUR_MODEL/StreamForest-Qwen2-7B \
  --vision-tower YOUR_MODEL/siglip-so400m-patch14-384 \
  --history-seconds 600 --fps 1 --num-gpus 1
```

The default aligned sampling uses a 600-second window, followed by native SigLIP
preprocessing. Set `--history-seconds 0` for full history. Each question resets
memory; multiple GPUs use one model replica per device. Relative video paths can
use `--video-root DATASET_PATH`.

## Evaluate

Prediction automatically writes an accuracy report. To score again:

```bash
python score_predictions.py --input results/streamforest.jsonl \
  --report results/accuracy.json --details results/scores.jsonl
```

Choices require exact option-set matching. Failed/unparseable objective questions
count as incorrect, while open-ended questions are excluded.

The main output preserves input fields and adds `prediction`; progress and manifest
files are written alongside it. This runner does not automatically resume: use a
new output path for a new run. Do not treat a partial progress file as a completed
benchmark. Source hashes are checked at startup. Upstream licenses are retained.

## Quick check

Add `--limit 2 --dry-run` to the prediction command to inspect input parsing and
prompts without loading a model. Use a separate output path, then run `--limit 2`
on a CUDA GPU before a full evaluation. Dry-run outputs are not accuracy results.
Full checkpoint inference requires the corresponding Linux/CUDA environment.

Source manifests describe the bundled files. Where path examples were sanitized,
`original_upstream_sha256` preserves their previous hashes.
