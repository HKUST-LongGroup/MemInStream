# TimeChat-Online evaluation

[TimeChat-Online](https://github.com/yaolinli/TimeChat-Online) evaluation using the
trained TimeChat checkpoint and official DTD token dropping. Source revision:
`25a15783a6fd83e3360b3326cee44029fcce0132`.

Run all commands from this method directory. Replace `DATASET_PATH` and
`YOUR_MODEL` with your own locations.

## Environment

Linux x86_64, a BF16/FlashAttention-capable NVIDIA GPU, Python 3.10, and FFmpeg.

```bash
conda create -n meminstream-timechat python=3.10 -y
conda activate meminstream-timechat
conda install -c conda-forge ffmpeg -y
bash setup.sh
```

The setup uses Torch 2.5.1 / torchvision 0.20.1 (CUDA 12.1), Transformers 4.50.0,
qwen-vl-utils 0.0.10 and the matching FlashAttention 2.7.4.post1 wheel. It checks
GPU visibility and the official model import after installation.

## Models

Use [wyccccc/TimeChatOnline-7B](https://huggingface.co/wyccccc/TimeChatOnline-7B):

```bash
python download_model.py --output YOUR_MODEL/TimeChatOnline-7B
```

The downloader records the resolved model revision and retrieves weights,
processor, tokenizer and generation configuration. Re-run it to resume an
interrupted download. Prediction reads local files only. A plain Qwen2.5-VL
checkpoint is not a substitute for the trained TimeChat model.

## Run

Set paths in `predict.sh`, then run `bash predict.sh`, or:

```bash
CUDA_VISIBLE_DEVICES=0 python predict.py \
  --input DATASET_PATH/annotations.json --output results/timechat.jsonl \
  --model-path YOUR_MODEL/TimeChatOnline-7B \
  --history-seconds 0 --num-gpus 1
```

The default uses full history and adaptive sampling: 1 FPS up to 300 seconds,
0.5 FPS up to 600 seconds, and 0.2 FPS beyond that. `--fps` explicitly overrides
this policy. The native frame cap is 1016, with DTD feature threshold 0.5.
Multiple GPUs use independent model replicas. Relative video paths can use
`--video-root DATASET_PATH`.

## Evaluate

Prediction automatically writes an accuracy report. To score again:

```bash
python score_predictions.py --input results/timechat.jsonl \
  --report results/accuracy.json --details results/scores.jsonl
```

Choices require exact option-set matching; failed/unparseable objective answers
count as incorrect. Open-ended questions are excluded from objective accuracy.

Append `--resume` to the identical command to continue. Keep the progress JSONL,
manifest and token-drop logs. Existing error rows are preserved during resume.
Use a new output path after changing inputs, source or settings. Upstream source
hashes are verified before running; licenses remain in `upstream/`.

## Quick check

Add `--limit 2 --dry-run` to the prediction command to inspect input parsing and
prompts without loading a model. Use a separate output path, then run `--limit 2`
on a CUDA GPU before a full evaluation. Dry-run outputs are not accuracy results.
Full checkpoint inference requires the corresponding Linux/CUDA environment.

Source manifests describe the bundled files. Where path examples were sanitized,
`original_upstream_sha256` preserves their previous hashes.
