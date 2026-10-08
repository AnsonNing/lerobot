# Pull Screw histogram-balanced baselines (301 episodes)

This experiment reproduces the architecture and optimizer recipes from:

- `pretrained_model_push_button_w_diffusion_baseline`
- `pretrained_model_push_button_stream_baseline_wo_learn_time`

It changes the dataset to the 301-episode Pull Screw histogram-balanced dataset.
Both runs start from scratch, train for 20,000 optimizer steps, and save at
5k, 10k, 15k, and 20k. The Streaming Flow recipe keeps
`sfp_use_step_scaling=true`, matching the reference checkpoint.

## Environment

```bash
cd ~/NingAn/NingAn_Train/lerobot
uv sync --locked
```

## Smoke tests

Use unique output directories so they cannot block the full runs:

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE=1 OUTPUT_DIR=/tmp/screw_diffusion_smoke \
  bash scripts/train_screw_histbalance_baselines.sh diffusion

CUDA_VISIBLE_DEVICES=0 SMOKE=1 OUTPUT_DIR=/tmp/screw_streaming_flow_smoke \
  bash scripts/train_screw_histbalance_baselines.sh streaming_flow
```

## Full training

Run sequentially on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_screw_histbalance_baselines.sh diffusion
CUDA_VISIBLE_DEVICES=0 bash scripts/train_screw_histbalance_baselines.sh streaming_flow
```

With two GPUs, the two commands may run concurrently using device 0 and 1.
The default run directories are:

```text
outputs/screw_histbalance_baselines/diffusion_8_1
outputs/screw_histbalance_baselines/streaming_flow_8_1
```

## Compare 15k and 20k

First recompute each saved checkpoint's loss with the same seed and all 301
episodes:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_screw_histbalance_baselines.sh all
```

Then summarize both the training-log window and checkpoint evaluation:

```bash
uv run python scripts/compare_screw_histbalance_baseline_losses.py
```

The command compares the exact logged loss and the trailing 1,000-step mean,
then writes `outputs/screw_histbalance_baselines/loss_comparison.json`.
When the offline reports exist, the suggestion uses the deterministic
all-301 checkpoint loss; otherwise it falls back to the trailing training-loss
mean. Because all 301 episodes are used for optimization, this is a
training-set diagnostic, not held-out validation. Do not compare the absolute
loss value of Diffusion against Streaming Flow; use it only to compare 15k
versus 20k within the same policy. Real-robot success rate should make the
final decision when the two checkpoints are close.

## Package the selected checkpoint

```bash
bash scripts/package_screw_histbalance_baseline.sh diffusion 15000
bash scripts/package_screw_histbalance_baseline.sh streaming_flow 20000
```

The selected step may be either `15000` or `20000`. Archives and their SHA-256
files are written under `outputs/screw_histbalance_baselines/delivery/`.

Transfer an archive to the inference machine, for example:

```bash
scp outputs/screw_histbalance_baselines/delivery/<bundle>.tar.gz* \
  <inference-user>@<inference-host>:/home/lerobot/NingAn/checkpoint/
```

On the inference machine:

```bash
cd /home/lerobot/NingAn/checkpoint
sha256sum -c <bundle>.tar.gz.sha256
tar -xzf <bundle>.tar.gz
cd <bundle>
sha256sum -c SHA256SUMS.txt
```

The loadable checkpoint is `<bundle>/pretrained_model`.
