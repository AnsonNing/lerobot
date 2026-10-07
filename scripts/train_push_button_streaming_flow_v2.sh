#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/train_push_button_streaming_flow_v2.sh 16_8|8_1
# Set SMOKE=1 for a one-step validation; a normal invocation starts training.
# DATA_ROOT, REPO_ID, and SPLIT_FILE select another dataset and train split.
variant="${1:?Choose 16_8 or 8_1}"
case "$variant" in
  16_8) chunk_size=17; prediction_horizon=16; execution_horizon=8 ;;
  8_1) chunk_size=9; prediction_horizon=8; execution_horizon=1 ;;
  *) echo "Unknown variant: $variant" >&2; exit 2 ;;
esac

repo_root=/home/ningan/lerobot
data_root="${DATA_ROOT:-/mnt/data/ningan/button_datasets/push_button_merged_260914}"
repo_id="${REPO_ID:-HZL/push_button_merged_260914}"
split_file="${SPLIT_FILE:-$data_root/meta/button_split_260914.json}"
venv=/mnt/data/ningan/lerobot_mimicgen/venv
export PYTHONPATH="$repo_root/src"
export VIRTUAL_ENV="$venv"
export PATH="$venv/bin:$PATH"
export TORCH_HOME=/mnt/data/ningan/lerobot_mimicgen/torch_cache
export HF_HOME=/mnt/data/ningan/lerobot_mimicgen/hf_cache
export WANDB_MODE=disabled

train_episodes="$($venv/bin/python - "$split_file" <<'PY'
import json, sys
print(json.dumps(json.load(open(sys.argv[1]))['train']))
PY
)"

steps="${STEPS:-50000}"
batch_size="${BATCH_SIZE:-16}"
num_workers="${NUM_WORKERS:-4}"
save_freq="${SAVE_FREQ:-10000}"
output_dir="${OUTPUT_DIR:-/mnt/data/ningan/button_datasets/outputs/streaming_flow_v2_button_${variant}_crop95}"
if [[ "${SMOKE:-0}" == 1 ]]; then
  steps=1
  batch_size=2
  num_workers=2
  save_freq=1
  output_dir="${OUTPUT_DIR:-/mnt/data/ningan/button_datasets/outputs/smoke_streaming_flow_v2_button_${variant}_crop95}"
fi

cd "$repo_root"
exec lerobot-train \
  --policy.type=streaming_flow_v2 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.n_obs_steps=2 \
  --policy.chunk_size="$chunk_size" \
  --policy.n_action_steps="$prediction_horizon" \
  --policy.execution_horizon="$execution_horizon" \
  --policy.use_previous_action_alignment=true \
  --policy.rollout_initial_action_mode=constant \
  --policy.rollout_initial_action="${ROLLOUT_INITIAL_ACTION:-[-11.098,-98.851,93.891,64.509,-3.59,0.866]}" \
  --policy.vision_backbone=resnet18 \
  --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
  --policy.resize_shape='[240,320]' \
  --policy.crop_ratio=0.95 \
  --policy.crop_is_random=true \
  --policy.freeze_vision_encoder=false \
  --policy.use_separate_rgb_encoder_per_camera=true \
  --policy.action_normalization_mode=per_dim \
  --policy.use_ema=true \
  --policy.sfp_use_adaptive_freq="${SFP_USE_ADAPTIVE_FREQ:-true}" \
  --policy.sfp_num_train_points="${SFP_NUM_TRAIN_POINTS:-4}" \
  --policy.sfp_use_step_scaling="${SFP_USE_STEP_SCALING:-true}" \
  --dataset.repo_id="$repo_id" \
  --dataset.root="$data_root" \
  --dataset.episodes="$train_episodes" \
  --dataset.image_transforms.enable=false \
  --num_workers="$num_workers" \
  --prefetch_factor=2 \
  --persistent_workers=true \
  --batch_size="$batch_size" \
  --steps="$steps" \
  --eval_freq=0 \
  --seed=100000 \
  --save_checkpoint=true \
  --save_freq="$save_freq" \
  --log_freq=100 \
  --wandb.enable=false \
  --output_dir="$output_dir"
