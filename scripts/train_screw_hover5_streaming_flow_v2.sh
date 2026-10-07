#!/usr/bin/env bash
set -euo pipefail

# Usage: CUDA_VISIBLE_DEVICES=0 scripts/train_screw_hover5_streaming_flow_v2.sh 8_1|16_1
# SMOKE=1 runs one optimizer step on the prepared dataset.
variant="${1:?Choose 8_1 or 16_1}"
case "$variant" in
  8_1) chunk_size=9; prediction_horizon=8 ;;
  16_1) chunk_size=17; prediction_horizon=16 ;;
  *) echo "Unknown variant: $variant" >&2; exit 2 ;;
esac

repo_root=/home/ningan/lerobot
data_root="${DATA_ROOT:-/mnt/data/ningan/screw_datasets/cut_pull_screw_hover5_gripper10_assemble_episode}"
dataset_repo_id="${DATASET_REPO_ID:-ningan/cut_pull_screw_hover5_gripper10_assemble_episode}"
split_file="${SPLIT_FILE:-screw_split_260930.json}"
venv=/mnt/data/ningan/lerobot_mimicgen/venv
export PYTHONPATH="$repo_root/src"
export VIRTUAL_ENV="$venv"
export PATH="$venv/bin:$PATH"
export TORCH_HOME=/mnt/data/ningan/lerobot_mimicgen/torch_cache
export HF_HOME=/mnt/data/ningan/lerobot_mimicgen/hf_cache
export WANDB_MODE=disabled

train_episodes="$($venv/bin/python - "$data_root/meta/$split_file" <<'PY'
import json, sys
print(json.dumps(json.load(open(sys.argv[1]))['train']))
PY
)"
steps="${STEPS:-20000}"
batch_size="${BATCH_SIZE:-16}"
num_workers="${NUM_WORKERS:-4}"
save_freq="${SAVE_FREQ:-2000}"
log_freq="${LOG_FREQ:-100}"
output_dir="${OUTPUT_DIR:-/mnt/data/ningan/screw_datasets/outputs/streaming_flow_v2_screw_hover5_${variant}_crop95}"
weighting_args=()
if [[ -n "${GRIPPER_CLOSE_BOOST:-}" ]]; then
  output_dir="${OUTPUT_DIR:-/mnt/data/ningan/screw_datasets/outputs/streaming_flow_v2_screw_hover5_${variant}_close_weighted_crop95}"
  weighting_args=(--sample_weighting.type=gripper_close "--sample_weighting.extra_params={\"threshold\":0.02,\"boost\":${GRIPPER_CLOSE_BOOST}}")
fi
if [[ "${SMOKE:-0}" == 1 ]]; then
  steps=1
  batch_size=2
  num_workers=2
  save_freq=1
  log_freq=1
  output_dir="${OUTPUT_DIR:-/mnt/data/ningan/screw_datasets/outputs/smoke_streaming_flow_v2_screw_hover5_${variant}_weighted${GRIPPER_CLOSE_BOOST:+_close}_crop95}"
fi

cd "$repo_root"
exec lerobot-train \
  --policy.type=streaming_flow_v2 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.n_obs_steps=2 \
  --policy.chunk_size="$chunk_size" \
  --policy.n_action_steps="$prediction_horizon" \
  --policy.execution_horizon=1 \
  --policy.use_previous_action_alignment=true \
  --policy.rollout_initial_action_mode=state \
  --policy.vision_backbone=resnet18 \
  --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
  --policy.resize_shape='[240,320]' \
  --policy.crop_ratio=0.95 \
  --policy.crop_is_random=true \
  --policy.freeze_vision_encoder=false \
  --policy.use_separate_rgb_encoder_per_camera=true \
  --policy.action_normalization_mode=per_dim \
  --policy.use_ema=true \
  --policy.sfp_use_adaptive_freq=true \
  --policy.sfp_num_train_points=4 \
  --policy.sfp_use_step_scaling=true \
  --dataset.repo_id="$dataset_repo_id" \
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
  --log_freq="$log_freq" \
  --wandb.enable=false \
  "${weighting_args[@]}" \
  --output_dir="$output_dir"
