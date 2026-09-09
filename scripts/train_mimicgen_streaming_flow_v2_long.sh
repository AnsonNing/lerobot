#!/usr/bin/env bash
set -euo pipefail

TASK_KEY="${1:?usage: $0 square|threading GPU_INDEX}"
GPU_INDEX="${2:?usage: $0 square|threading GPU_INDEX}"

RUNTIME_ROOT=/mnt/data/ningan/lerobot_mimicgen
VENV_ROOT="$RUNTIME_ROOT/venv"

case "$TASK_KEY" in
  square)
    TASK_NAME=Square_D1
    DATASET_REPO=ningan/mimicgen_square_d1
    DATASET_ROOT="$RUNTIME_ROOT/datasets/mimicgen_square_d1"
    SOURCE_HDF5=/home/ningan/internal-dispo/data/mimicgen/core/square_d1.hdf5
    OUTPUT_DIR="$RUNTIME_ROOT/outputs/mimicgen_square_d1_streaming_flow_v2_aligned_h1_full"
    ;;
  threading)
    TASK_NAME=Threading_D1
    DATASET_REPO=ningan/mimicgen_threading_d1
    DATASET_ROOT="$RUNTIME_ROOT/datasets/mimicgen_threading_d1"
    SOURCE_HDF5=/home/ningan/internal-dispo/data/mimicgen/core/threading_d1.hdf5
    OUTPUT_DIR="$RUNTIME_ROOT/outputs/mimicgen_threading_d1_streaming_flow_v2_aligned_h1_full"
    ;;
  *)
    echo "unknown task '$TASK_KEY' (expected square or threading)" >&2
    exit 2
    ;;
esac

if [[ ! -f "$DATASET_ROOT/meta/info.json" ]]; then
  echo "dataset is not finalized: $DATASET_ROOT" >&2
  exit 1
fi
DATASET_EPISODES=$("$VENV_ROOT/bin/python" -c \
  'import json, sys; print(json.load(open(sys.argv[1]))["total_episodes"])' \
  "$DATASET_ROOT/meta/info.json")
if [[ "$DATASET_EPISODES" != 1000 ]]; then
  echo "dataset conversion is incomplete: $DATASET_EPISODES/1000 episodes" >&2
  exit 1
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "refusing to overwrite existing output: $OUTPUT_DIR" >&2
  exit 1
fi

export HF_HOME="$RUNTIME_ROOT/hf_cache"
export HF_LEROBOT_HOME="$RUNTIME_ROOT/datasets"
export TORCH_HOME="$RUNTIME_ROOT/torch_cache"
export WANDB_MODE=disabled
export PYNPUT_BACKEND=dummy
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

exec env CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$VENV_ROOT/bin/lerobot-train" \
  --policy.type=streaming_flow_v2 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.n_obs_steps=2 \
  --policy.chunk_size=16 \
  --policy.n_action_steps=8 \
  --policy.execution_horizon=1 \
  --policy.use_previous_action_alignment=true \
  --policy.rollout_initial_action_mode=constant \
  --policy.rollout_initial_action='[0,0,0,0,0,0,-1]' \
  --policy.resize_shape='[84,84]' \
  --policy.crop_ratio=1.0 \
  --policy.freeze_vision_encoder=false \
  --policy.use_separate_rgb_encoder_per_camera=true \
  --policy.sfp_use_clip_image_conditioning=false \
  --policy.sfp_use_clip_text_conditioning=false \
  --policy.sfp_num_train_points=8 \
  --dataset.repo_id="$DATASET_REPO" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.image_transforms.enable=false \
  --env.type=mimicgen \
  --env.task="$TASK_NAME" \
  --env.dataset_path="$SOURCE_HDF5" \
  --env.episode_length=400 \
  --num_workers=8 \
  --prefetch_factor=2 \
  --persistent_workers=true \
  --batch_size=128 \
  --steps=300000 \
  --eval_freq=50000 \
  --eval.n_episodes=50 \
  --eval.batch_size=1 \
  --eval.use_async_envs=false \
  --seed=100000 \
  --save_checkpoint=true \
  --save_freq=10000 \
  --log_freq=100 \
  --wandb.enable=false \
  --output_dir="$OUTPUT_DIR"
