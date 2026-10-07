#!/usr/bin/env bash
set -euo pipefail
run_root=/mnt/data/ningan/screw_datasets/outputs/streaming_flow_v2_screw_gripper25_20261005
export PYTHONPATH=/home/ningan/lerobot/src
export TORCH_HOME=/mnt/data/ningan/lerobot_mimicgen/torch_cache
export HF_HOME=/mnt/data/ningan/lerobot_mimicgen/hf_cache
export CUDA_VISIBLE_DEVICES=0
while [[ ! -f "$run_root/JOB_DONE_8_1" || ! -f "$run_root/JOB_DONE_16_1" ]]; do
  if [[ -f "$run_root/FAILED_8_1" || -f "$run_root/FAILED_16_1" ]]; then
    printf 'Training or validation failed; packaging not started.\n' >&2
    exit 1
  fi
  sleep 10
done
/mnt/data/ningan/lerobot_mimicgen/venv/bin/python /home/ningan/lerobot/scripts/finalize_screw_gripper25_training.py
printf 'success\n' > "$run_root/DELIVERY_DONE"
