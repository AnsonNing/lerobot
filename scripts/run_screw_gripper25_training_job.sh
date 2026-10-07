#!/usr/bin/env bash
set -euo pipefail
variant="${1:?Choose 8_1 or 16_1}"
run_root=/mnt/data/ningan/screw_datasets/outputs/streaming_flow_v2_screw_gripper25_20261005
data_root=/mnt/data/ningan/screw_datasets/cut_pull_screw_all_hover_gripper25_merged_20261005
python_bin=/mnt/data/ningan/lerobot_mimicgen/venv/bin/python
export PYTHONPATH=/home/ningan/lerobot/src
export TORCH_HOME=/mnt/data/ningan/lerobot_mimicgen/torch_cache
export HF_HOME=/mnt/data/ningan/lerobot_mimicgen/hf_cache
trap 'status=$?; if (( status != 0 )); then printf "%s\n" "$status" > "$run_root/FAILED_${variant}"; fi' EXIT
bash /home/ningan/lerobot/scripts/train_screw_gripper25_streaming_flow_v2.sh "$variant" > "$run_root/train_${variant}.log" 2>&1
printf '20000\n' > "$run_root/TRAIN_DONE_${variant}"
for step in 005000 010000 015000 020000; do
 "$python_bin" /home/ningan/lerobot/scripts/eval_screw_hover5_streaming_flow_v2.py \
  "$run_root/$variant/checkpoints/$step/pretrained_model" \
  --root "$data_root" --repo-id ningan/cut_pull_screw_all_hover_gripper25_merged_20261005 \
  --split-file gripper25_split_20261005.json --device cuda \
  > "$run_root/val_${variant}_${step}.json" 2> "$run_root/val_${variant}_${step}.log"
done
printf 'success\n' > "$run_root/JOB_DONE_${variant}"
