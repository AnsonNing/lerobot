#!/usr/bin/env bash
set -euo pipefail
# Predict 8 or 16 new actions, execute one, with the supplied episode sequences.
variant="${1:?Choose 8_1 or 16_1}"
export DATA_ROOT=/mnt/data/ningan/screw_datasets/cut_pull_screw_all_hover_gripper25_merged_20261005
export DATASET_REPO_ID=ningan/cut_pull_screw_all_hover_gripper25_merged_20261005
export SPLIT_FILE=gripper25_split_20261005.json
export SAVE_FREQ="${SAVE_FREQ:-5000}"
export GRIPPER_CLOSE_BOOST=""
run_root=/mnt/data/ningan/screw_datasets/outputs/streaming_flow_v2_screw_gripper25_20261005
if [[ "${SMOKE:-0}" == 1 ]]; then
  export OUTPUT_DIR="${OUTPUT_DIR:-$run_root/smoke_${variant}}"
else
  export OUTPUT_DIR="${OUTPUT_DIR:-$run_root/${variant}}"
fi
exec bash /home/ningan/lerobot/scripts/train_screw_hover5_streaming_flow_v2.sh "$variant"
