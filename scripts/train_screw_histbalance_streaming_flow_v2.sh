#!/usr/bin/env bash
set -euo pipefail
# Usage: CUDA_VISIBLE_DEVICES=0 scripts/train_screw_histbalance_streaming_flow_v2.sh 8_1
#        CUDA_VISIBLE_DEVICES=1 scripts/train_screw_histbalance_streaming_flow_v2.sh 16_1
variant="${1:?Choose 8_1 or 16_1}"
export DATA_ROOT=/mnt/data/ningan/screw_datasets/histogram_balance_cut_pull_screw_all_hover_gripper_25_assemble_episode
export DATASET_REPO_ID=ningan/histogram_balance_cut_pull_screw_all_hover_gripper_25_assemble_episode
export SPLIT_FILE=histbalance_split_20261004.json
export SAVE_FREQ="${SAVE_FREQ:-5000}"
# Use the supplied histogram-balanced sequence with the ordinary flow objective.
export GRIPPER_CLOSE_BOOST=""
if [[ "${SMOKE:-0}" == 1 ]]; then
  export OUTPUT_DIR="${OUTPUT_DIR:-/home/ningan/training_outputs/screw_histbalance_20261004/smoke_${variant}}"
else
  export OUTPUT_DIR="${OUTPUT_DIR:-/home/ningan/training_outputs/screw_histbalance_20261004/${variant}}"
fi
exec bash /home/ningan/lerobot/scripts/train_screw_hover5_streaming_flow_v2.sh "$variant"
