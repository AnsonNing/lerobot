#!/usr/bin/env bash
set -euo pipefail

# Compare 15k and 20k on all 301 episodes used by the Allen red-screw run.
# These are training-set diagnostics, not a held-out generalization evaluation.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
train_root="$(cd -- "$repo_root/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
data_root="${DATA_ROOT:-/home/allenchou0708/ICRA/NingAn_Train/Dataset/histogram_balance_cut_pull_red_screw_keyframes_all_hover_gripper_25_assemble_episode}"
run_root="${RUN_ROOT:-$train_root/outputs/streaming_flow_v2_screw_red_keyframes_allen_301_8_1}"
dataset_repo_id="${DATASET_REPO_ID:-allen/histogram_balance_cut_pull_red_screw_keyframes_all_hover_gripper_25_assemble_episode}"
device="${DEVICE:-cuda}"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

for step in 015000 020000; do
  checkpoint="$run_root/checkpoints/$step/pretrained_model"
  report="$run_root/all301_training_diagnostic_${step}.json"
  [[ -d "$checkpoint" ]] || { echo "Checkpoint not found: $checkpoint" >&2; exit 2; }
  "$python_bin" "$repo_root/scripts/eval_screw_hover5_streaming_flow_v2.py" \
    "$checkpoint" \
    --root "$data_root" \
    --repo-id "$dataset_repo_id" \
    --episodes all \
    --device "$device" \
    > "$report"
  echo "Wrote training-set diagnostic: $report"
done

echo "Compare flow_loss, first_action_mae_per_normalized_joint, and gripper-close metrics."
echo "These reports are not held-out validation because all 301 episodes were trained on."
