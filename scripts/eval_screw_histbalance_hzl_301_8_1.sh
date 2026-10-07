#!/usr/bin/env bash
set -euo pipefail

# Compare 15k and 20k checkpoints on all 301 training episodes.
# This is a training-set diagnostic, NOT held-out validation. It is useful for
# checking continued optimization but must not be reported as generalization.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
data_root="${DATA_ROOT:-/ICRA/Dataset/histogram_balance_cut_pull_screw_all_hover_gripper_25_assemble_episode}"
run_root="${RUN_ROOT:-/ICRA/Output/streaming_flow_v2_screw_histbalance_hzl_301_8_1}"
dataset_repo_id="${DATASET_REPO_ID:-HZL/histogram_balance_cut_pull_screw_all_hover_gripper_25_assemble_episode}"
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
echo "These reports use the same 301 episodes used in optimization; they are not held-out validation."
