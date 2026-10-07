#!/usr/bin/env bash
set -euo pipefail

# Train Streaming Flow V2 8->1 from scratch on all episodes of the red-screw
# keyframe-rendered dataset on Allen's training host.
#
# This intentionally trains on all 301 episodes. Checkpoint diagnostics at 15k
# and 20k therefore measure the training set; they are not held-out validation.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
train_root="$(cd -- "$repo_root/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
data_root="${DATA_ROOT:-/home/allenchou0708/ICRA/NingAn_Train/Dataset/histogram_balance_cut_pull_red_screw_keyframes_all_hover_gripper_25_assemble_episode}"
output_dir="${OUTPUT_DIR:-$train_root/outputs/streaming_flow_v2_screw_red_keyframes_allen_301_8_1}"
dataset_repo_id="${DATASET_REPO_ID:-allen/histogram_balance_cut_pull_red_screw_keyframes_all_hover_gripper_25_assemble_episode}"
config_path="${CONFIG_PATH:-$repo_root/examples/streaming_flow_v2/experiments/screw_histbalance_20261004/8_1/train_config.json}"

steps="${STEPS:-20000}"
batch_size="${BATCH_SIZE:-16}"
num_workers="${NUM_WORKERS:-4}"
save_freq="${SAVE_FREQ:-5000}"
log_freq="${LOG_FREQ:-100}"
device="${DEVICE:-cuda}"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=disabled

all_episodes="$($python_bin - "$config_path" "$data_root" <<'PY'
import json
import sys
from pathlib import Path

config_path, root = map(Path, sys.argv[1:3])
cfg = json.loads(config_path.read_text())
policy = cfg["policy"]
assert policy["type"] == "streaming_flow_v2"
assert policy["pretrained_path"] is None and not cfg["resume"], "Expected a fresh training recipe"
assert policy["n_action_steps"] == 8 and policy["chunk_size"] == 9
assert policy["execution_horizon"] == 1 and policy["use_previous_action_alignment"]
assert policy["sfp_use_step_scaling"] is True, "The recipe must preserve SFP step scaling"

info = json.loads((root / "meta" / "info.json").read_text())
assert info["codebase_version"] == "v3.0" and info["fps"] == 30
assert info["total_episodes"] == 301, f"Expected exactly 301 episodes, got {info['total_episodes']}"
joint_names = [
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
]
for key in ("action", "observation.state"):
    assert info["features"][key]["shape"] == [6]
    assert info["features"][key]["names"] == joint_names
for camera in ("front", "side"):
    assert info["features"][f"observation.images.{camera}"]["shape"] == [480, 640, 3]

print(json.dumps(list(range(301)), separators=(",", ":")))
PY
)"

if [[ "${SMOKE:-0}" == 1 ]]; then
  steps=1
  batch_size=2
  num_workers=0
  save_freq=1
  log_freq=1
fi

args=(
  -m lerobot.scripts.lerobot_train
  "--config_path=$config_path"
  "--dataset.root=$data_root"
  "--dataset.repo_id=$dataset_repo_id"
  "--dataset.episodes=$all_episodes"
  "--output_dir=$output_dir"
  "--policy.device=$device"
  "--steps=$steps"
  "--batch_size=$batch_size"
  "--num_workers=$num_workers"
  "--save_freq=$save_freq"
  "--log_freq=$log_freq"
  "--wandb.enable=false"
)

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "$python_bin" "${args[@]}"
  printf '\n'
  exit 0
fi
if [[ -e "$output_dir" ]]; then
  echo "OUTPUT_DIR already exists: $output_dir" >&2
  echo "Choose a new directory. Do not overwrite an existing training run." >&2
  exit 2
fi

exec "$python_bin" "${args[@]}"
