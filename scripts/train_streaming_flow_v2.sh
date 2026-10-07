#!/usr/bin/env bash
set -euo pipefail

# Portable, from-scratch launcher. See examples/streaming_flow_v2/README.md.
# DATA_ROOT and OUTPUT_DIR must be explicitly supplied; DRY_RUN=1 never trains.
variant="${1:?Usage: DATA_ROOT=... OUTPUT_DIR=... bash scripts/train_streaming_flow_v2.sh 8_1|16_1}"
case "$variant" in
  8_1) prediction_horizon=8 ;;
  16_1) prediction_horizon=16 ;;
  *) echo "Unknown variant: $variant (expected 8_1 or 16_1)" >&2; exit 2 ;;
esac

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
config_path="${CONFIG_PATH:-$repo_root/examples/streaming_flow_v2/experiments/screw_gripper25_20261005/$variant/train_config.json}"
data_root="${DATA_ROOT:?Set DATA_ROOT to the prepared LeRobot dataset directory}"
output_dir="${OUTPUT_DIR:?Set OUTPUT_DIR to a new run directory}"
split_file="${SPLIT_FILE:-$data_root/meta/gripper25_split_20261005.json}"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

# Validate the data/action contract and split before any model download or training.
train_episodes="$($python_bin - "$config_path" "$data_root" "$split_file" "$prediction_horizon" <<'PY'
import json
import sys
from pathlib import Path

config_path, root, split_file = map(Path, sys.argv[1:4])
cfg = json.loads(config_path.read_text())
policy = cfg["policy"]
assert policy["type"] == "streaming_flow_v2", "Expected a Streaming V2 recipe"
assert policy["pretrained_path"] is None and not cfg["resume"], "Launcher starts a fresh run"
assert policy["n_action_steps"] == int(sys.argv[4]), "Variant does not match CONFIG_PATH"
assert policy["chunk_size"] == int(sys.argv[4]) + 1, "Aligned path includes the previous action"
assert policy["use_previous_action_alignment"] and policy["execution_horizon"] == 1
info = json.loads((root / "meta/info.json").read_text())
assert info["codebase_version"] == "v3.0" and info["fps"] == 30
names = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
         "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
for key in ("action", "observation.state"):
    assert info["features"][key]["shape"] == [6] and info["features"][key]["names"] == names, key
for camera in ("front", "side"):
    key = f"observation.images.{camera}"
    assert info["features"][key]["shape"] == [480, 640, 3], key
split = json.loads(split_file.read_text())
train, validation = split["train"], split["validation"]
assert train and validation, "Both train and validation splits are required"
assert len(set(train)) == len(train) and len(set(validation)) == len(validation), "Duplicate episodes"
assert not set(train) & set(validation), "Training/validation overlap"
assert all(type(i) is int and 0 <= i < info["total_episodes"] for i in train + validation)
print(json.dumps(train, separators=(",", ":")))
PY
)"
dataset_repo_id="${DATASET_REPO_ID:-$($python_bin - "$config_path" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["dataset"]["repo_id"])
PY
)}"

args=(
  -m lerobot.scripts.lerobot_train
  "--config_path=$config_path"
  "--dataset.root=$data_root"
  "--dataset.repo_id=$dataset_repo_id"
  "--dataset.episodes=$train_episodes"
  "--output_dir=$output_dir"
  "--policy.device=${DEVICE:-cuda}"
)
if [[ "${SMOKE:-0}" == 1 ]]; then
  STEPS=1 BATCH_SIZE=2 NUM_WORKERS=0 SAVE_FREQ=1 LOG_FREQ=1
fi
for setting in STEPS BATCH_SIZE NUM_WORKERS SAVE_FREQ LOG_FREQ; do
  if [[ -n "${!setting:-}" ]]; then
    args+=("--${setting,,}=${!setting}")
  fi
done

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "$python_bin" "${args[@]}"
  printf '\n'
  exit 0
fi
if [[ -e "$output_dir" ]]; then
  echo "OUTPUT_DIR already exists: $output_dir. Use a new path; see the resume instructions." >&2
  exit 2
fi
exec "$python_bin" "${args[@]}"
