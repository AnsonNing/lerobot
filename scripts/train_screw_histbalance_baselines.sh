#!/usr/bin/env bash
set -euo pipefail

# Train one Pull Screw baseline from scratch on all 301 histogram-balanced episodes.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/train_screw_histbalance_baselines.sh diffusion
#   CUDA_VISIBLE_DEVICES=0 bash scripts/train_screw_histbalance_baselines.sh streaming_flow

variant="${1:-}"
case "${variant}" in
  diffusion|streaming_flow) ;;
  *) echo "Usage: $0 diffusion|streaming_flow [extra lerobot-train arguments]" >&2; exit 2 ;;
esac
shift

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
default_data_root="/home/lerobot/NingAn/lerobot/NingAn_Inference/Pull_Screw/cut_datasets/"
default_data_root+="histogram_balance_cut_pull_screw_all_hover_gripper_25_assemble_episode"
data_root="${DATA_ROOT:-$default_data_root}"
default_repo_id="HZL/histogram_balance_cut_pull_screw_all_hover_gripper_25_assemble_episode"
dataset_repo_id="${DATASET_REPO_ID:-$default_repo_id}"
output_base="${OUTPUT_BASE:-$repo_root/outputs/screw_histbalance_baselines}"

steps="${STEPS:-20000}"
save_freq="${SAVE_FREQ:-5000}"
log_freq="${LOG_FREQ:-100}"
batch_size="${BATCH_SIZE:-16}"
num_workers="${NUM_WORKERS:-4}"
device="${DEVICE:-cuda}"

case "${variant}" in
  diffusion)
    default_config="$repo_root/examples/baselines/screw_histbalance_hzl_301/diffusion/train_config.json"
    config_path="${CONFIG_PATH:-$default_config}"
    output_dir="${OUTPUT_DIR:-$output_base/diffusion_8_1}"
    ;;
  streaming_flow)
    default_config="$repo_root/examples/baselines/screw_histbalance_hzl_301/streaming_flow/train_config.json"
    config_path="${CONFIG_PATH:-$default_config}"
    output_dir="${OUTPUT_DIR:-$output_base/streaming_flow_8_1}"
    ;;
esac

if [[ -n "${PYTHON_BIN:-}" ]]; then
  [[ -x "${PYTHON_BIN}" ]] || { echo "PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2; exit 2; }
  python_cmd=("${PYTHON_BIN}")
elif [[ -x "$repo_root/.venv/bin/python" ]]; then
  python_cmd=("$repo_root/.venv/bin/python")
elif command -v uv >/dev/null 2>&1; then
  python_cmd=(uv run --project "$repo_root" python)
else
  python_cmd=(python3)
fi

[[ -f "$config_path" ]] || { echo "Training config not found: $config_path" >&2; exit 2; }
[[ -f "$data_root/meta/info.json" ]] || { echo "Dataset not found: $data_root" >&2; exit 2; }

all_episodes="$("${python_cmd[@]}" - "$config_path" "$data_root" "$variant" <<'PY'
import json
import sys
from pathlib import Path

config_path, data_root = map(Path, sys.argv[1:3])
variant = sys.argv[3]
config = json.loads(config_path.read_text(encoding="utf-8"))
policy = config["policy"]

if config.get("resume") or policy.get("pretrained_path") is not None:
    raise SystemExit("Baseline training must start from scratch")
if variant == "diffusion":
    assert policy["type"] == "diffusion"
    assert policy["n_obs_steps"] == 2
    assert policy["horizon"] == 8 and policy["n_action_steps"] == 1
    assert policy["noise_scheduler_type"] == "DDIM"
    assert policy["num_train_timesteps"] == 100 and policy["num_inference_steps"] == 10
else:
    assert policy["type"] == "streaming_flow_v2"
    assert policy["n_obs_steps"] == 2
    assert policy["chunk_size"] == 9 and policy["n_action_steps"] == 8
    assert policy["execution_horizon"] == 1 and policy["use_previous_action_alignment"]
    assert policy["sfp_num_train_points"] == 1
    assert policy["sfp_use_adaptive_freq"] is False
    assert policy["sfp_use_step_scaling"] is True

info = json.loads((data_root / "meta" / "info.json").read_text(encoding="utf-8"))
assert info["codebase_version"] == "v3.0" and info["fps"] == 30
assert info["total_episodes"] == 301, info["total_episodes"]
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
  save_freq=1
  log_freq=1
  batch_size=2
  num_workers=0
fi

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=disabled

args=(
  -m lerobot.scripts.lerobot_train
  "--config_path=$config_path"
  "--dataset.root=$data_root"
  "--dataset.repo_id=$dataset_repo_id"
  "--dataset.episodes=$all_episodes"
  "--output_dir=$output_dir"
  "--policy.device=$device"
  "--steps=$steps"
  "--save_freq=$save_freq"
  "--log_freq=$log_freq"
  "--batch_size=$batch_size"
  "--num_workers=$num_workers"
  "--eval_freq=0"
  "--wandb.enable=false"
)
args+=("$@")

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${python_cmd[@]}" "${args[@]}"
  printf '\n'
  exit 0
fi

if ! "${python_cmd[@]}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)'; then
  echo "NingAn training requires Python >= 3.12." >&2
  exit 2
fi
if ! "${python_cmd[@]}" -c 'import datasets, diffusers, draccus, safetensors, torch' 2>/dev/null; then
  echo "Training dependencies are missing. Run: cd $repo_root && uv sync --locked" >&2
  exit 2
fi
if [[ -e "$output_dir" ]]; then
  echo "OUTPUT_DIR already exists; refusing to overwrite: $output_dir" >&2
  exit 2
fi

echo "Variant          : $variant"
echo "Dataset          : $data_root"
echo "Episodes         : all 301 (0..300)"
echo "Output           : $output_dir"
echo "Steps/checkpoints: $steps / every $save_freq"
exec "${python_cmd[@]}" "${args[@]}"
