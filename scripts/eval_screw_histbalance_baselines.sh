#!/usr/bin/env bash
set -euo pipefail

# Recompute deterministic checkpoint loss for 15k and 20k on all 301 episodes.
# Usage: bash scripts/eval_screw_histbalance_baselines.sh all|diffusion|streaming_flow

selection="${1:-all}"
case "$selection" in
  all) variants=(diffusion streaming_flow) ;;
  diffusion|streaming_flow) variants=("$selection") ;;
  *) echo "Usage: $0 all|diffusion|streaming_flow" >&2; exit 2 ;;
esac

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
default_data_root="/home/lerobot/NingAn/lerobot/NingAn_Inference/Pull_Screw/cut_datasets/"
default_data_root+="histogram_balance_cut_pull_screw_all_hover_gripper_25_assemble_episode"
data_root="${DATA_ROOT:-$default_data_root}"
default_repo_id="HZL/histogram_balance_cut_pull_screw_all_hover_gripper_25_assemble_episode"
dataset_repo_id="${DATASET_REPO_ID:-$default_repo_id}"
output_base="${OUTPUT_BASE:-$repo_root/outputs/screw_histbalance_baselines}"
device="${DEVICE:-cuda}"
batch_size="${BATCH_SIZE:-16}"
num_workers="${NUM_WORKERS:-2}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_cmd=("${PYTHON_BIN}")
elif [[ -x "$repo_root/.venv/bin/python" ]]; then
  python_cmd=("$repo_root/.venv/bin/python")
elif command -v uv >/dev/null 2>&1; then
  python_cmd=(uv run --project "$repo_root" python)
else
  python_cmd=(python3)
fi
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

for variant in "${variants[@]}"; do
  case "$variant" in
    diffusion) run_name=diffusion_8_1 ;;
    streaming_flow) run_name=streaming_flow_8_1 ;;
  esac
  run_dir="$output_base/$run_name"
  for step in 015000 020000; do
    checkpoint="$run_dir/checkpoints/$step/pretrained_model"
    report="$run_dir/all301_checkpoint_loss_${step}.json"
    temporary_report="${report}.tmp"
    [[ -d "$checkpoint" ]] || { echo "Checkpoint not found: $checkpoint" >&2; exit 2; }
    [[ ! -e "$report" && ! -e "$temporary_report" ]] || {
      echo "Refusing to overwrite an existing report: $report or $temporary_report" >&2
      exit 2
    }
    "${python_cmd[@]}" "$repo_root/scripts/eval_screw_histbalance_baseline_loss.py" \
      "$checkpoint" \
      --root "$data_root" \
      --repo-id "$dataset_repo_id" \
      --device "$device" \
      --batch-size "$batch_size" \
      --num-workers "$num_workers" \
      > "$temporary_report"
    mv "$temporary_report" "$report"
    echo "Wrote: $report"
  done
done
