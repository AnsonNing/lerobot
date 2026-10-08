#!/usr/bin/env bash
set -euo pipefail

# Package a selected 15k or 20k baseline checkpoint for the inference machine.
# Usage:
#   bash scripts/package_screw_histbalance_baseline.sh diffusion 15000
#   bash scripts/package_screw_histbalance_baseline.sh streaming_flow 20000

variant="${1:-}"
step="${2:-}"
case "${variant}" in
  diffusion|streaming_flow) ;;
  *) echo "Usage: $0 diffusion|streaming_flow 15000|20000" >&2; exit 2 ;;
esac
case "${step}" in
  15000|20000) ;;
  *) echo "Only checkpoint 15000 or 20000 may be packaged." >&2; exit 2 ;;
esac

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output_base="${OUTPUT_BASE:-$repo_root/outputs/screw_histbalance_baselines}"
case "${variant}" in
  diffusion)
    run_name="diffusion_8_1"
    bundle_name="screw_histbalance_diffusion_8_1_step_${step}"
    ;;
  streaming_flow)
    run_name="streaming_flow_8_1"
    bundle_name="screw_histbalance_streaming_flow_baseline_8_1_step_${step}"
    ;;
esac

run_dir="$output_base/$run_name"
checkpoint="$run_dir/checkpoints/$(printf '%06d' "$step")/pretrained_model"
delivery_root="${DELIVERY_ROOT:-$output_base/delivery}"
bundle_dir="$delivery_root/$bundle_name"
archive="$delivery_root/${bundle_name}.tar.gz"
archive_checksum="${archive}.sha256"

[[ -d "$checkpoint" ]] || { echo "Checkpoint not found: $checkpoint" >&2; exit 2; }
[[ ! -e "$bundle_dir" && ! -e "$archive" && ! -e "$archive_checksum" ]] || {
  echo "Refusing to overwrite an existing delivery artifact for $bundle_name" >&2
  exit 2
}

mkdir -p "$delivery_root"
mkdir "$bundle_dir"
cp -a "$checkpoint" "$bundle_dir/pretrained_model"
cp -a "$run_dir/train_metrics.jsonl" "$bundle_dir/train_metrics.jsonl"
if [[ -f "$output_base/loss_comparison.json" ]]; then
  cp -a "$output_base/loss_comparison.json" "$bundle_dir/loss_comparison.json"
fi
for report in "$run_dir"/all301_checkpoint_loss_*.json; do
  [[ -e "$report" ]] && cp -a "$report" "$bundle_dir/"
done

git -C "$repo_root" rev-parse HEAD > "$bundle_dir/lerobot_commit.txt"
{
  printf 'variant=%s\n' "$variant"
  printf 'selected_checkpoint_step=%s\n' "$step"
  printf 'training_episodes=0..300\n'
  printf 'training_steps=20000\n'
  printf 'selection_note=training_loss_is_not_held_out_validation\n'
} > "$bundle_dir/manifest.txt"
(
  cd "$bundle_dir"
  find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.txt
)
tar -C "$delivery_root" -czf "$archive" "$bundle_name"
(
  cd "$delivery_root"
  sha256sum "${bundle_name}.tar.gz" > "${bundle_name}.tar.gz.sha256"
)

echo "Created: $archive"
echo "Created: $archive_checksum"
echo "After extraction, point CHECKPOINT at: $bundle_name/pretrained_model"
