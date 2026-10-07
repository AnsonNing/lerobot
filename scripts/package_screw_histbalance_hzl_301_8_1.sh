#!/usr/bin/env bash
set -euo pipefail

# Package one selected 8->1 checkpoint for transfer to the inference machine.
# Usage: bash scripts/package_screw_histbalance_hzl_301_8_1.sh 15000
#        bash scripts/package_screw_histbalance_hzl_301_8_1.sh 20000

step="${1:?Usage: $0 15000|20000}"
case "$step" in
  15000|20000) ;;
  *) echo "Only the planned selection checkpoints 15000 and 20000 are accepted." >&2; exit 2 ;;
esac

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
run_root="${RUN_ROOT:-/ICRA/Output/streaming_flow_v2_screw_histbalance_hzl_301_8_1}"
delivery_root="${DELIVERY_ROOT:-$run_root/delivery}"
checkpoint="$run_root/checkpoints/$(printf '%06d' "$step")/pretrained_model"
bundle_name="screw_histbalance_hzl_301_8_1_step_${step}"
bundle_dir="$delivery_root/$bundle_name"
archive="$delivery_root/${bundle_name}.tar.gz"

[[ -d "$checkpoint" ]] || { echo "Checkpoint not found: $checkpoint" >&2; exit 2; }
[[ ! -e "$bundle_dir" && ! -e "$archive" ]] || {
  echo "Refusing to overwrite existing delivery artifact: $bundle_dir or $archive" >&2
  exit 2
}

mkdir -p "$delivery_root"
mkdir "$bundle_dir"
cp -a "$checkpoint" "$bundle_dir/8_1"
for report in "$run_root"/all301_training_diagnostic_*.json; do
  [[ -e "$report" ]] && cp -a "$report" "$bundle_dir/"
done
git -C "$repo_root" rev-parse HEAD > "$bundle_dir/lerobot_commit.txt"
printf 'selected_checkpoint_step=%s\n' "$step" > "$bundle_dir/manifest.txt"
printf 'selection_note=all301_training_diagnostic_is_not_held_out_validation\n' >> "$bundle_dir/manifest.txt"
(
  cd "$bundle_dir"
  find . -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.txt
)
tar -C "$delivery_root" -czf "$archive" "$bundle_name"

echo "Created: $archive"
echo "Transfer this archive to inference, then extract it and point CHECKPOINT at the extracted 8_1 directory."
