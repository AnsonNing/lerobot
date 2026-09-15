#!/usr/bin/env bash
set -euo pipefail

inference_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${inference_dir}/.." && pwd)"
project_dir="$(cd "${repo_dir}/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
checkpoint="${CHECKPOINT:-${project_dir}/checkpoint/pretrained_model_8_1}"
device="${DEVICE:-cuda}"
duration="${DURATION:-10}"
max_relative_target="${MAX_RELATIVE_TARGET:-5}"
robot_port="${ROBOT_PORT:-/dev/ttyACM0}"
front_camera="${FRONT_CAMERA:-4}"
side_camera="${SIDE_CAMERA:-6}"

if [[ ! -f "${checkpoint}/config.json" || ! -f "${checkpoint}/model.safetensors" ]]; then
    echo "Checkpoint is incomplete: ${checkpoint}" >&2
    exit 2
fi

"${python_bin}" -c 'import sys; assert sys.version_info >= (3, 10), "NingAn inference requires Python >= 3.10"'

if [[ "${device}" == cuda* ]]; then
    "${python_bin}" -c 'import torch; assert torch.cuda.is_available(), "CUDA was requested but torch.cuda.is_available() is false"'
fi

export PYTHONPATH="${repo_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${python_bin}" "${inference_dir}/lerobot_ningan_inference.py" \
    --checkpoint="${checkpoint}" \
    --device="${device}" \
    --duration="${duration}" \
    --robot-port="${robot_port}" \
    --front-camera="${front_camera}" \
    --side-camera="${side_camera}" \
    --max-relative-target="${max_relative_target}" \
    "$@"
