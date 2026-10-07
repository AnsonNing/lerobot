"""Check that preserved recipes can be rebased without starting training."""

import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import draccus
import pytest

from lerobot.configs.train import TrainPipelineConfig

REPO = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO / "examples/streaming_flow_v2/experiments"


@pytest.fixture
def prepared_metadata(tmp_path):
    root = tmp_path / "dataset with spaces"
    (root / "meta").mkdir(parents=True)
    names = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]
    features = {key: {"shape": [6], "names": names} for key in ("action", "observation.state")}
    features.update(
        {f"observation.images.{camera}": {"shape": [480, 640, 3]} for camera in ("front", "side")}
    )
    (root / "meta/info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "fps": 30,
                "total_episodes": 3,
                "features": features,
            }
        )
    )
    split = root / "meta/gripper25_split_20261005.json"
    split.write_text(json.dumps({"train": [0, 1], "validation": [2]}))
    return root, split


@pytest.mark.parametrize("variant,horizon", [("8_1", 8), ("16_1", 16)])
@pytest.mark.parametrize("smoke", [False, True])
def test_portable_recipe_rebases_paths_and_preserves_training_contract(
    prepared_metadata, tmp_path, variant, horizon, smoke
):
    root, split = prepared_metadata
    output = tmp_path / "new run with spaces"
    env = os.environ | {
        "DATA_ROOT": str(root),
        "SPLIT_FILE": str(split),
        "OUTPUT_DIR": str(output),
        "DATASET_REPO_ID": "handoff/screw",
        "PYTHON_BIN": sys.executable,
        "DEVICE": "cpu",
        "DRY_RUN": "1",
        "SMOKE": str(int(smoke)),
        "CONFIG_PATH": str(EXPERIMENTS / "screw_gripper25_20261005" / variant / "train_config.json"),
    }
    # Run outside the checkout so no original-host or cwd assumptions are hidden.
    result = subprocess.run(
        ["bash", str(REPO / "scripts/train_streaming_flow_v2.sh"), variant],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    command = shlex.split(result.stdout)
    cfg = draccus.parse(TrainPipelineConfig, args=command[3:])
    cfg.validate()
    assert Path(cfg.dataset.root) == root and cfg.dataset.repo_id == "handoff/screw"
    assert cfg.dataset.episodes == [0, 1] and cfg.output_dir == output
    assert cfg.policy.n_action_steps == horizon and cfg.policy.chunk_size == horizon + 1
    assert cfg.policy.execution_horizon == 1 and cfg.policy.use_previous_action_alignment
    assert cfg.policy.use_separate_rgb_encoder_per_camera and not cfg.policy.freeze_vision_encoder
    assert cfg.policy.action_normalization_mode == "per_dim"
    assert cfg.policy.crop_shape == (228, 304)
    assert cfg.seed == 100000 and cfg.optimizer.lr == 1e-4
    assert cfg.sample_weighting is None and not cfg.resume
    assert cfg.steps == (1 if smoke else 20000)
    assert cfg.batch_size == (2 if smoke else 16)
    assert not output.exists(), "Dry run must not create a training output"


def test_portable_recipe_rejects_overlapping_split_before_training(prepared_metadata, tmp_path):
    root, split = prepared_metadata
    split.write_text(json.dumps({"train": [0, 1], "validation": [1, 2]}))
    env = os.environ | {
        "DATA_ROOT": str(root),
        "OUTPUT_DIR": str(tmp_path / "run"),
        "PYTHON_BIN": sys.executable,
        "DRY_RUN": "1",
    }
    result = subprocess.run(
        ["bash", str(REPO / "scripts/train_streaming_flow_v2.sh"), "8_1"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "Training/validation overlap" in result.stderr
    assert not (tmp_path / "run").exists()


def test_historical_snapshots_match_recorded_sha256():
    manifest = json.loads((EXPERIMENTS / "PROVENANCE.json").read_text())
    for experiment, files in manifest["experiments"].items():
        for name, provenance in files.items():
            assert (
                hashlib.sha256((EXPERIMENTS / experiment / name).read_bytes()).hexdigest()
                == provenance["sha256"]
            )
