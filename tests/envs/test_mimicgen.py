# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import numpy as np
import pytest

from lerobot.envs.configs import MimicGenEnv
from lerobot.envs.mimicgen import MimicGenGymEnv, _image_to_hwc_uint8, canonical_mimicgen_task
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


def _raw_obs() -> dict[str, np.ndarray]:
    return {
        "agentview_image": np.full((3, 84, 84), 0.5, dtype=np.float32),
        "robot0_eye_in_hand_image": np.full((3, 84, 84), 0.25, dtype=np.float32),
        "robot0_eef_pos": np.arange(3, dtype=np.float64),
        "robot0_eef_quat": np.arange(4, dtype=np.float64),
        "robot0_gripper_qpos": np.arange(2, dtype=np.float64),
    }


class _FakeRobomimicEnv:
    def __init__(self):
        self.reset_calls = 0
        self.reset_to_calls = 0

    def reset(self):
        self.reset_calls += 1
        return _raw_obs()

    def reset_to(self, state):
        assert state["states"].shape == (3,)
        self.reset_to_calls += 1
        return _raw_obs()

    def get_state(self):
        return {"states": np.arange(3, dtype=np.float64)}

    def step(self, action):
        return _raw_obs(), 1.0, False, {"backend": True}

    def is_success(self):
        return {"task": True}

    def close(self):
        pass


def test_mimicgen_config_contract():
    cfg = MimicGenEnv(dataset_path="/tmp/threading_d1.hdf5")
    assert cfg.type == "mimicgen"
    assert cfg.fps == 20
    assert cfg.features[ACTION].shape == (7,)
    assert cfg.features["agent_pos"].shape == (9,)
    assert cfg.features_map["agent_pos"] == OBS_STATE
    assert cfg.features_map["pixels/agentview_image"] == f"{OBS_IMAGES}.agentview_image"


def test_canonical_mimicgen_task():
    assert canonical_mimicgen_task("square_d1") == "Square_D1"
    assert canonical_mimicgen_task("Threading_D1") == "Threading_D1"
    with pytest.raises(ValueError, match="Unsupported MimicGen task"):
        canonical_mimicgen_task("Coffee_D1")


def test_image_to_hwc_uint8():
    image = _image_to_hwc_uint8(np.full((3, 84, 84), 0.5, dtype=np.float32))
    assert image.shape == (84, 84, 3)
    assert image.dtype == np.uint8
    assert image[0, 0, 0] == 128


def test_seeded_reset_cache_and_step(tmp_path):
    dataset_path = tmp_path / "threading_d1.hdf5"
    dataset_path.touch()
    env = MimicGenGymEnv(task="Threading_D1", dataset_path=dataset_path)
    backend = _FakeRobomimicEnv()
    env._env = backend

    observation, info = env.reset(seed=100000)
    assert observation["agent_pos"].shape == (9,)
    assert observation["pixels"]["agentview_image"].shape == (84, 84, 3)
    assert info["is_success"] is False
    env.reset(seed=100000)
    assert backend.reset_calls == 1
    assert backend.reset_to_calls == 1

    _, reward, terminated, truncated, info = env.step(np.zeros(7, dtype=np.float32))
    assert reward == 1.0
    assert terminated is True
    assert truncated is False
    assert info["is_success"] is True
