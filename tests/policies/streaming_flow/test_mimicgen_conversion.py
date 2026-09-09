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

import pytest

from lerobot.policies.streaming_flow.convert_mimicgen_hdf5_to_lerobot import (
    _dataset_features,
    _demo_sort_key,
    _episode_length,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


class _Array:
    def __init__(self, shape):
        self.shape = shape


class _Demo(dict):
    name = "/data/demo_0"


def _demo(length=5):
    return _Demo(
        {
            "actions": _Array((length, 7)),
            "obs/agentview_image": _Array((length, 84, 84, 3)),
            "obs/robot0_eye_in_hand_image": _Array((length, 84, 84, 3)),
            "obs/robot0_eef_pos": _Array((length, 3)),
            "obs/robot0_eef_quat": _Array((length, 4)),
            "obs/robot0_gripper_qpos": _Array((length, 2)),
        }
    )


def test_demo_sort_key_is_numeric():
    assert sorted(["demo_10", "demo_2", "demo_1"], key=_demo_sort_key) == [
        "demo_1",
        "demo_2",
        "demo_10",
    ]
    with pytest.raises(ValueError, match="demo_<integer>"):
        _demo_sort_key("episode_0")


def test_mimicgen_episode_contract_and_features():
    demo = _demo()
    assert _episode_length(demo) == 5
    features = _dataset_features(demo, use_videos=True)
    assert features[ACTION]["shape"] == (7,)
    assert features[OBS_STATE]["shape"] == (9,)
    assert features[f"{OBS_IMAGES}.agentview_image"]["shape"] == (84, 84, 3)
    assert features[f"{OBS_IMAGES}.agentview_image"]["dtype"] == "video"


def test_mimicgen_episode_rejects_length_mismatch():
    demo = _demo()
    demo["obs/robot0_gripper_qpos"] = _Array((4, 2))
    with pytest.raises(ValueError, match="inconsistent trajectory lengths"):
        _episode_length(demo)
