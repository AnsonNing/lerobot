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

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.streaming_flow.configuration_streaming_flow import (
    StreamingFlowMambaLiteConfig,
    StreamingFlowV2Config,
    StreamingFlowV3Config,
)
from lerobot.policies.streaming_flow.modeling_streaming_flow_v2 import StreamingFlowModel, StreamingFlowPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


def _policy_for_initialization(
    action_dim: int,
    mode: str = "auto",
    initial_action: tuple[float, ...] | None = None,
    action_min: list[float] | None = None,
    action_max: list[float] | None = None,
) -> StreamingFlowPolicy:
    policy = StreamingFlowPolicy.__new__(StreamingFlowPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        action_feature=PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
        rollout_initial_action_mode=mode,
        rollout_initial_action=initial_action,
    )
    policy.register_buffer("_action_min", torch.tensor(action_min or []), persistent=True)
    policy.register_buffer("_action_max", torch.tensor(action_max or []), persistent=True)
    return policy


def _image_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {OBS_IMAGES: torch.zeros(batch_size, 2, 1, 3, 8, 8)}


def test_auto_initialization_preserves_pusht_notebook_center() -> None:
    policy = _policy_for_initialization(2, action_min=[0.0, 0.0], action_max=[512.0, 512.0])

    initial_action, source = policy._initial_rollout_action(_image_batch())

    assert source == "notebook_fixed_center"
    torch.testing.assert_close(initial_action, torch.zeros(2, 1, 2))


def test_auto_initialization_uses_normalized_neutral_action_for_non_2d_actions() -> None:
    policy = _policy_for_initialization(7, action_min=[-2.0] * 7, action_max=[6.0] * 7)

    initial_action, source = policy._initial_rollout_action(_image_batch())

    assert source == "zero_action"
    torch.testing.assert_close(initial_action, torch.full((2, 1, 7), -0.5))


def test_constant_initialization_supports_dataset_specific_raw_action() -> None:
    policy = _policy_for_initialization(
        4,
        mode="constant",
        initial_action=(0.0, 1.0, 2.0, 3.0),
        action_min=[0.0] * 4,
        action_max=[4.0] * 4,
    )

    initial_action, source = policy._initial_rollout_action(_image_batch())

    assert source == "configured_constant"
    torch.testing.assert_close(initial_action[0, 0], torch.tensor([-1.0, -0.5, 0.0, 0.5]))


def test_state_initialization_uses_latest_processed_state() -> None:
    policy = _policy_for_initialization(4, mode="state")
    batch = _image_batch()
    batch[OBS_STATE] = torch.tensor(
        [
            [[-0.1, -0.2, -0.3, -0.4], [0.1, 0.2, 0.3, 0.4]],
            [[-0.5, -0.6, -0.7, -0.8], [0.5, 0.6, 0.7, 0.8]],
        ]
    )

    initial_action, source = policy._initial_rollout_action(batch)

    assert source == "state_action"
    torch.testing.assert_close(initial_action, batch[OBS_STATE][:, -1:, :])


def test_constant_mode_requires_an_initial_action() -> None:
    with pytest.raises(ValueError, match="rollout_initial_action"):
        StreamingFlowV2Config(rollout_initial_action_mode="constant")


def test_v2_can_align_previous_action_and_execute_a_single_predicted_step() -> None:
    config = StreamingFlowV2Config(
        use_previous_action_alignment=True,
        execution_horizon=1,
    )

    assert config.use_previous_action_alignment is True
    assert config.n_action_steps == 8
    assert config.effective_execution_horizon == 1
    assert config.action_delta_indices[0] == -1

    policy = StreamingFlowPolicy.__new__(StreamingFlowPolicy)
    nn.Module.__init__(policy)
    policy.config = config
    policy.reset()
    assert policy._queues[ACTION].maxlen == 1


@pytest.mark.parametrize("execution_horizon", [0, 9])
def test_execution_horizon_must_fit_predicted_prefix(execution_horizon: int) -> None:
    with pytest.raises(ValueError, match="execution_horizon"):
        StreamingFlowV2Config(n_action_steps=8, execution_horizon=execution_horizon)


def test_previous_action_alignment_uses_matching_training_and_rollout_intervals() -> None:
    model = StreamingFlowModel.__new__(StreamingFlowModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        chunk_size=16,
        n_obs_steps=2,
        use_previous_action_alignment=True,
    )

    assert model._trajectory_start_index(action_sequence_length=16) == 0
    assert model._integration_dt() == pytest.approx(1 / 15)

    model.config.use_previous_action_alignment = False
    assert model._trajectory_start_index(action_sequence_length=16) == 1
    assert model._integration_dt() == pytest.approx(1 / 14)


def test_policy_continuity_tracks_last_executed_action() -> None:
    policy = StreamingFlowPolicy.__new__(StreamingFlowPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        action_feature=PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    )
    executed = torch.arange(21, dtype=torch.float32).reshape(1, 3, 7)

    policy.set_executed_action_state(executed)

    torch.testing.assert_close(policy._prev_action_state, executed[:, -1:])


def test_resnet_variants_default_to_separate_full_camera_encoders() -> None:
    for config in (StreamingFlowV2Config(), StreamingFlowMambaLiteConfig()):
        assert config.use_separate_rgb_encoder_per_camera is True
        assert config.freeze_vision_encoder is False

    # CLIP variants retain the existing shared/frozen CLIP defaults.
    clip_config = StreamingFlowV3Config()
    assert clip_config.use_separate_rgb_encoder_per_camera is False
    assert clip_config.sfp_freeze_clip is True
    assert clip_config.use_previous_action_alignment is False
    assert clip_config.effective_execution_horizon == clip_config.n_action_steps


def test_direct_frequency_prediction_reports_raw_and_eval_clamped_values() -> None:
    config = SimpleNamespace(sfp_use_adaptive_freq=True, sfp_freq_min=0.2, sfp_freq_max=5.0)
    model = StreamingFlowModel.__new__(StreamingFlowModel)
    nn.Module.__init__(model)
    model.config = config
    model.velocity_model = SimpleNamespace(
        granularity_predictor=lambda _cond: torch.tensor([-1.0, 3.0, 8.0])
    )
    raw_cond = torch.zeros(3, 2)

    raw_freq, training_freq, clamped_freq = model._predict_frequency(raw_cond, clamp=False)
    _, eval_freq, _ = model._predict_frequency(raw_cond, clamp=True)

    torch.testing.assert_close(raw_freq, torch.tensor([-1.0, 3.0, 8.0]))
    torch.testing.assert_close(training_freq, raw_freq)
    torch.testing.assert_close(clamped_freq, torch.tensor([0.2, 3.0, 5.0]))
    torch.testing.assert_close(eval_freq, clamped_freq)


def test_rollout_frequency_diagnostics_are_exposed_to_eval() -> None:
    policy = StreamingFlowPolicy.__new__(StreamingFlowPolicy)
    nn.Module.__init__(policy)
    policy._rollout_chunk_index = -1
    policy._last_rollout_info = None

    policy._record_rollout_info(
        {"pred_freq": 0.2, "raw_pred_freq": -0.4, "clamped_pred_freq": 0.2}
    )

    assert policy.get_rollout_info() == {
        "pred_freq": 0.2,
        "raw_pred_freq": -0.4,
        "clamped_pred_freq": 0.2,
        "chunk_index": 0,
    }
