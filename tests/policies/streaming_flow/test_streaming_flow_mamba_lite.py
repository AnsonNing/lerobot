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

import pytest
import torch
from torch import nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.policies.streaming_flow import modeling_streaming_flow_mamba_lite
from lerobot.policies.streaming_flow.configuration_streaming_flow import (
    StreamingFlowMambaLiteConfig,
    StreamingFlowV2Config,
)
from lerobot.policies.streaming_flow.modeling_streaming_flow_mamba_lite import (
    StreamingFlowMambaLiteModel,
    StreamingFlowMambaLitePolicy,
)
from lerobot.policies.streaming_flow.modeling_streaming_flow_v2 import (
    StreamingFlowModel as StreamingFlowV2Model,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


class _FakeTemporalImageEncoder(nn.Module):
    def __init__(self, config: StreamingFlowMambaLiteConfig):
        super().__init__()
        self.feature_dim = config.image_feature_dim
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        feature = observations.mean(dim=(1, 2, 3, 4), keepdim=False).unsqueeze(-1)
        return feature.expand(-1, self.feature_dim) * self.scale


def _config(**overrides) -> StreamingFlowMambaLiteConfig:
    values = {
        "input_features": {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            "observation.images.cam0": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 16, 16),
            ),
            "observation.images.cam1": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 16, 16),
            ),
        },
        "output_features": {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        },
        "chunk_size": 8,
        "n_action_steps": 4,
        "sfp_num_train_points": 4,
        "image_feature_dim": 16,
        "embedding_dim": 16,
        "mamba_hidden_dim": 32,
        "mamba_current_depth": 2,
        "mamba_d_state": 4,
        "mamba_mimo_rank": 2,
        "mamba_dropout": 0.0,
        "mamba_use_cuda_kernel": False,
        "previous_tail_len": 3,
        "cross_attention_heads": 4,
        "history_dropout_prob": 0.0,
        "history_noise_std": 0.0,
        "lite_num_tasks": 8,
        "lite_task_embedding_dim": 8,
        "pretrained_backbone_weights": None,
        "use_ema": False,
    }
    values.update(overrides)
    return StreamingFlowMambaLiteConfig(**values)


def _model_batch(
    config: StreamingFlowMambaLiteConfig,
    batch_size: int = 2,
) -> dict[str, torch.Tensor]:
    return {
        OBS_STATE: torch.randn(batch_size, config.n_obs_steps, 8),
        OBS_IMAGES: torch.rand(batch_size, config.n_obs_steps, 2, 3, 16, 16),
        "task_index": torch.tensor([[1], [3]])[:batch_size],
        ACTION: torch.randn(batch_size, len(config.action_delta_indices), 7),
    }


def test_factory_routes_lite_policy_through_streaming_flow_v2() -> None:
    assert get_policy_class("streaming_flow_mamba_lite") is StreamingFlowMambaLitePolicy
    config = make_policy_config("streaming_flow_mamba_lite")
    assert isinstance(config, StreamingFlowMambaLiteConfig)
    assert isinstance(config, StreamingFlowV2Config)
    assert config.mamba_mimo_rank == 4


def test_lite_conditioning_uses_camera_state_and_task_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        modeling_streaming_flow_mamba_lite,
        "TemporalImageEncoder",
        _FakeTemporalImageEncoder,
    )
    config = _config()
    model = StreamingFlowMambaLiteModel(config)
    batch = _model_batch(config)

    assert isinstance(model, StreamingFlowV2Model)
    raw, normalized, tokens, mask = model._prepare_token_conditioning(batch)

    expected_raw_dim = 2 * config.image_feature_dim + 2 * 8 + config.lite_task_embedding_dim
    assert raw.shape == (2, expected_raw_dim)
    assert normalized.shape == raw.shape
    assert tokens.shape == (2, 5, config.transformer_hidden_dim)
    assert mask.shape == tokens.shape[:2]
    assert mask.all()
    torch.testing.assert_close(normalized.norm(dim=-1), torch.ones(2))


def test_lite_multi_point_training_and_incremental_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        modeling_streaming_flow_mamba_lite,
        "TemporalImageEncoder",
        _FakeTemporalImageEncoder,
    )
    torch.manual_seed(5)
    config = _config()
    model = StreamingFlowMambaLiteModel(config)
    batch = _model_batch(config)

    loss, metrics = model.compute_loss(batch)
    loss.backward()

    assert metrics is not None
    assert metrics["sfp_num_train_points"] == config.sfp_num_train_points
    assert model.task_embedding.weight.grad is not None
    assert isinstance(model.rgb_encoder, nn.ModuleList)
    assert all(encoder.scale.grad is not None for encoder in model.rgb_encoder)

    model.eval()
    with torch.no_grad():
        actions = model.generate_actions(batch)
        integrated, final_action, rollout_metrics = model.integrate_actions(batch)

    assert actions.shape == (2, config.n_action_steps, 7)
    assert integrated.shape == actions.shape
    assert final_action.shape == (2, 1, 7)
    assert rollout_metrics["inference_mode"] == "incremental"


def test_streaming_flow_v2_environment_select_action_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        modeling_streaming_flow_mamba_lite,
        "TemporalImageEncoder",
        _FakeTemporalImageEncoder,
    )
    config = _config(n_action_steps=2)
    policy = StreamingFlowMambaLitePolicy(config).eval()
    rollout_observation = {
        OBS_STATE: torch.randn(1, 8),
        "observation.images.cam0": torch.rand(1, 3, 16, 16),
        "observation.images.cam1": torch.rand(1, 3, 16, 16),
        "task_index": torch.tensor([2]),
    }

    first = policy.select_action(rollout_observation)
    second = policy.select_action(rollout_observation)

    assert first.shape == (1, 7)
    assert second.shape == (1, 7)
    assert len(policy._executed_action_history) == 2
    assert policy.get_rollout_info()["chunk_index"] == 0


def test_streaming_flow_v2_training_wrapper_stacks_named_cameras(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        modeling_streaming_flow_mamba_lite,
        "TemporalImageEncoder",
        _FakeTemporalImageEncoder,
    )
    config = _config()
    policy = StreamingFlowMambaLitePolicy(config)
    training_batch = {
        OBS_STATE: torch.randn(2, config.n_obs_steps, 8),
        "observation.images.cam0": torch.rand(2, config.n_obs_steps, 3, 16, 16),
        "observation.images.cam1": torch.rand(2, config.n_obs_steps, 3, 16, 16),
        "task_index": torch.tensor([1, 3]),
        ACTION: torch.randn(2, len(config.action_delta_indices), 7),
    }

    loss, metrics = policy(training_batch)

    assert loss.ndim == 0
    assert metrics is not None
    assert metrics["sfp_num_train_points"] == config.sfp_num_train_points


def test_single_task_environment_can_fall_back_to_task_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        modeling_streaming_flow_mamba_lite,
        "TemporalImageEncoder",
        _FakeTemporalImageEncoder,
    )
    config = _config()
    model = StreamingFlowMambaLiteModel(config)
    batch = _model_batch(config)
    del batch["task_index"]

    with pytest.raises(ValueError, match="task_index"):
        model._prepare_token_conditioning(batch)

    model.eval()
    raw, _, _, _ = model._prepare_token_conditioning(batch)
    assert raw.shape[0] == 2


def test_lite_rejects_cuda_settings_that_would_silently_fallback() -> None:
    with pytest.raises(ValueError, match="fused Mamba3 CUDA step"):
        _config(mamba_use_cuda_kernel=True, mamba_mimo_rank=2)
