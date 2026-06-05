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
from lerobot.policies.streaming_flow.configuration_streaming_flow import StreamingFlowV5Config
from lerobot.policies.streaming_flow import modeling_streaming_flow_v5
from lerobot.policies.streaming_flow.modeling_streaming_flow_v5 import (
    AdaLNZeroCrossAttentionBlock,
    AdaptiveStreamingFlowTransformer,
    StepScalingLayer,
    StreamingFlowModel,
    StreamingFlowPolicy,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


def _config() -> StreamingFlowV5Config:
    return StreamingFlowV5Config(
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        transformer_hidden_dim=32,
        transformer_num_layers=2,
        transformer_num_heads=4,
        transformer_ffn_dim=64,
        transformer_dropout=0.0,
        transformer_visual_tokens_per_frame=4,
        embedding_dim=16,
        use_ema=False,
    )


def test_visual_token_pooling_requires_square_token_count() -> None:
    with pytest.raises(ValueError, match="transformer_visual_tokens_per_frame"):
        StreamingFlowV5Config(transformer_visual_tokens_per_frame=6)


def test_v5_frequency_init_must_be_within_eval_range() -> None:
    with pytest.raises(ValueError, match="sfp_freq_init"):
        StreamingFlowV5Config(sfp_freq_min=0.2, sfp_freq_max=5.0, sfp_freq_init=0.1)


def test_v5_vision_encoder_lr_multiplier_must_be_positive() -> None:
    with pytest.raises(ValueError, match="vision_encoder_lr_multiplier"):
        StreamingFlowV5Config(vision_encoder_lr_multiplier=0.0)


def test_step_scaling_is_explicitly_proportional_to_frequency() -> None:
    layer = StepScalingLayer(8)
    embedding = torch.randn(2, 8)
    low = layer(embedding, torch.tensor([0.5, 1.0]))
    high = layer(embedding, torch.tensor([1.0, 2.0]))

    torch.testing.assert_close(high, 2.0 * low)


def test_v5_direct_frequency_prediction_only_clamps_when_requested() -> None:
    config = _config()
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
    torch.testing.assert_close(clamped_freq, torch.tensor([config.sfp_freq_min, 3.0, config.sfp_freq_max]))
    torch.testing.assert_close(eval_freq, clamped_freq)


def test_frequency_delta_scales_transformer_block_residual_updates() -> None:
    config = _config()
    block = AdaLNZeroCrossAttentionBlock(config).eval()
    hidden_dim = config.transformer_hidden_dim
    with torch.no_grad():
        block.adaLN_modulation[-1].bias[2 * hidden_dim : 3 * hidden_dim].fill_(1.0)
        block.adaLN_modulation[-1].bias[5 * hidden_dim : 6 * hidden_dim].fill_(1.0)
        block.adaLN_modulation[-1].bias[8 * hidden_dim : 9 * hidden_dim].fill_(1.0)

    x = torch.randn(1, 1, hidden_dim)
    memory = torch.randn(1, 4, hidden_dim)
    mask = torch.zeros(1, 4, dtype=torch.bool)
    condition = torch.randn(1, hidden_dim)

    no_update = block(x, memory, mask, condition, delta=torch.zeros(1))
    scaled_update = block(x, memory, mask, condition, delta=torch.ones(1))

    torch.testing.assert_close(no_update, x)
    assert not torch.allclose(scaled_update, x)


def test_transformer_velocity_expert_supports_token_memory_and_backward() -> None:
    config = _config()
    model = AdaptiveStreamingFlowTransformer(config, global_cond_dim=12)
    batch_size = 2
    output = model(
        sample=torch.randn(batch_size, 1, 7),
        timestep=torch.rand(batch_size),
        global_cond=torch.randn(batch_size, 12),
        context_tokens=torch.randn(batch_size, 9, config.transformer_hidden_dim),
        context_mask=torch.ones(batch_size, 9, dtype=torch.bool),
        freq=torch.ones(batch_size),
    )

    assert output.shape == (batch_size, 1, 7)
    output.square().mean().backward()
    assert model.blocks[0].adaLN_modulation[-1].weight.grad is not None


def test_learned_cross_attention_gate_uses_context_tokens() -> None:
    config = _config()
    model = AdaptiveStreamingFlowTransformer(config, global_cond_dim=12).eval()
    hidden_dim = config.transformer_hidden_dim
    with torch.no_grad():
        for block in model.blocks:
            block.adaLN_modulation[-1].bias[5 * hidden_dim : 6 * hidden_dim].fill_(1.0)

    kwargs = {
        "sample": torch.randn(1, 1, 7),
        "timestep": torch.tensor([0.5]),
        "global_cond": torch.randn(1, 12),
        "context_mask": torch.ones(1, 5, dtype=torch.bool),
        "freq": torch.ones(1),
    }
    first = model(context_tokens=torch.zeros(1, 5, hidden_dim), **kwargs)
    second = model(context_tokens=torch.ones(1, 5, hidden_dim), **kwargs)

    assert not torch.allclose(first, second)


class _FakeVisionEncoder(nn.Module):
    config = SimpleNamespace(hidden_size=24)

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    @classmethod
    def from_pretrained(cls, _model_name: str):
        return cls()

    def forward(self, pixel_values: torch.Tensor, output_hidden_states: bool = False):
        del output_hidden_states
        batch_size = pixel_values.shape[0]
        hidden = pixel_values.mean(dim=(1, 2, 3), keepdim=True).reshape(batch_size, 1, 1) * self.scale
        return SimpleNamespace(last_hidden_state=hidden.expand(batch_size, 197, self.config.hidden_size))


class _FakeTextEncoder(nn.Module):
    config = SimpleNamespace(hidden_size=24)

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    @classmethod
    def from_pretrained(cls, _model_name: str):
        return cls()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        del attention_mask
        hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, self.config.hidden_size) * self.scale
        return SimpleNamespace(last_hidden_state=hidden, pooler_output=hidden[:, 0])


def test_token_conditioning_drives_sfp_loss_and_rollout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(modeling_streaming_flow_v5, "CLIPVisionModel", _FakeVisionEncoder)
    monkeypatch.setattr(modeling_streaming_flow_v5, "CLIPTextModel", _FakeTextEncoder)
    config = StreamingFlowV5Config(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            f"{OBS_IMAGES}.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        chunk_size=8,
        n_action_steps=4,
        transformer_hidden_dim=32,
        transformer_num_layers=2,
        transformer_num_heads=4,
        transformer_ffn_dim=64,
        transformer_dropout=0.0,
        transformer_visual_tokens_per_frame=4,
        embedding_dim=16,
        image_feature_dim=16,
        clip_text_projection_dim=8,
        clip_image_crop_shape=(32, 32),
        use_ema=False,
    )
    model = StreamingFlowModel(config)
    batch_size = 2
    batch = {
        OBS_STATE: torch.randn(batch_size, 2, 8),
        OBS_IMAGES: torch.rand(batch_size, 2, 1, 3, 32, 32),
        OBS_LANGUAGE_TOKENS: torch.randint(0, 10, (batch_size, 6)),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(batch_size, 6, dtype=torch.long),
        ACTION: torch.randn(batch_size, 8, 7),
    }

    _, _, tokens, mask = model._prepare_token_conditioning(batch)
    assert tokens.shape == (batch_size, 16, config.transformer_hidden_dim)
    assert mask.all()

    loss, metrics = model.compute_loss(batch)
    assert metrics is not None
    assert metrics["pred_freq"] == pytest.approx(config.sfp_freq_init)
    assert metrics["clamped_pred_freq"] == pytest.approx(config.sfp_freq_init)
    assert config.sfp_freq_min < metrics["pred_freq"] < config.sfp_freq_max
    loss.backward()
    assert model.rgb_encoder.model.scale.grad is not None
    assert not model.text_encoder.text_encoder.scale.requires_grad
    assert model.velocity_model.granularity_predictor.net[-1].weight.grad is not None
    actions, final_action, rollout_metrics = model.integrate_actions(batch)
    assert actions.shape == (batch_size, config.n_action_steps, 7)
    assert final_action.shape == (batch_size, 1, 7)
    assert rollout_metrics["raw_pred_freq"] == pytest.approx(metrics["raw_pred_freq"])
    assert rollout_metrics["clamped_pred_freq"] == pytest.approx(config.sfp_freq_init)

    policy = StreamingFlowPolicy.__new__(StreamingFlowPolicy)
    nn.Module.__init__(policy)
    policy.config = config
    policy.model = model
    param_groups = policy.get_optim_params()
    vision_param_ids = {id(param) for param in model.rgb_encoder.model.parameters()}
    lower_lr_param_ids = {id(param) for param in param_groups[1]["params"]}
    assert vision_param_ids == lower_lr_param_ids
    assert param_groups[1]["lr"] == pytest.approx(config.optimizer_lr * config.vision_encoder_lr_multiplier)
