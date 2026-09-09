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
from lerobot.policies.dispo import modeling_dispo_mamba3
from lerobot.policies.streaming_flow.configuration_streaming_flow import StreamingFlowMambaConfig
from lerobot.policies.streaming_flow.modeling_streaming_flow_mamba import (
    GENERATED_ACTION_HISTORY,
    StreamingFlowMambaModel,
    StreamingFlowMambaVelocityModel,
    StreamingMamba3Mixer,
)
from lerobot.utils.constants import ACTION


def _config(**overrides) -> StreamingFlowMambaConfig:
    values = {
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        "chunk_size": 8,
        "n_action_steps": 4,
        "sfp_num_train_points": 4,
        "transformer_hidden_dim": 32,
        "transformer_num_layers": 1,
        "transformer_num_heads": 4,
        "transformer_ffn_dim": 64,
        "transformer_visual_tokens_per_frame": 4,
        "embedding_dim": 16,
        "mamba_hidden_dim": 32,
        "mamba_history_depth": 1,
        "mamba_current_depth": 2,
        "mamba_d_state": 4,
        "mamba_mimo_rank": 4,
        "mamba_dropout": 0.0,
        "previous_tail_len": 3,
        "cross_attention_heads": 4,
        "history_dropout_prob": 0.0,
        "history_noise_std": 0.0,
        "use_ema": False,
    }
    values.update(overrides)
    return StreamingFlowMambaConfig(**values)


def test_mamba_config_extends_action_indices_with_previous_tail() -> None:
    config = _config(previous_tail_len=3)

    assert config.action_delta_indices == list(range(-3, 7))


@pytest.mark.parametrize("use_complex", [False, True])
def test_mamba_full_scan_matches_incremental_reference(use_complex: bool) -> None:
    torch.manual_seed(1)
    config = _config(
        mamba_use_complex_ssm=use_complex,
        mamba_use_rotary_angle=use_complex,
        mamba_use_cuda_kernel=False,
    )
    mixer = StreamingMamba3Mixer(config).eval()
    hidden = torch.randn(2, 5, config.mamba_hidden_dim)
    rate = torch.rand(2, 5) + 0.2
    eta = torch.rand(2, 5)

    with torch.no_grad():
        full = mixer(hidden, delta_rate=rate, eta=eta, stream_context={})
        state = mixer.init_state(hidden.shape[0], hidden.device)
        incremental = torch.cat(
            [
                mixer.step(
                    hidden[:, index : index + 1],
                    delta_rate=rate[:, index],
                    eta=eta[:, index],
                    state=state,
                )
                for index in range(hidden.shape[1])
            ],
            dim=1,
        )

    torch.testing.assert_close(incremental, full, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available() or modeling_dispo_mamba3.triton is None,
    reason="CUDA and Triton are required",
)
def test_mamba_cuda_step_kernel_matches_full_scan() -> None:
    torch.manual_seed(2)
    config = _config(mamba_use_cuda_kernel=True)
    mixer = StreamingMamba3Mixer(config).cuda().eval()
    hidden = torch.randn(2, 5, config.mamba_hidden_dim, device="cuda")
    rate = torch.rand(2, 5, device="cuda") + 0.2
    eta = torch.rand(2, 5, device="cuda")

    with torch.no_grad():
        full = mixer(hidden, delta_rate=rate, eta=eta, stream_context={})
        state = mixer.init_state(hidden.shape[0], hidden.device)
        incremental = torch.cat(
            [
                mixer.step(
                    hidden[:, index : index + 1],
                    delta_rate=rate[:, index],
                    eta=eta[:, index],
                    state=state,
                )
                for index in range(hidden.shape[1])
            ],
            dim=1,
        )
        torch.cuda.synchronize()

    assert not mixer._cuda_fast_ssm_disabled
    torch.testing.assert_close(incremental, full, atol=2e-4, rtol=2e-4)


def test_velocity_model_full_sequence_matches_incremental_path() -> None:
    torch.manual_seed(3)
    config = _config(mamba_use_cuda_kernel=False)
    model = StreamingFlowMambaVelocityModel(config, global_cond_dim=12).eval()
    batch_size = 2
    sequence_len = 4
    sample = torch.randn(batch_size, sequence_len, 7)
    timestep = torch.linspace(0.0, 0.8, sequence_len)[None].expand(batch_size, -1)
    global_cond = torch.randn(batch_size, 12)
    context_tokens = torch.randn(batch_size, 6, config.transformer_hidden_dim)
    context_mask = torch.ones(batch_size, 6, dtype=torch.bool)
    freq = torch.full((batch_size,), 1.2)
    previous_tail = torch.randn(batch_size, config.previous_tail_len, 7)
    history_mask = torch.ones(batch_size, config.previous_tail_len, dtype=torch.bool)

    with torch.no_grad():
        full_cache = model.prepare_cache(
            global_cond=global_cond,
            context_tokens=context_tokens,
            context_mask=context_mask,
            freq=freq,
            previous_tail=previous_tail,
            history_mask=history_mask,
        )
        full = model.forward_prepared(sample, timestep, cache=full_cache, freq=freq)
        step_cache = model.prepare_cache(
            global_cond=global_cond,
            context_tokens=context_tokens,
            context_mask=context_mask,
            freq=freq,
            previous_tail=previous_tail,
            history_mask=history_mask,
        )
        incremental = torch.cat(
            [
                model.step(
                    sample[:, index : index + 1],
                    timestep[:, index],
                    cache=step_cache,
                    freq=freq,
                    position=index,
                )
                for index in range(sequence_len)
            ],
            dim=1,
        )

    torch.testing.assert_close(incremental, full, atol=3e-5, rtol=3e-5)


def test_cross_chunk_attention_responds_to_previous_tail() -> None:
    torch.manual_seed(4)
    config = _config(mamba_use_cuda_kernel=False)
    model = StreamingFlowMambaVelocityModel(config, global_cond_dim=12).eval()
    common = {
        "sample": torch.randn(1, 4, 7),
        "timestep": torch.linspace(0.1, 0.9, 4)[None],
        "global_cond": torch.randn(1, 12),
        "context_tokens": torch.randn(1, 5, config.transformer_hidden_dim),
        "context_mask": torch.ones(1, 5, dtype=torch.bool),
        "freq": torch.ones(1),
        "history_mask": torch.ones(1, config.previous_tail_len, dtype=torch.bool),
    }

    first = model(previous_tail=torch.zeros(1, config.previous_tail_len, 7), **common)
    second = model(previous_tail=torch.ones(1, config.previous_tail_len, 7), **common)

    assert not torch.allclose(first, second)


def test_generated_history_can_replace_demonstration_tail() -> None:
    config = _config(
        history_training_mode="generated",
        history_noise_std=0.0,
        history_dropout_prob=0.0,
    )
    model = StreamingFlowMambaModel.__new__(StreamingFlowMambaModel)
    nn.Module.__init__(model)
    model.config = config
    demonstration = torch.zeros(2, len(config.action_delta_indices), 7)
    generated = torch.ones(2, config.previous_tail_len, 7)

    previous_tail, history_mask, trajectory, _, generated_fraction = model._training_history(
        {
            ACTION: demonstration,
            GENERATED_ACTION_HISTORY: generated,
        }
    )

    torch.testing.assert_close(previous_tail, generated)
    assert history_mask.all()
    assert trajectory.shape[1] == demonstration.shape[1] - config.previous_tail_len
    assert generated_fraction.item() == 1.0
