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

"""Cross-chunk Streaming Flow with Mamba3 sequence dynamics.

Training uses the existing ControlFlow interpolation and multi-point velocity
targets. Inference preserves causal Euler integration while retaining Mamba
state only for the duration of one generated chunk.
"""

# The recurrence follows the Mamba3 paper's X/Delta/A/B/C notation.
# ruff: noqa: N803, N806

import math
from collections import deque
from copy import deepcopy
from types import SimpleNamespace
from typing import NamedTuple

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)
from lerobot.utils.import_utils import require_package

from ..dispo.modeling_dispo_mamba3 import (
    NOISY_ACTION_STREAM,
    GatedMIMOTrapezoidalSSMMixer,
    run_complex_mimo_trapezoidal_ssm_step_cuda,
)
from ..pretrained import PreTrainedPolicy
from ..utils import populate_queues
from .configuration_streaming_flow import StreamingFlowMambaConfig
from .modeling_streaming_flow_v3 import (
    SinusoidalPosEmb,
    linearly_interpolate_trajectory,
    sample_cfm_inputs_and_targets,
)
from .modeling_streaming_flow_v5 import (
    CLIPPatchTokenImageEncoder,
    CLIPTokenTextConditionEncoder,
    DirectFrequencyPredictor,
    StreamingFlowModel as StreamingFlowV5Model,
    StreamingFlowPolicy as StreamingFlowV5Policy,
)

GENERATED_ACTION_HISTORY = "generated_action_history"
GENERATED_ACTION_HISTORY_IS_PAD = "generated_action_history_is_pad"


class AttentionKV(NamedTuple):
    key: Tensor
    value: Tensor
    valid_mask: Tensor


class ChunkInferenceCache:
    """Ephemeral cache; it is discarded after one action chunk."""

    def __init__(
        self,
        *,
        pooled_context: Tensor,
        frequency_token: Tensor,
        observation_kv: AttentionKV | None,
        history_kv: AttentionKV | None,
        mamba_states: list[dict[str, Tensor]],
    ):
        self.pooled_context = pooled_context
        self.frequency_token = frequency_token
        self.observation_kv = observation_kv
        self.history_kv = history_kv
        self.mamba_states = mamba_states


class EfficientCrossAttention(nn.Module):
    """SDPA cross-attention with reusable projected K/V tensors."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.kv_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def prepare_kv(self, memory: Tensor, valid_mask: Tensor) -> AttentionKV:
        batch_size, memory_len, _ = memory.shape
        key, value = self.kv_proj(memory).chunk(2, dim=-1)
        key = key.view(batch_size, memory_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, memory_len, self.num_heads, self.head_dim).transpose(1, 2)
        return AttentionKV(key, value, valid_mask.bool())

    def forward(self, query: Tensor, cache: AttentionKV) -> Tensor:
        batch_size, query_len, _ = query.shape
        projected_query = self.q_proj(query)
        projected_query = projected_query.view(
            batch_size, query_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        attention_mask = cache.valid_mask[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            projected_query,
            cache.key,
            cache.value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, query_len, -1)
        return self.out_proj(attended)


def _mamba_adapter(config: StreamingFlowMambaConfig) -> SimpleNamespace:
    """Expose the policy-neutral subset expected by the DiSPo Mamba3 mixer."""
    return SimpleNamespace(
        hidden_dim=config.mamba_hidden_dim,
        d_state=config.mamba_d_state,
        dropout=config.mamba_dropout,
        mlp_ratio=config.mamba_mlp_ratio,
        mamba3_mimo_rank=config.mamba_mimo_rank,
        mamba3_omega_min=config.mamba_omega_min,
        mamba3_use_rotary_angle=config.mamba_use_rotary_angle,
        mamba3_use_complex_ssm=config.mamba_use_complex_ssm,
        mamba3_use_output_gate=config.mamba_use_output_gate,
    )


class StreamingMamba3Mixer(GatedMIMOTrapezoidalSSMMixer):
    """Mamba3 mixer with a stateful one-token inference API."""

    def __init__(self, config: StreamingFlowMambaConfig):
        super().__init__(
            _mamba_adapter(config),
            stream_dims={NOISY_ACTION_STREAM: config.mamba_hidden_dim},
        )
        self.use_cuda_step_kernel = config.mamba_use_cuda_kernel
        self.use_cuda_fast_ssm = self.use_cuda_fast_ssm and config.mamba_use_cuda_kernel

    def init_state(self, batch_size: int, device: torch.device) -> dict[str, Tensor]:
        state = {
            "h_real": torch.zeros(batch_size, self.d_state, self.d_model, device=device, dtype=torch.float32),
            "prev_b_real": torch.zeros(
                batch_size, self.d_state, self.rank, device=device, dtype=torch.float32
            ),
            "prev_x": torch.zeros(batch_size, self.d_model, self.rank, device=device, dtype=torch.float32),
        }
        if self.use_complex_ssm:
            state.update(
                {
                    "h_imag": torch.zeros_like(state["h_real"]),
                    "prev_b_imag": torch.zeros_like(state["prev_b_real"]),
                    "angle": torch.zeros(batch_size, self.d_state, device=device, dtype=torch.float32),
                }
            )
        return state

    def _step_parameters(
        self,
        hidden_states: Tensor,
        *,
        delta_rate: Tensor,
        eta: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
        batch_size = hidden_states.shape[0]
        delta_rate = self._expand_rate(
            delta_rate,
            name="delta_rate",
            batch_size=batch_size,
            seqlen=1,
            device=hidden_states.device,
        )
        eta = self._expand_rate(
            eta,
            name="eta",
            batch_size=batch_size,
            seqlen=1,
            device=hidden_states.device,
        )
        gated_streams = self._encode_and_gate_streams(
            hidden_states,
            delta_rate=delta_rate,
            eta=eta,
            stream_context={},
        )
        rank_proj = self.stream_rank_proj.to(device=hidden_states.device, dtype=gated_streams.dtype)
        X = torch.einsum("blds,sr->bldr", gated_streams, rank_proj)
        u = self.u_proj(X.flatten(start_dim=2))
        u = u + self.r_embed(delta_rate.unsqueeze(-1)) + self.eta_embed(eta.unsqueeze(-1))
        Delta = delta_rate.unsqueeze(-1) * F.softplus(self.delta_proj(u).float())
        A = -F.softplus(self.A_proj(u).float())
        B = self.B_proj(u).float().view(batch_size, 1, self.d_state, self.rank)
        C = self.C_proj(u).float().view(batch_size, 1, self.d_state, self.rank)
        lambd = torch.sigmoid(self.lambda_proj(u).float())
        angle_velocity = torch.tanh(self.angle_proj(u).float()) * math.pi if self.use_complex_ssm else None
        z = self.z_proj(u).float() if self.use_output_gate else None
        return (
            X[:, 0].float(),
            Delta[:, 0],
            A[:, 0],
            B[:, 0],
            C[:, 0],
            lambd[:, 0],
            angle_velocity[:, 0] if angle_velocity is not None else None,
            z[:, 0] if z is not None else None,
        )

    def _complex_step_reference(
        self,
        X: Tensor,
        Delta: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        lambd: Tensor,
        angle_velocity: Tensor,
        state: dict[str, Tensor],
    ) -> Tensor:
        theta = Delta * angle_velocity
        alpha = torch.exp((Delta * A).clamp(min=-20.0, max=20.0))
        beta = (1.0 - lambd) * Delta * alpha
        gamma = lambd * Delta
        step_cos = torch.cos(theta).unsqueeze(-1)
        step_sin = torch.sin(theta).unsqueeze(-1)
        rotated_real = step_cos * state["h_real"] - step_sin * state["h_imag"]
        rotated_imag = step_sin * state["h_real"] + step_cos * state["h_imag"]
        angle = state["angle"] + theta
        angle_cos = torch.cos(angle).unsqueeze(-1)
        angle_sin = torch.sin(angle).unsqueeze(-1)
        b_real = B * angle_cos
        b_imag = B * angle_sin
        c_real = C * angle_cos
        c_imag = C * angle_sin
        prev_outer_real = torch.einsum("bnr,bdr->bnd", state["prev_b_real"], state["prev_x"])
        prev_outer_imag = torch.einsum("bnr,bdr->bnd", state["prev_b_imag"], state["prev_x"])
        current_outer_real = torch.einsum("bnr,bdr->bnd", b_real, X)
        current_outer_imag = torch.einsum("bnr,bdr->bnd", b_imag, X)
        h_real = (
            alpha.unsqueeze(-1) * rotated_real
            + beta.unsqueeze(-1) * prev_outer_real
            + gamma.unsqueeze(-1) * current_outer_real
        )
        h_imag = (
            alpha.unsqueeze(-1) * rotated_imag
            + beta.unsqueeze(-1) * prev_outer_imag
            + gamma.unsqueeze(-1) * current_outer_imag
        )
        y = torch.einsum("bnr,bnd->brd", c_real, h_real)
        y = y + torch.einsum("bnr,bnd->brd", c_imag, h_imag)
        state.update(
            {
                "h_real": h_real,
                "h_imag": h_imag,
                "prev_b_real": b_real,
                "prev_b_imag": b_imag,
                "prev_x": X,
                "angle": angle,
            }
        )
        return y.flatten(start_dim=1)

    def _real_step_reference(
        self,
        X: Tensor,
        Delta: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        lambd: Tensor,
        state: dict[str, Tensor],
    ) -> Tensor:
        alpha = torch.exp((Delta * A).clamp(min=-20.0, max=20.0))
        beta = (1.0 - lambd) * Delta * alpha
        gamma = lambd * Delta
        previous_outer = torch.einsum("bnr,bdr->bnd", state["prev_b_real"], state["prev_x"])
        current_outer = torch.einsum("bnr,bdr->bnd", B, X)
        hidden = (
            alpha.unsqueeze(-1) * state["h_real"]
            + beta.unsqueeze(-1) * previous_outer
            + gamma.unsqueeze(-1) * current_outer
        )
        y = torch.einsum("bnr,bnd->brd", C, hidden).flatten(start_dim=1)
        state.update({"h_real": hidden, "prev_b_real": B, "prev_x": X})
        return y

    def step(
        self,
        hidden_states: Tensor,
        *,
        delta_rate: Tensor,
        eta: Tensor,
        state: dict[str, Tensor],
    ) -> Tensor:
        if hidden_states.shape[1] != 1:
            raise ValueError(f"Mamba3 step expects a length-one input. Got {hidden_states.shape}.")
        X, Delta, A, B, C, lambd, angle_velocity, z = self._step_parameters(
            hidden_states, delta_rate=delta_rate, eta=eta
        )
        y = None
        if (
            self.use_complex_ssm
            and self.use_cuda_step_kernel
            and self.use_cuda_fast_ssm
            and not self._cuda_fast_ssm_disabled
            and X.is_cuda
            and self.d_state in {1, 2, 4, 8, 16, 32, 64}
            and self.rank in {1, 4}
        ):
            try:
                y = run_complex_mimo_trapezoidal_ssm_step_cuda(
                    X=X,
                    Delta=Delta,
                    A=A,
                    B=B,
                    C=C,
                    lambd=lambd,
                    angle_velocity=angle_velocity,
                    state=state,
                    z=z,
                    d_model=self.d_model,
                    d_state=self.d_state,
                    rank=self.rank,
                )
            except Exception:
                self._cuda_fast_ssm_disabled = True
        if y is None:
            if self.use_complex_ssm:
                y = self._complex_step_reference(X, Delta, A, B, C, lambd, angle_velocity, state)
            else:
                y = self._real_step_reference(X, Delta, A, B, C, lambd, state)
            if z is not None:
                y = y * z * torch.sigmoid(z)
        return self.output_proj(y.to(dtype=self.output_proj.weight.dtype)).unsqueeze(1)


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1.0 + scale) + shift


class StreamingMamba3Block(nn.Module):
    """Conditioned Mamba3 block shared by training scans and inference steps."""

    def __init__(self, config: StreamingFlowMambaConfig):
        super().__init__()
        hidden_dim = config.mamba_hidden_dim
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mixer = StreamingMamba3Mixer(config)
        self.mlp_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_dim = int(hidden_dim * config.mamba_mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_dim, hidden_dim),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.dropout = nn.Dropout(config.mamba_dropout)
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def _inputs(self, x: Tensor, condition: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        shift_mamba, scale_mamba, gate_mamba, shift_mlp, scale_mlp, gate_mlp = self.modulation(
            condition
        ).chunk(6, dim=-1)
        mamba_input = _modulate(self.norm(x), shift_mamba, scale_mamba)
        mamba_gate = 1.0 + torch.tanh(gate_mamba)
        mlp_gate = 1.0 + torch.tanh(gate_mlp)
        return mamba_input, mamba_gate, shift_mlp, scale_mlp, mlp_gate

    def forward(
        self,
        x: Tensor,
        *,
        condition: Tensor,
        delta_rate: Tensor,
        eta: Tensor,
    ) -> Tensor:
        mamba_input, mamba_gate, shift_mlp, scale_mlp, mlp_gate = self._inputs(x, condition)
        x = x + mamba_gate * self.dropout(
            self.mixer(
                mamba_input,
                delta_rate=delta_rate,
                eta=eta,
                stream_context={},
            )
        )
        mlp_input = _modulate(self.mlp_norm(x), shift_mlp, scale_mlp)
        return x + mlp_gate * self.dropout(self.mlp(mlp_input))

    def step(
        self,
        x: Tensor,
        *,
        condition: Tensor,
        delta_rate: Tensor,
        eta: Tensor,
        state: dict[str, Tensor],
    ) -> Tensor:
        mamba_input, mamba_gate, shift_mlp, scale_mlp, mlp_gate = self._inputs(x, condition)
        x = x + mamba_gate * self.dropout(
            self.mixer.step(mamba_input, delta_rate=delta_rate, eta=eta, state=state)
        )
        mlp_input = _modulate(self.mlp_norm(x), shift_mlp, scale_mlp)
        return x + mlp_gate * self.dropout(self.mlp(mlp_input))


class StreamingFlowMambaVelocityModel(nn.Module):
    """Previous-tail encoder, dual cross-attention stem, and current Mamba3."""

    def __init__(self, config: StreamingFlowMambaConfig, global_cond_dim: int):
        super().__init__()
        self.config = config
        action_dim = config.action_feature.shape[0]
        hidden_dim = config.mamba_hidden_dim
        self.action_in_proj = nn.Linear(action_dim, hidden_dim)
        self.action_out_proj = nn.Linear(hidden_dim, action_dim)
        self.motion_in_proj = nn.Linear(action_dim * 3, hidden_dim)
        self.pooled_context_proj = nn.Sequential(
            nn.LayerNorm(global_cond_dim),
            nn.Linear(global_cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context_token_proj = (
            nn.Identity()
            if config.transformer_hidden_dim == hidden_dim
            else nn.Linear(config.transformer_hidden_dim, hidden_dim)
        )
        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(config.embedding_dim, scale=config.timestep_embedding_scale),
            nn.Linear(config.embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.frequency_encoder = nn.Sequential(
            SinusoidalPosEmb(config.embedding_dim, scale=config.frequency_embedding_scale),
            nn.Linear(config.embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.current_position = nn.Embedding(
            max(config.chunk_size, config.sfp_num_train_points) + 1, hidden_dim
        )
        self.history_position = nn.Embedding(config.previous_tail_len + 1, hidden_dim)
        self.history_bos = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.current_stem = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.condition_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rate_proj = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.rate_proj.weight)
        nn.init.constant_(self.rate_proj.bias, math.log(math.expm1(1.0)))

        self.observation_attention = (
            EfficientCrossAttention(hidden_dim, config.cross_attention_heads, config.mamba_dropout)
            if config.use_observation_cross_attention
            else None
        )
        self.history_attention = (
            EfficientCrossAttention(hidden_dim, config.cross_attention_heads, config.mamba_dropout)
            if config.use_cross_chunk_attention and config.history_encoder != "none"
            else None
        )
        self.observation_gate = nn.Parameter(torch.zeros(()))
        self.history_gate = nn.Parameter(torch.zeros(()))
        self.history_blocks = nn.ModuleList(
            [StreamingMamba3Block(config) for _ in range(config.mamba_history_depth)]
            if config.history_encoder == "mamba3"
            else []
        )
        self.current_blocks = nn.ModuleList(
            [StreamingMamba3Block(config) for _ in range(config.mamba_current_depth)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.granularity_predictor = DirectFrequencyPredictor(global_cond_dim, config)
        nn.init.normal_(self.history_bos, std=0.02)

    def _time_tensor(self, timestep: Tensor, batch_size: int, query_len: int, device: torch.device) -> Tensor:
        timestep = torch.as_tensor(timestep, dtype=torch.float32, device=device)
        if timestep.ndim == 0:
            timestep = timestep.view(1, 1).expand(batch_size, query_len)
        elif timestep.ndim == 1:
            if timestep.numel() == 1:
                timestep = timestep.view(1, 1).expand(batch_size, query_len)
            elif timestep.numel() == batch_size:
                timestep = timestep[:, None].expand(-1, query_len)
            else:
                raise ValueError(f"Invalid timestep shape {tuple(timestep.shape)}.")
        elif timestep.shape != (batch_size, query_len):
            raise ValueError(
                f"Timestep must have shape {(batch_size, query_len)}. Got {tuple(timestep.shape)}."
            )
        return timestep

    @staticmethod
    def _motion_features(actions: Tensor) -> Tensor:
        first = torch.zeros_like(actions)
        first[:, 1:] = actions[:, 1:] - actions[:, :-1]
        second = torch.zeros_like(actions)
        second[:, 2:] = first[:, 2:] - first[:, 1:-1]
        return torch.cat([actions, first, second], dim=-1)

    def encode_history(
        self,
        previous_tail: Tensor,
        history_mask: Tensor,
        *,
        pooled_context: Tensor,
        frequency_token: Tensor,
        dynamics_rate: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size, history_len, _ = previous_tail.shape
        # Episode-start padding and stochastic history dropout can leave holes.
        # Stable-compacting valid actions to the left prevents padded tokens from
        # changing the recurrent state seen by real history tokens.
        original_positions = torch.arange(history_len, device=previous_tail.device)[None]
        sort_key = (~history_mask.bool()).long() * history_len + original_positions
        compact_order = sort_key.argsort(dim=1)
        previous_tail = previous_tail.gather(1, compact_order[:, :, None].expand_as(previous_tail))
        valid_count = history_mask.sum(dim=1)
        history_mask = original_positions < valid_count[:, None]
        positions = torch.arange(1, history_len + 1, device=previous_tail.device)
        hidden = self.motion_in_proj(self._motion_features(previous_tail.float()))
        hidden = hidden + self.history_position(positions)[None]
        condition = self.condition_proj(
            torch.cat([pooled_context, torch.zeros_like(pooled_context), frequency_token], dim=-1)
        )
        condition = condition[:, None].expand(-1, history_len, -1)
        history_eta = torch.linspace(0.0, 1.0, history_len, device=hidden.device)[None].expand(batch_size, -1)
        history_rate = dynamics_rate[:, None].expand(-1, history_len) * history_mask
        if self.config.history_encoder == "mamba3":
            for block in self.history_blocks:
                hidden = block(
                    hidden,
                    condition=condition,
                    delta_rate=history_rate,
                    eta=history_eta,
                )
        bos = self.history_bos.expand(batch_size, -1, -1)
        hidden = torch.cat([bos, hidden], dim=1)
        valid = torch.cat(
            [torch.ones(batch_size, 1, dtype=torch.bool, device=hidden.device), history_mask.bool()],
            dim=1,
        )
        return hidden, valid

    def prepare_cache(
        self,
        *,
        global_cond: Tensor,
        context_tokens: Tensor,
        context_mask: Tensor,
        freq: Tensor,
        previous_tail: Tensor,
        history_mask: Tensor,
        allocate_mamba_state: bool = True,
    ) -> ChunkInferenceCache:
        pooled_context = self.pooled_context_proj(global_cond.float())
        frequency_token = self.frequency_encoder(freq.float())
        if not self.config.lambda_condition_token:
            frequency_token = torch.zeros_like(frequency_token)
        dynamics_rate = (
            freq.float().clamp(self.config.sfp_freq_min, self.config.sfp_freq_max)
            if self.config.lambda_condition_step_size
            else torch.ones_like(freq, dtype=torch.float32)
        )
        observation_kv = None
        if self.observation_attention is not None:
            projected_context = self.context_token_proj(context_tokens)
            observation_memory = torch.cat(
                [projected_context, pooled_context[:, None], frequency_token[:, None]], dim=1
            )
            observation_valid = torch.cat(
                [
                    context_mask.bool(),
                    torch.ones(context_mask.shape[0], 2, dtype=torch.bool, device=context_mask.device),
                ],
                dim=1,
            )
            observation_kv = self.observation_attention.prepare_kv(observation_memory, observation_valid)
        history_kv = None
        if self.history_attention is not None:
            history_tokens, history_valid = self.encode_history(
                previous_tail,
                history_mask,
                pooled_context=pooled_context,
                frequency_token=frequency_token,
                dynamics_rate=dynamics_rate,
            )
            history_kv = self.history_attention.prepare_kv(history_tokens, history_valid)
        states = (
            [
                block.mixer.init_state(global_cond.shape[0], global_cond.device)
                for block in self.current_blocks
            ]
            if allocate_mamba_state
            else []
        )
        return ChunkInferenceCache(
            pooled_context=pooled_context,
            frequency_token=frequency_token,
            observation_kv=observation_kv,
            history_kv=history_kv,
            mamba_states=states,
        )

    def _embed_current(
        self,
        sample: Tensor,
        timestep: Tensor,
        cache: ChunkInferenceCache,
        freq: Tensor,
        *,
        position_offset: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, query_len, _ = sample.shape
        timestep = self._time_tensor(timestep, batch_size, query_len, sample.device)
        time_token = self.time_encoder(timestep.flatten()).view(batch_size, query_len, -1)
        frequency_token = cache.frequency_token[:, None].expand(-1, query_len, -1)
        hidden = self.current_stem(
            torch.cat([self.action_in_proj(sample.float()), time_token, frequency_token], dim=-1)
        )
        positions = torch.arange(
            position_offset, position_offset + query_len, device=sample.device
        ).clamp_max(self.current_position.num_embeddings - 1)
        hidden = hidden + self.current_position(positions)[None]
        pooled = cache.pooled_context[:, None].expand(-1, query_len, -1)
        gate_frequency = (
            frequency_token if self.config.lambda_condition_gate else torch.zeros_like(frequency_token)
        )
        condition = self.condition_proj(torch.cat([pooled, time_token, gate_frequency], dim=-1))
        base_rate = (
            freq.float().clamp(self.config.sfp_freq_min, self.config.sfp_freq_max)
            if self.config.lambda_condition_step_size
            else torch.ones_like(freq, dtype=torch.float32)
        )
        learned_rate = F.softplus(self.rate_proj(condition)).squeeze(-1)
        delta_rate = base_rate[:, None] * learned_rate
        return hidden, condition, delta_rate

    def _cross_attention(self, hidden: Tensor, cache: ChunkInferenceCache) -> Tensor:
        if self.observation_attention is not None and cache.observation_kv is not None:
            hidden = hidden + torch.sigmoid(self.observation_gate) * self.observation_attention(
                hidden, cache.observation_kv
            )
        if self.history_attention is not None and cache.history_kv is not None:
            hidden = hidden + torch.sigmoid(self.history_gate) * self.history_attention(
                hidden, cache.history_kv
            )
        return hidden

    def forward_prepared(
        self,
        sample: Tensor,
        timestep: Tensor,
        *,
        cache: ChunkInferenceCache,
        freq: Tensor,
    ) -> Tensor:
        if sample.ndim == 2:
            sample = sample.unsqueeze(1)
        hidden, condition, delta_rate = self._embed_current(sample, timestep, cache, freq)
        hidden = self._cross_attention(hidden, cache)
        eta = self._time_tensor(timestep, sample.shape[0], sample.shape[1], sample.device)
        for block in self.current_blocks:
            hidden = block(hidden, condition=condition, delta_rate=delta_rate, eta=eta)
        return self.action_out_proj(self.output_norm(hidden))

    def step(
        self,
        sample: Tensor,
        timestep: Tensor,
        *,
        cache: ChunkInferenceCache,
        freq: Tensor,
        position: int,
    ) -> Tensor:
        if sample.ndim == 2:
            sample = sample.unsqueeze(1)
        hidden, condition, delta_rate = self._embed_current(
            sample, timestep, cache, freq, position_offset=position
        )
        hidden = self._cross_attention(hidden, cache)
        eta = self._time_tensor(timestep, sample.shape[0], 1, sample.device)
        for block, state in zip(self.current_blocks, cache.mamba_states, strict=True):
            hidden = block.step(
                hidden,
                condition=condition,
                delta_rate=delta_rate,
                eta=eta,
                state=state,
            )
        return self.action_out_proj(self.output_norm(hidden))

    def forward(
        self,
        sample: Tensor,
        timestep: Tensor,
        global_cond: Tensor,
        context_tokens: Tensor,
        context_mask: Tensor,
        freq: Tensor,
        previous_tail: Tensor,
        history_mask: Tensor,
    ) -> Tensor:
        cache = self.prepare_cache(
            global_cond=global_cond,
            context_tokens=context_tokens,
            context_mask=context_mask,
            freq=freq,
            previous_tail=previous_tail,
            history_mask=history_mask,
            allocate_mamba_state=False,
        )
        return self.forward_prepared(sample, timestep, cache=cache, freq=freq)


class StreamingFlowMambaModel(StreamingFlowV5Model):
    """v5 CLIP conditioning with a Mamba3 velocity expert."""

    def __init__(self, config: StreamingFlowMambaConfig):
        nn.Module.__init__(self)
        self.config = config
        hidden_dim = config.transformer_hidden_dim
        global_cond_dim = 0
        if config.image_features:
            num_images = len(config.image_features)
            if config.use_separate_rgb_encoder_per_camera:
                self.rgb_encoder = nn.ModuleList(
                    [CLIPPatchTokenImageEncoder(config) for _ in range(num_images)]
                )
                global_cond_dim += num_images * self.rgb_encoder[0].feature_dim
            else:
                self.rgb_encoder = CLIPPatchTokenImageEncoder(config)
                global_cond_dim += num_images * self.rgb_encoder.feature_dim
        else:
            self.rgb_encoder = None
        if config.robot_state_feature is not None:
            state_dim = config.robot_state_feature.shape[0]
            global_cond_dim += config.n_obs_steps * state_dim
            self.state_token_proj = nn.Sequential(nn.LayerNorm(state_dim), nn.Linear(state_dim, hidden_dim))
        else:
            self.state_token_proj = None
        self.text_encoder = (
            CLIPTokenTextConditionEncoder(config) if config.sfp_use_clip_text_conditioning else None
        )
        if self.text_encoder is not None:
            global_cond_dim += config.n_obs_steps * self.text_encoder.feature_dim
        self.context_type_embedding = nn.Parameter(torch.zeros(1, 3, hidden_dim))
        nn.init.normal_(self.context_type_embedding, std=0.02)
        self.global_cond_dim = global_cond_dim
        self.velocity_model = StreamingFlowMambaVelocityModel(config, global_cond_dim)

    def _empty_history(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        action_dim = self.config.action_feature.shape[0]
        return (
            torch.zeros(
                batch_size,
                self.config.previous_tail_len,
                action_dim,
                device=device,
                dtype=dtype,
            ),
            torch.zeros(batch_size, self.config.previous_tail_len, device=device, dtype=torch.bool),
        )

    def integrate_actions(
        self,
        batch: dict[str, Tensor],
        init_action: Tensor | None = None,
        previous_tail: Tensor | None = None,
        history_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, float]]:
        raw_cond, normalized_cond, context_tokens, context_mask = self._prepare_token_conditioning(batch)
        raw_freq, freq, clamped_freq = self._predict_frequency(
            raw_cond, clamp=self.config.sfp_clamp_freq_during_eval
        )
        action = self._initial_action(batch, init_action=init_action)
        if previous_tail is None or history_mask is None:
            previous_tail, history_mask = self._empty_history(action.shape[0], action.device, action.dtype)
        cache = self.velocity_model.prepare_cache(
            global_cond=normalized_cond,
            context_tokens=context_tokens,
            context_mask=context_mask,
            freq=freq,
            previous_tail=previous_tail,
            history_mask=history_mask,
            allocate_mamba_state=self.config.inference_mode == "incremental",
        )
        dt = 1.0 / max(self.config.chunk_size - self.config.n_obs_steps, 1)
        action_chunk = []
        prefix_actions: list[Tensor] = []
        prefix_times: list[Tensor] = []
        initial_action = action.detach()
        for step_idx in range(self.config.n_action_steps):
            timestep = torch.full(
                (action.shape[0],), step_idx * dt, device=action.device, dtype=torch.float32
            )
            if self.config.inference_mode == "incremental":
                velocity = self.velocity_model.step(
                    action.float(),
                    timestep,
                    cache=cache,
                    freq=freq,
                    position=step_idx,
                )
            else:
                prefix_actions.append(action.float())
                prefix_times.append(timestep)
                velocity = self.velocity_model.forward_prepared(
                    torch.cat(prefix_actions, dim=1),
                    torch.stack(prefix_times, dim=1),
                    cache=cache,
                    freq=freq,
                )[:, -1:]
            action = action + velocity * dt
            action_chunk.append(action.squeeze(1))
        stacked_chunk = torch.stack(action_chunk, dim=1)
        final_action = action.detach()
        cuda_step_active = (
            self.config.inference_mode == "incremental"
            and action.is_cuda
            and self.config.mamba_d_state in {1, 2, 4, 8, 16, 32, 64}
            and self.config.mamba_mimo_rank in {1, 4}
            and all(
                block.mixer.use_cuda_fast_ssm and not block.mixer._cuda_fast_ssm_disabled
                for block in self.velocity_model.current_blocks
            )
        )
        return (
            stacked_chunk,
            final_action,
            {
                "pred_freq": float(freq.mean().detach().cpu()),
                "raw_pred_freq": float(raw_freq.mean().detach().cpu()),
                "clamped_pred_freq": float(clamped_freq.mean().detach().cpu()),
                "dt": float(dt),
                "history_valid_steps": float(history_mask.sum(dim=1).float().mean().detach().cpu()),
                "inference_mode": self.config.inference_mode,
                "mamba_cuda_step_enabled": float(cuda_step_active),
                "init_action_mean_abs": float(initial_action.abs().mean().detach().cpu()),
                "final_action_mean_abs": float(final_action.abs().mean().detach().cpu()),
                "boundary_action_delta_mean_abs": float(
                    (stacked_chunk[:, 0] - initial_action.squeeze(1)).abs().mean().detach().cpu()
                ),
            },
        )

    def _generated_training_history(
        self, batch: dict[str, Tensor], reference: Tensor
    ) -> tuple[Tensor, Tensor]:
        if GENERATED_ACTION_HISTORY not in batch:
            raise ValueError(
                f"`history_training_mode={self.config.history_training_mode}` requires "
                f"`{GENERATED_ACTION_HISTORY}` in each batch. Populate it from an offline "
                "sequential rollout cache to avoid an extra model rollout in every train step."
            )
        generated = batch[GENERATED_ACTION_HISTORY].detach()
        expected_shape = reference.shape
        if (
            generated.ndim != 3
            or generated.shape[0] != expected_shape[0]
            or generated.shape[2] != expected_shape[2]
        ):
            raise ValueError(
                f"`{GENERATED_ACTION_HISTORY}` must have shape (B, H, A) matching "
                f"B={expected_shape[0]} and A={expected_shape[2]}. Got {tuple(generated.shape)}."
            )
        if generated.shape[1] < self.config.previous_tail_len:
            pad_len = self.config.previous_tail_len - generated.shape[1]
            generated = F.pad(generated, (0, 0, pad_len, 0))
            generated_mask = torch.cat(
                [
                    torch.zeros(generated.shape[0], pad_len, dtype=torch.bool, device=generated.device),
                    torch.ones(
                        generated.shape[0],
                        generated.shape[1] - pad_len,
                        dtype=torch.bool,
                        device=generated.device,
                    ),
                ],
                dim=1,
            )
        else:
            generated = generated[:, -self.config.previous_tail_len :]
            generated_mask = torch.ones(generated.shape[:2], dtype=torch.bool, device=generated.device)
        if GENERATED_ACTION_HISTORY_IS_PAD in batch:
            generated_pad = batch[GENERATED_ACTION_HISTORY_IS_PAD].bool()
            generated_pad = generated_pad[:, -self.config.previous_tail_len :]
            if generated_pad.shape[1] < self.config.previous_tail_len:
                generated_pad = F.pad(
                    generated_pad,
                    (self.config.previous_tail_len - generated_pad.shape[1], 0),
                    value=True,
                )
            generated_mask &= ~generated_pad
        return generated.to(device=reference.device, dtype=reference.dtype), generated_mask.to(
            reference.device
        )

    def _training_history(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        history_len = self.config.previous_tail_len
        previous_tail = batch[ACTION][:, :history_len]
        trajectory = batch[ACTION][:, history_len:]
        if "action_is_pad" in batch:
            history_mask = ~batch["action_is_pad"][:, :history_len].bool()
            trajectory_pad = batch["action_is_pad"][:, history_len:].bool()
        else:
            history_mask = torch.ones(previous_tail.shape[:2], dtype=torch.bool, device=previous_tail.device)
            trajectory_pad = torch.zeros(trajectory.shape[:2], dtype=torch.bool, device=trajectory.device)
        generated_fraction = torch.zeros((), device=previous_tail.device)
        use_generated_history = self.config.history_training_mode == "generated" or (
            self.config.history_training_mode == "scheduled_mix"
            and self.config.generated_history_probability > 0.0
        )
        if use_generated_history:
            generated_tail, generated_mask = self._generated_training_history(batch, previous_tail)
            if self.config.history_training_mode == "generated":
                use_generated = torch.ones(
                    previous_tail.shape[0], dtype=torch.bool, device=previous_tail.device
                )
            else:
                use_generated = (
                    torch.rand(previous_tail.shape[0], device=previous_tail.device)
                    < self.config.generated_history_probability
                )
            previous_tail = torch.where(use_generated[:, None, None], generated_tail, previous_tail)
            history_mask = torch.where(use_generated[:, None], generated_mask, history_mask)
            generated_fraction = use_generated.float().mean()
        if self.training:
            if self.config.history_noise_std > 0.0:
                previous_tail = (
                    previous_tail + torch.randn_like(previous_tail) * self.config.history_noise_std
                )
            if self.config.history_dropout_prob > 0.0:
                keep = (
                    torch.rand(previous_tail.shape[:2], device=previous_tail.device)
                    >= self.config.history_dropout_prob
                )
                history_mask = history_mask & keep
        return previous_tail, history_mask, trajectory, trajectory_pad, generated_fraction

    def compute_loss(
        self,
        batch: dict[str, Tensor],
        reduction: str = "mean",
    ) -> tuple[Tensor, dict[str, float] | None]:
        if ACTION not in batch:
            raise ValueError(f"Missing `{ACTION}` in batch. Available keys: {list(batch)}")
        raw_cond, normalized_cond, context_tokens, context_mask = self._prepare_token_conditioning(batch)
        raw_freq, freq, clamped_freq = self._predict_frequency(
            raw_cond, clamp=self.config.sfp_clamp_freq_during_training
        )
        previous_tail, history_mask, trajectory, trajectory_pad, generated_fraction = self._training_history(
            batch
        )
        num_queries = max(1, int(self.config.sfp_num_train_points))
        time_shape = (trajectory.shape[0], num_queries) if num_queries > 1 else (trajectory.shape[0],)
        time = torch.rand(time_shape, device=trajectory.device, dtype=torch.float32) * 0.999 + 0.001
        if num_queries > 1 and self.config.multi_point_mode == "ordered_sequence":
            time = time.sort(dim=1).values
        xi_t, dxi_dt = linearly_interpolate_trajectory(trajectory, time)
        noised_action, target_velocity = sample_cfm_inputs_and_targets(
            xi_t, dxi_dt, time, k=self.config.sfp_k, sigma0=self.config.sfp_sigma0
        )
        if num_queries > 1 and self.config.multi_point_mode == "independent":
            batch_size, _, action_dim = noised_action.shape
            sample = noised_action.reshape(batch_size * num_queries, 1, action_dim)
            target = target_velocity.reshape(batch_size * num_queries, 1, action_dim)
            query_time = time.reshape(batch_size * num_queries)
            cond = normalized_cond.repeat_interleave(num_queries, dim=0)
            query_freq = freq.repeat_interleave(num_queries, dim=0)
            tokens = context_tokens.repeat_interleave(num_queries, dim=0)
            token_mask = context_mask.repeat_interleave(num_queries, dim=0)
            tail = previous_tail.repeat_interleave(num_queries, dim=0)
            tail_mask = history_mask.repeat_interleave(num_queries, dim=0)
        else:
            sample = noised_action.unsqueeze(1) if noised_action.ndim == 2 else noised_action
            target = target_velocity.unsqueeze(1) if target_velocity.ndim == 2 else target_velocity
            query_time = time
            cond = normalized_cond
            query_freq = freq
            tokens = context_tokens
            token_mask = context_mask
            tail = previous_tail
            tail_mask = history_mask
        pred_velocity = self.velocity_model(
            sample=sample,
            timestep=query_time,
            global_cond=cond,
            context_tokens=tokens,
            context_mask=token_mask,
            freq=query_freq,
            previous_tail=tail,
            history_mask=tail_mask,
        )
        per_query_loss = F.mse_loss(pred_velocity, target, reduction="none").mean(dim=-1)
        per_sample_loss = per_query_loss.reshape(trajectory.shape[0], num_queries).mean(dim=1)
        if self.config.sfp_freq_reg_weight > 0.0:
            per_sample_loss = per_sample_loss + self.config.sfp_freq_reg_weight * (freq - 1.0).pow(2)
        if self.config.do_mask_loss_for_padding:
            valid_mask = (~trajectory_pad).all(dim=1)
            per_sample_loss = per_sample_loss * valid_mask.to(per_sample_loss.dtype)
            mean_loss = per_sample_loss.sum() / valid_mask.sum().clamp_min(1)
        else:
            mean_loss = per_sample_loss.mean()
        metrics = {
            "loss": float(mean_loss.detach()),
            "pred_freq": float(freq.mean().detach()),
            "raw_pred_freq": float(raw_freq.mean().detach()),
            "clamped_pred_freq": float(clamped_freq.mean().detach()),
            "sfp_num_train_points": float(num_queries),
            "history_valid_fraction": float(history_mask.float().mean().detach()),
            "generated_history_fraction": float(generated_fraction.detach()),
        }
        return (per_sample_loss, metrics) if reduction == "none" else (mean_loss, metrics)


class StreamingFlowMambaPolicy(StreamingFlowV5Policy):
    """LeRobot policy wrapper with an executed-action tail buffer."""

    config_class = StreamingFlowMambaConfig
    name = "streaming_flow_mamba"

    def __init__(self, config: StreamingFlowMambaConfig, **kwargs):
        require_package("transformers", extra="multi_task_dit")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.register_buffer("_action_min", torch.empty(0), persistent=True)
        self.register_buffer("_action_max", torch.empty(0), persistent=True)
        self.register_buffer("_ema_step", torch.zeros((), dtype=torch.long), persistent=True)
        self._init_normalization_buffers(kwargs.get("dataset_stats"))
        self.model = StreamingFlowMambaModel(config)
        self.ema_model = deepcopy(self.model) if config.use_ema else None
        if self.ema_model is not None:
            self.ema_model.requires_grad_(False)
        self.reset()

    def reset(self):
        super().reset()
        self._executed_action_history: deque[Tensor] = deque(maxlen=self.config.previous_tail_len)

    def _previous_tail(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        action_dim = self.config.action_feature.shape[0]
        tail = torch.zeros(
            batch_size,
            self.config.previous_tail_len,
            action_dim,
            device=device,
            dtype=dtype,
        )
        mask = torch.zeros(batch_size, self.config.previous_tail_len, device=device, dtype=torch.bool)
        if not self._executed_action_history:
            return tail, mask
        if any(action.shape[0] != batch_size for action in self._executed_action_history):
            # Stale history cannot be paired safely after a rollout batch-size change.
            self._executed_action_history.clear()
            return tail, mask
        history = torch.stack(list(self._executed_action_history), dim=1).to(device=device, dtype=dtype)
        history = history[:, -self.config.previous_tail_len :]
        tail[:, -history.shape[1] :] = history
        mask[:, -history.shape[1] :] = True
        return tail, mask

    @torch.no_grad()
    def _predict_action_chunk_and_state(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, float | str]]:
        del noise
        queued_batch = self._queued_batch()
        # Language tokens and lightweight dataset task IDs are episode-level
        # metadata, not temporal observations, so pass them through directly.
        for key in (OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, "task_index"):
            if key in batch:
                queued_batch[key] = batch[key]
        init_source = "prev_action_state"
        init_action = self._prev_action_state
        if init_action is None:
            init_action, init_source = self._initial_rollout_action(queued_batch)
        if init_action is None:
            raise RuntimeError("StreamingFlowMambaPolicy could not construct an initial rollout action.")
        previous_tail, history_mask = self._previous_tail(
            init_action.shape[0], init_action.device, init_action.dtype
        )
        chunk, final_action, info = self._rollout_model().integrate_actions(
            queued_batch,
            init_action=init_action,
            previous_tail=previous_tail,
            history_mask=history_mask,
        )
        info = dict(info)
        info["init_source"] = init_source
        info["history_source"] = "executed_policy_actions"
        return chunk, final_action, info

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        batch = self._prepare_image_batch(batch)
        self._queues = populate_queues(self._queues, batch)
        self._left_pad_observation_queues()
        chunk, final_action, info = self._predict_action_chunk_and_state(batch, noise=noise)
        self._prev_action_state = final_action
        # Direct chunk callers conventionally execute the returned chunk. Keep
        # continuity for that API while select_action records actions one-by-one.
        for action in chunk.transpose(0, 1):
            self._executed_action_history.append(action.detach())
        self._record_rollout_info(info)
        return chunk

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        if ACTION in batch:
            batch = dict(batch)
            batch.pop(ACTION)
        batch = self._prepare_image_batch(batch)
        self._queues = populate_queues(self._queues, batch)
        self._left_pad_observation_queues()
        if len(self._queues[ACTION]) == 0:
            actions, final_action, info = self._predict_action_chunk_and_state(batch, noise=noise)
            self._prev_action_state = final_action
            self._record_rollout_info(info)
            self._queues[ACTION].extend(actions.transpose(0, 1))
        action = self._queues[ACTION].popleft()
        self._executed_action_history.append(action.detach())
        return action


# Keep the conventional module-level policy name used by LeRobot policy modules.
StreamingFlowPolicy = StreamingFlowMambaPolicy
