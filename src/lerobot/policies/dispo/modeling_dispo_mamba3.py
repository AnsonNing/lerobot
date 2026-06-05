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

import math
import os

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from .configuration_dispo import DiSPoConfig

try:
    import triton
    import triton.language as tl  # noqa: N812
except ImportError:  # pragma: no cover - optional CUDA fast path
    triton = None
    tl = None


GLOBAL_VISUAL_STREAM = "global_visual"
LOCAL_OR_WRIST_VISUAL_STREAM = "local_or_wrist_visual"
PROPRIO_STREAM = "proprio"
NOISY_ACTION_STREAM = "noisy_action"
GRANULARITY_CONDITION_STREAM = "granularity_condition"
FUSED_OBSERVATION_STREAM = "fused_observation"
TASK_TEXT_STREAM = "task_text"


if triton is not None:

    @triton.jit
    def _complex_mimo_trapezoidal_ssm_kernel(
        X,
        Delta,
        A,
        B_param,
        C_param,
        lambd,
        angle_velocity,
        z,
        out,
        D_MODEL: tl.constexpr,
        SEQLEN: tl.constexpr,
        D_STATE: tl.constexpr,
        RANK: tl.constexpr,
        HAS_Z: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ) -> None:
        batch_idx = tl.program_id(0)
        d_block_idx = tl.program_id(1)

        d_offsets = d_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D_MODEL
        n_offsets = tl.arange(0, D_STATE)

        H_real = tl.zeros((D_STATE, BLOCK_D), tl.float32)
        H_imag = tl.zeros((D_STATE, BLOCK_D), tl.float32)
        prev_B_real_0 = tl.zeros((D_STATE,), tl.float32)
        prev_B_imag_0 = tl.zeros((D_STATE,), tl.float32)
        prev_X_0 = tl.zeros((BLOCK_D,), tl.float32)
        prev_B_real_1 = tl.zeros((D_STATE,), tl.float32)
        prev_B_imag_1 = tl.zeros((D_STATE,), tl.float32)
        prev_X_1 = tl.zeros((BLOCK_D,), tl.float32)
        prev_B_real_2 = tl.zeros((D_STATE,), tl.float32)
        prev_B_imag_2 = tl.zeros((D_STATE,), tl.float32)
        prev_X_2 = tl.zeros((BLOCK_D,), tl.float32)
        prev_B_real_3 = tl.zeros((D_STATE,), tl.float32)
        prev_B_imag_3 = tl.zeros((D_STATE,), tl.float32)
        prev_X_3 = tl.zeros((BLOCK_D,), tl.float32)
        angle_state = tl.zeros((D_STATE,), tl.float32)

        for step_idx in tl.static_range(0, SEQLEN):
            base_bln = (batch_idx * SEQLEN + step_idx) * D_STATE + n_offsets
            Delta_t = tl.load(Delta + base_bln).to(tl.float32)
            A_t = tl.load(A + base_bln).to(tl.float32)
            lambda_t = tl.load(lambd + base_bln).to(tl.float32)
            theta_t = Delta_t * tl.load(angle_velocity + base_bln).to(tl.float32)

            alpha_arg = tl.minimum(tl.maximum(Delta_t * A_t, -20.0), 20.0)
            alpha = tl.exp(alpha_arg)
            beta = (1.0 - lambda_t) * Delta_t * alpha
            gamma = lambda_t * Delta_t

            step_cos = tl.cos(theta_t)
            step_sin = tl.sin(theta_t)
            rotated_H_real = step_cos[:, None] * H_real - step_sin[:, None] * H_imag
            rotated_H_imag = step_sin[:, None] * H_real + step_cos[:, None] * H_imag

            angle_state += theta_t
            angle_cos = tl.cos(angle_state)
            angle_sin = tl.sin(angle_state)

            base_bl_nr = (batch_idx * SEQLEN + step_idx) * D_STATE * RANK
            B_t_0 = tl.load(B_param + base_bl_nr + n_offsets * RANK).to(tl.float32)
            C_t_0 = tl.load(C_param + base_bl_nr + n_offsets * RANK).to(tl.float32)
            B_real_0 = B_t_0 * angle_cos
            B_imag_0 = B_t_0 * angle_sin
            C_real_0 = C_t_0 * angle_cos
            C_imag_0 = C_t_0 * angle_sin

            base_bl_dr = (batch_idx * SEQLEN + step_idx) * D_MODEL * RANK
            X_t_0 = tl.load(
                X + base_bl_dr + d_offsets * RANK,
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)

            prev_outer_real = prev_B_real_0[:, None] * prev_X_0[None, :]
            prev_outer_imag = prev_B_imag_0[:, None] * prev_X_0[None, :]
            curr_outer_real = B_real_0[:, None] * X_t_0[None, :]
            curr_outer_imag = B_imag_0[:, None] * X_t_0[None, :]

            if RANK == 4:
                B_t_1 = tl.load(B_param + base_bl_nr + n_offsets * RANK + 1).to(tl.float32)
                B_t_2 = tl.load(B_param + base_bl_nr + n_offsets * RANK + 2).to(tl.float32)
                B_t_3 = tl.load(B_param + base_bl_nr + n_offsets * RANK + 3).to(tl.float32)
                C_t_1 = tl.load(C_param + base_bl_nr + n_offsets * RANK + 1).to(tl.float32)
                C_t_2 = tl.load(C_param + base_bl_nr + n_offsets * RANK + 2).to(tl.float32)
                C_t_3 = tl.load(C_param + base_bl_nr + n_offsets * RANK + 3).to(tl.float32)
                B_real_1 = B_t_1 * angle_cos
                B_real_2 = B_t_2 * angle_cos
                B_real_3 = B_t_3 * angle_cos
                B_imag_1 = B_t_1 * angle_sin
                B_imag_2 = B_t_2 * angle_sin
                B_imag_3 = B_t_3 * angle_sin
                C_real_1 = C_t_1 * angle_cos
                C_real_2 = C_t_2 * angle_cos
                C_real_3 = C_t_3 * angle_cos
                C_imag_1 = C_t_1 * angle_sin
                C_imag_2 = C_t_2 * angle_sin
                C_imag_3 = C_t_3 * angle_sin

                X_t_1 = tl.load(X + base_bl_dr + d_offsets * RANK + 1, mask=d_mask, other=0.0).to(
                    tl.float32
                )
                X_t_2 = tl.load(X + base_bl_dr + d_offsets * RANK + 2, mask=d_mask, other=0.0).to(
                    tl.float32
                )
                X_t_3 = tl.load(X + base_bl_dr + d_offsets * RANK + 3, mask=d_mask, other=0.0).to(
                    tl.float32
                )

                prev_outer_real += (
                    prev_B_real_1[:, None] * prev_X_1[None, :]
                    + prev_B_real_2[:, None] * prev_X_2[None, :]
                    + prev_B_real_3[:, None] * prev_X_3[None, :]
                )
                prev_outer_imag += (
                    prev_B_imag_1[:, None] * prev_X_1[None, :]
                    + prev_B_imag_2[:, None] * prev_X_2[None, :]
                    + prev_B_imag_3[:, None] * prev_X_3[None, :]
                )
                curr_outer_real += (
                    B_real_1[:, None] * X_t_1[None, :]
                    + B_real_2[:, None] * X_t_2[None, :]
                    + B_real_3[:, None] * X_t_3[None, :]
                )
                curr_outer_imag += (
                    B_imag_1[:, None] * X_t_1[None, :]
                    + B_imag_2[:, None] * X_t_2[None, :]
                    + B_imag_3[:, None] * X_t_3[None, :]
                )

            H_real = (
                alpha[:, None] * rotated_H_real
                + beta[:, None] * prev_outer_real
                + gamma[:, None] * curr_outer_real
            )
            H_imag = (
                alpha[:, None] * rotated_H_imag
                + beta[:, None] * prev_outer_imag
                + gamma[:, None] * curr_outer_imag
            )

            out_base = (batch_idx * SEQLEN + step_idx) * RANK * D_MODEL
            Y_0 = tl.sum(C_real_0[:, None] * H_real + C_imag_0[:, None] * H_imag, axis=0)
            if HAS_Z:
                z_0 = tl.load(z + out_base + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
                Y_0 = Y_0 * z_0 * tl.sigmoid(z_0)
            tl.store(out + out_base + d_offsets, Y_0, mask=d_mask)

            prev_B_real_0 = B_real_0
            prev_B_imag_0 = B_imag_0
            prev_X_0 = X_t_0

            if RANK == 4:
                Y_1 = tl.sum(C_real_1[:, None] * H_real + C_imag_1[:, None] * H_imag, axis=0)
                Y_2 = tl.sum(C_real_2[:, None] * H_real + C_imag_2[:, None] * H_imag, axis=0)
                Y_3 = tl.sum(C_real_3[:, None] * H_real + C_imag_3[:, None] * H_imag, axis=0)
                if HAS_Z:
                    z_1 = tl.load(z + out_base + D_MODEL + d_offsets, mask=d_mask, other=0.0).to(
                        tl.float32
                    )
                    z_2 = tl.load(
                        z + out_base + 2 * D_MODEL + d_offsets, mask=d_mask, other=0.0
                    ).to(tl.float32)
                    z_3 = tl.load(
                        z + out_base + 3 * D_MODEL + d_offsets, mask=d_mask, other=0.0
                    ).to(tl.float32)
                    Y_1 = Y_1 * z_1 * tl.sigmoid(z_1)
                    Y_2 = Y_2 * z_2 * tl.sigmoid(z_2)
                    Y_3 = Y_3 * z_3 * tl.sigmoid(z_3)
                tl.store(out + out_base + D_MODEL + d_offsets, Y_1, mask=d_mask)
                tl.store(out + out_base + 2 * D_MODEL + d_offsets, Y_2, mask=d_mask)
                tl.store(out + out_base + 3 * D_MODEL + d_offsets, Y_3, mask=d_mask)

                prev_B_real_1 = B_real_1
                prev_B_imag_1 = B_imag_1
                prev_X_1 = X_t_1
                prev_B_real_2 = B_real_2
                prev_B_imag_2 = B_imag_2
                prev_X_2 = X_t_2
                prev_B_real_3 = B_real_3
                prev_B_imag_3 = B_imag_3
                prev_X_3 = X_t_3

    @triton.jit
    def _complex_mimo_trapezoidal_ssm_fwd_cache_kernel(
        X,
        Delta,
        A,
        B_param,
        C_param,
        lambd,
        angle_velocity,
        z,
        out,
        h_real_cache,
        h_imag_cache,
        D_MODEL: tl.constexpr,
        SEQLEN: tl.constexpr,
        D_STATE: tl.constexpr,
        RANK: tl.constexpr,
        HAS_Z: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ) -> None:
        batch_idx = tl.program_id(0)
        d_block_idx = tl.program_id(1)

        d_offsets = d_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D_MODEL
        n_offsets = tl.arange(0, D_STATE)

        H_real = tl.zeros((D_STATE, BLOCK_D), tl.float32)
        H_imag = tl.zeros((D_STATE, BLOCK_D), tl.float32)
        prev_B_real_0 = tl.zeros((D_STATE,), tl.float32)
        prev_B_imag_0 = tl.zeros((D_STATE,), tl.float32)
        prev_X_0 = tl.zeros((BLOCK_D,), tl.float32)
        prev_B_real_1 = tl.zeros((D_STATE,), tl.float32)
        prev_B_imag_1 = tl.zeros((D_STATE,), tl.float32)
        prev_X_1 = tl.zeros((BLOCK_D,), tl.float32)
        prev_B_real_2 = tl.zeros((D_STATE,), tl.float32)
        prev_B_imag_2 = tl.zeros((D_STATE,), tl.float32)
        prev_X_2 = tl.zeros((BLOCK_D,), tl.float32)
        prev_B_real_3 = tl.zeros((D_STATE,), tl.float32)
        prev_B_imag_3 = tl.zeros((D_STATE,), tl.float32)
        prev_X_3 = tl.zeros((BLOCK_D,), tl.float32)
        angle_state = tl.zeros((D_STATE,), tl.float32)

        for step_idx in tl.static_range(0, SEQLEN):
            base_bln = (batch_idx * SEQLEN + step_idx) * D_STATE + n_offsets
            Delta_t = tl.load(Delta + base_bln).to(tl.float32)
            A_t = tl.load(A + base_bln).to(tl.float32)
            lambda_t = tl.load(lambd + base_bln).to(tl.float32)
            theta_t = Delta_t * tl.load(angle_velocity + base_bln).to(tl.float32)

            alpha_arg = tl.minimum(tl.maximum(Delta_t * A_t, -20.0), 20.0)
            alpha = tl.exp(alpha_arg)
            beta = (1.0 - lambda_t) * Delta_t * alpha
            gamma = lambda_t * Delta_t

            step_cos = tl.cos(theta_t)
            step_sin = tl.sin(theta_t)
            rotated_H_real = step_cos[:, None] * H_real - step_sin[:, None] * H_imag
            rotated_H_imag = step_sin[:, None] * H_real + step_cos[:, None] * H_imag

            angle_state += theta_t
            angle_cos = tl.cos(angle_state)
            angle_sin = tl.sin(angle_state)

            base_bl_nr = (batch_idx * SEQLEN + step_idx) * D_STATE * RANK
            B_t_0 = tl.load(B_param + base_bl_nr + n_offsets * RANK).to(tl.float32)
            C_t_0 = tl.load(C_param + base_bl_nr + n_offsets * RANK).to(tl.float32)
            B_real_0 = B_t_0 * angle_cos
            B_imag_0 = B_t_0 * angle_sin
            C_real_0 = C_t_0 * angle_cos
            C_imag_0 = C_t_0 * angle_sin

            base_bl_dr = (batch_idx * SEQLEN + step_idx) * D_MODEL * RANK
            X_t_0 = tl.load(X + base_bl_dr + d_offsets * RANK, mask=d_mask, other=0.0).to(
                tl.float32
            )

            prev_outer_real = prev_B_real_0[:, None] * prev_X_0[None, :]
            prev_outer_imag = prev_B_imag_0[:, None] * prev_X_0[None, :]
            curr_outer_real = B_real_0[:, None] * X_t_0[None, :]
            curr_outer_imag = B_imag_0[:, None] * X_t_0[None, :]

            if RANK == 4:
                B_t_1 = tl.load(B_param + base_bl_nr + n_offsets * RANK + 1).to(tl.float32)
                B_t_2 = tl.load(B_param + base_bl_nr + n_offsets * RANK + 2).to(tl.float32)
                B_t_3 = tl.load(B_param + base_bl_nr + n_offsets * RANK + 3).to(tl.float32)
                C_t_1 = tl.load(C_param + base_bl_nr + n_offsets * RANK + 1).to(tl.float32)
                C_t_2 = tl.load(C_param + base_bl_nr + n_offsets * RANK + 2).to(tl.float32)
                C_t_3 = tl.load(C_param + base_bl_nr + n_offsets * RANK + 3).to(tl.float32)
                B_real_1 = B_t_1 * angle_cos
                B_real_2 = B_t_2 * angle_cos
                B_real_3 = B_t_3 * angle_cos
                B_imag_1 = B_t_1 * angle_sin
                B_imag_2 = B_t_2 * angle_sin
                B_imag_3 = B_t_3 * angle_sin
                C_real_1 = C_t_1 * angle_cos
                C_real_2 = C_t_2 * angle_cos
                C_real_3 = C_t_3 * angle_cos
                C_imag_1 = C_t_1 * angle_sin
                C_imag_2 = C_t_2 * angle_sin
                C_imag_3 = C_t_3 * angle_sin

                X_t_1 = tl.load(X + base_bl_dr + d_offsets * RANK + 1, mask=d_mask, other=0.0).to(
                    tl.float32
                )
                X_t_2 = tl.load(X + base_bl_dr + d_offsets * RANK + 2, mask=d_mask, other=0.0).to(
                    tl.float32
                )
                X_t_3 = tl.load(X + base_bl_dr + d_offsets * RANK + 3, mask=d_mask, other=0.0).to(
                    tl.float32
                )

                prev_outer_real += (
                    prev_B_real_1[:, None] * prev_X_1[None, :]
                    + prev_B_real_2[:, None] * prev_X_2[None, :]
                    + prev_B_real_3[:, None] * prev_X_3[None, :]
                )
                prev_outer_imag += (
                    prev_B_imag_1[:, None] * prev_X_1[None, :]
                    + prev_B_imag_2[:, None] * prev_X_2[None, :]
                    + prev_B_imag_3[:, None] * prev_X_3[None, :]
                )
                curr_outer_real += (
                    B_real_1[:, None] * X_t_1[None, :]
                    + B_real_2[:, None] * X_t_2[None, :]
                    + B_real_3[:, None] * X_t_3[None, :]
                )
                curr_outer_imag += (
                    B_imag_1[:, None] * X_t_1[None, :]
                    + B_imag_2[:, None] * X_t_2[None, :]
                    + B_imag_3[:, None] * X_t_3[None, :]
                )

            H_real = (
                alpha[:, None] * rotated_H_real
                + beta[:, None] * prev_outer_real
                + gamma[:, None] * curr_outer_real
            )
            H_imag = (
                alpha[:, None] * rotated_H_imag
                + beta[:, None] * prev_outer_imag
                + gamma[:, None] * curr_outer_imag
            )

            cache_base = (batch_idx * SEQLEN + step_idx) * D_STATE * D_MODEL
            cache_ptrs = cache_base + n_offsets[:, None] * D_MODEL + d_offsets[None, :]
            tl.store(h_real_cache + cache_ptrs, H_real, mask=d_mask[None, :])
            tl.store(h_imag_cache + cache_ptrs, H_imag, mask=d_mask[None, :])

            out_base = (batch_idx * SEQLEN + step_idx) * RANK * D_MODEL
            Y_0 = tl.sum(C_real_0[:, None] * H_real + C_imag_0[:, None] * H_imag, axis=0)
            if HAS_Z:
                z_0 = tl.load(z + out_base + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
                Y_0 = Y_0 * z_0 * tl.sigmoid(z_0)
            tl.store(out + out_base + d_offsets, Y_0, mask=d_mask)

            prev_B_real_0 = B_real_0
            prev_B_imag_0 = B_imag_0
            prev_X_0 = X_t_0

            if RANK == 4:
                Y_1 = tl.sum(C_real_1[:, None] * H_real + C_imag_1[:, None] * H_imag, axis=0)
                Y_2 = tl.sum(C_real_2[:, None] * H_real + C_imag_2[:, None] * H_imag, axis=0)
                Y_3 = tl.sum(C_real_3[:, None] * H_real + C_imag_3[:, None] * H_imag, axis=0)
                if HAS_Z:
                    z_1 = tl.load(z + out_base + D_MODEL + d_offsets, mask=d_mask, other=0.0).to(
                        tl.float32
                    )
                    z_2 = tl.load(
                        z + out_base + 2 * D_MODEL + d_offsets, mask=d_mask, other=0.0
                    ).to(tl.float32)
                    z_3 = tl.load(
                        z + out_base + 3 * D_MODEL + d_offsets, mask=d_mask, other=0.0
                    ).to(tl.float32)
                    Y_1 = Y_1 * z_1 * tl.sigmoid(z_1)
                    Y_2 = Y_2 * z_2 * tl.sigmoid(z_2)
                    Y_3 = Y_3 * z_3 * tl.sigmoid(z_3)
                tl.store(out + out_base + D_MODEL + d_offsets, Y_1, mask=d_mask)
                tl.store(out + out_base + 2 * D_MODEL + d_offsets, Y_2, mask=d_mask)
                tl.store(out + out_base + 3 * D_MODEL + d_offsets, Y_3, mask=d_mask)

                prev_B_real_1 = B_real_1
                prev_B_imag_1 = B_imag_1
                prev_X_1 = X_t_1
                prev_B_real_2 = B_real_2
                prev_B_imag_2 = B_imag_2
                prev_X_2 = X_t_2
                prev_B_real_3 = B_real_3
                prev_B_imag_3 = B_imag_3
                prev_X_3 = X_t_3

    @triton.jit
    def _complex_mimo_trapezoidal_ssm_bwd_kernel(
        grad_y,
        X,
        Delta,
        A,
        B_param,
        C_param,
        lambd,
        angle_velocity,
        z,
        h_real_cache,
        h_imag_cache,
        dX,
        dDelta,
        dA,
        dB,
        dC,
        dlambd,
        dangle_velocity,
        dZ,
        D_MODEL: tl.constexpr,
        SEQLEN: tl.constexpr,
        D_STATE: tl.constexpr,
        RANK: tl.constexpr,
        HAS_Z: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ) -> None:
        batch_idx = tl.program_id(0)
        state_idx = tl.program_id(1)

        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D_MODEL

        dH_real_carry = tl.zeros((BLOCK_D,), tl.float32)
        dH_imag_carry = tl.zeros((BLOCK_D,), tl.float32)
        dX_next_0 = tl.zeros((BLOCK_D,), tl.float32)
        dB_real_next_0 = tl.full((), 0.0, tl.float32)
        dB_imag_next_0 = tl.full((), 0.0, tl.float32)
        dX_next_1 = tl.zeros((BLOCK_D,), tl.float32)
        dB_real_next_1 = tl.full((), 0.0, tl.float32)
        dB_imag_next_1 = tl.full((), 0.0, tl.float32)
        dX_next_2 = tl.zeros((BLOCK_D,), tl.float32)
        dB_real_next_2 = tl.full((), 0.0, tl.float32)
        dB_imag_next_2 = tl.full((), 0.0, tl.float32)
        dX_next_3 = tl.zeros((BLOCK_D,), tl.float32)
        dB_real_next_3 = tl.full((), 0.0, tl.float32)
        dB_imag_next_3 = tl.full((), 0.0, tl.float32)
        d_angle_state_carry = tl.full((), 0.0, tl.float32)

        for step_idx in tl.static_range(SEQLEN - 1, -1, -1):
            base_bln = (batch_idx * SEQLEN + step_idx) * D_STATE + state_idx
            Delta_t = tl.load(Delta + base_bln).to(tl.float32)
            A_t = tl.load(A + base_bln).to(tl.float32)
            lambda_t = tl.load(lambd + base_bln).to(tl.float32)
            angle_velocity_t = tl.load(angle_velocity + base_bln).to(tl.float32)
            theta_t = Delta_t * angle_velocity_t

            alpha_arg = tl.minimum(tl.maximum(Delta_t * A_t, -20.0), 20.0)
            alpha = tl.exp(alpha_arg)
            beta = (1.0 - lambda_t) * Delta_t * alpha
            gamma = lambda_t * Delta_t

            angle_state = tl.full((), 0.0, tl.float32)
            for angle_idx in tl.static_range(0, SEQLEN):
                if angle_idx <= step_idx:
                    angle_base = (batch_idx * SEQLEN + angle_idx) * D_STATE + state_idx
                    angle_state += tl.load(Delta + angle_base).to(tl.float32) * tl.load(
                        angle_velocity + angle_base
                    ).to(tl.float32)

            angle_cos = tl.cos(angle_state)
            angle_sin = tl.sin(angle_state)
            prev_angle_state = angle_state - theta_t
            prev_angle_cos = tl.cos(prev_angle_state)
            prev_angle_sin = tl.sin(prev_angle_state)
            step_cos = tl.cos(theta_t)
            step_sin = tl.sin(theta_t)

            cache_base = (batch_idx * SEQLEN + step_idx) * D_STATE * D_MODEL + state_idx * D_MODEL
            H_real = tl.load(h_real_cache + cache_base + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
            H_imag = tl.load(h_imag_cache + cache_base + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
            if step_idx > 0:
                prev_cache_base = (
                    (batch_idx * SEQLEN + step_idx - 1) * D_STATE * D_MODEL + state_idx * D_MODEL
                )
                H_prev_real = tl.load(
                    h_real_cache + prev_cache_base + d_offsets, mask=d_mask, other=0.0
                ).to(tl.float32)
                H_prev_imag = tl.load(
                    h_imag_cache + prev_cache_base + d_offsets, mask=d_mask, other=0.0
                ).to(tl.float32)
            else:
                H_prev_real = tl.zeros((BLOCK_D,), tl.float32)
                H_prev_imag = tl.zeros((BLOCK_D,), tl.float32)

            rotated_H_real = step_cos * H_prev_real - step_sin * H_prev_imag
            rotated_H_imag = step_sin * H_prev_real + step_cos * H_prev_imag

            base_bl_nr = (batch_idx * SEQLEN + step_idx) * D_STATE * RANK + state_idx * RANK
            B_t_0 = tl.load(B_param + base_bl_nr).to(tl.float32)
            C_t_0 = tl.load(C_param + base_bl_nr).to(tl.float32)
            B_real_0 = B_t_0 * angle_cos
            B_imag_0 = B_t_0 * angle_sin
            C_real_0 = C_t_0 * angle_cos
            C_imag_0 = C_t_0 * angle_sin

            base_bl_dr = (batch_idx * SEQLEN + step_idx) * D_MODEL * RANK
            X_t_0 = tl.load(X + base_bl_dr + d_offsets * RANK, mask=d_mask, other=0.0).to(
                tl.float32
            )

            prev_B_real_0 = tl.full((), 0.0, tl.float32)
            prev_B_imag_0 = tl.full((), 0.0, tl.float32)
            prev_X_0 = tl.zeros((BLOCK_D,), tl.float32)
            if step_idx > 0:
                prev_base_bl_nr = (batch_idx * SEQLEN + step_idx - 1) * D_STATE * RANK + state_idx * RANK
                prev_base_bl_dr = (batch_idx * SEQLEN + step_idx - 1) * D_MODEL * RANK
                prev_B_t_0 = tl.load(B_param + prev_base_bl_nr).to(tl.float32)
                prev_B_real_0 = prev_B_t_0 * prev_angle_cos
                prev_B_imag_0 = prev_B_t_0 * prev_angle_sin
                prev_X_0 = tl.load(
                    X + prev_base_bl_dr + d_offsets * RANK, mask=d_mask, other=0.0
                ).to(tl.float32)

            prev_outer_real = prev_B_real_0 * prev_X_0
            prev_outer_imag = prev_B_imag_0 * prev_X_0
            curr_outer_real = B_real_0 * X_t_0
            curr_outer_imag = B_imag_0 * X_t_0

            if RANK == 4:
                B_t_1 = tl.load(B_param + base_bl_nr + 1).to(tl.float32)
                B_t_2 = tl.load(B_param + base_bl_nr + 2).to(tl.float32)
                B_t_3 = tl.load(B_param + base_bl_nr + 3).to(tl.float32)
                C_t_1 = tl.load(C_param + base_bl_nr + 1).to(tl.float32)
                C_t_2 = tl.load(C_param + base_bl_nr + 2).to(tl.float32)
                C_t_3 = tl.load(C_param + base_bl_nr + 3).to(tl.float32)
                B_real_1 = B_t_1 * angle_cos
                B_real_2 = B_t_2 * angle_cos
                B_real_3 = B_t_3 * angle_cos
                B_imag_1 = B_t_1 * angle_sin
                B_imag_2 = B_t_2 * angle_sin
                B_imag_3 = B_t_3 * angle_sin
                C_real_1 = C_t_1 * angle_cos
                C_real_2 = C_t_2 * angle_cos
                C_real_3 = C_t_3 * angle_cos
                C_imag_1 = C_t_1 * angle_sin
                C_imag_2 = C_t_2 * angle_sin
                C_imag_3 = C_t_3 * angle_sin

                X_t_1 = tl.load(X + base_bl_dr + d_offsets * RANK + 1, mask=d_mask, other=0.0).to(
                    tl.float32
                )
                X_t_2 = tl.load(X + base_bl_dr + d_offsets * RANK + 2, mask=d_mask, other=0.0).to(
                    tl.float32
                )
                X_t_3 = tl.load(X + base_bl_dr + d_offsets * RANK + 3, mask=d_mask, other=0.0).to(
                    tl.float32
                )

                prev_B_real_1 = tl.full((), 0.0, tl.float32)
                prev_B_imag_1 = tl.full((), 0.0, tl.float32)
                prev_B_real_2 = tl.full((), 0.0, tl.float32)
                prev_B_imag_2 = tl.full((), 0.0, tl.float32)
                prev_B_real_3 = tl.full((), 0.0, tl.float32)
                prev_B_imag_3 = tl.full((), 0.0, tl.float32)
                prev_X_1 = tl.zeros((BLOCK_D,), tl.float32)
                prev_X_2 = tl.zeros((BLOCK_D,), tl.float32)
                prev_X_3 = tl.zeros((BLOCK_D,), tl.float32)
                if step_idx > 0:
                    prev_base_bl_nr = (
                        (batch_idx * SEQLEN + step_idx - 1) * D_STATE * RANK + state_idx * RANK
                    )
                    prev_base_bl_dr = (batch_idx * SEQLEN + step_idx - 1) * D_MODEL * RANK
                    prev_B_t_1 = tl.load(B_param + prev_base_bl_nr + 1).to(tl.float32)
                    prev_B_t_2 = tl.load(B_param + prev_base_bl_nr + 2).to(tl.float32)
                    prev_B_t_3 = tl.load(B_param + prev_base_bl_nr + 3).to(tl.float32)
                    prev_B_real_1 = prev_B_t_1 * prev_angle_cos
                    prev_B_real_2 = prev_B_t_2 * prev_angle_cos
                    prev_B_real_3 = prev_B_t_3 * prev_angle_cos
                    prev_B_imag_1 = prev_B_t_1 * prev_angle_sin
                    prev_B_imag_2 = prev_B_t_2 * prev_angle_sin
                    prev_B_imag_3 = prev_B_t_3 * prev_angle_sin
                    prev_X_1 = tl.load(
                        X + prev_base_bl_dr + d_offsets * RANK + 1, mask=d_mask, other=0.0
                    ).to(tl.float32)
                    prev_X_2 = tl.load(
                        X + prev_base_bl_dr + d_offsets * RANK + 2, mask=d_mask, other=0.0
                    ).to(tl.float32)
                    prev_X_3 = tl.load(
                        X + prev_base_bl_dr + d_offsets * RANK + 3, mask=d_mask, other=0.0
                    ).to(tl.float32)

                prev_outer_real += (
                    prev_B_real_1 * prev_X_1 + prev_B_real_2 * prev_X_2 + prev_B_real_3 * prev_X_3
                )
                prev_outer_imag += (
                    prev_B_imag_1 * prev_X_1 + prev_B_imag_2 * prev_X_2 + prev_B_imag_3 * prev_X_3
                )
                curr_outer_real += B_real_1 * X_t_1 + B_real_2 * X_t_2 + B_real_3 * X_t_3
                curr_outer_imag += B_imag_1 * X_t_1 + B_imag_2 * X_t_2 + B_imag_3 * X_t_3

            grad_base = (batch_idx * SEQLEN + step_idx) * RANK * D_MODEL
            grad_out_0 = tl.load(grad_y + grad_base + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
            grad_Y_0 = grad_out_0
            if HAS_Z:
                z_0 = tl.load(z + grad_base + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
                z_sigmoid_0 = tl.sigmoid(z_0)
                Y_state_0 = C_real_0 * H_real + C_imag_0 * H_imag
                dz_gate_0 = z_sigmoid_0 * (1.0 + z_0 * (1.0 - z_sigmoid_0))
                tl.atomic_add(
                    dZ + grad_base + d_offsets,
                    grad_out_0 * Y_state_0 * dz_gate_0,
                    sem="relaxed",
                    mask=d_mask,
                )
                grad_Y_0 = grad_out_0 * z_0 * z_sigmoid_0
            dC_real_0 = tl.sum(grad_Y_0 * H_real, axis=0)
            dC_imag_0 = tl.sum(grad_Y_0 * H_imag, axis=0)
            dH_real = dH_real_carry + grad_Y_0 * C_real_0
            dH_imag = dH_imag_carry + grad_Y_0 * C_imag_0

            if RANK == 4:
                grad_out_1 = tl.load(
                    grad_y + grad_base + D_MODEL + d_offsets, mask=d_mask, other=0.0
                ).to(tl.float32)
                grad_out_2 = tl.load(
                    grad_y + grad_base + 2 * D_MODEL + d_offsets, mask=d_mask, other=0.0
                ).to(tl.float32)
                grad_out_3 = tl.load(
                    grad_y + grad_base + 3 * D_MODEL + d_offsets, mask=d_mask, other=0.0
                ).to(tl.float32)
                grad_Y_1 = grad_out_1
                grad_Y_2 = grad_out_2
                grad_Y_3 = grad_out_3
                if HAS_Z:
                    z_1 = tl.load(
                        z + grad_base + D_MODEL + d_offsets, mask=d_mask, other=0.0
                    ).to(tl.float32)
                    z_2 = tl.load(
                        z + grad_base + 2 * D_MODEL + d_offsets, mask=d_mask, other=0.0
                    ).to(tl.float32)
                    z_3 = tl.load(
                        z + grad_base + 3 * D_MODEL + d_offsets, mask=d_mask, other=0.0
                    ).to(tl.float32)
                    z_sigmoid_1 = tl.sigmoid(z_1)
                    z_sigmoid_2 = tl.sigmoid(z_2)
                    z_sigmoid_3 = tl.sigmoid(z_3)
                    Y_state_1 = C_real_1 * H_real + C_imag_1 * H_imag
                    Y_state_2 = C_real_2 * H_real + C_imag_2 * H_imag
                    Y_state_3 = C_real_3 * H_real + C_imag_3 * H_imag
                    dz_gate_1 = z_sigmoid_1 * (1.0 + z_1 * (1.0 - z_sigmoid_1))
                    dz_gate_2 = z_sigmoid_2 * (1.0 + z_2 * (1.0 - z_sigmoid_2))
                    dz_gate_3 = z_sigmoid_3 * (1.0 + z_3 * (1.0 - z_sigmoid_3))
                    tl.atomic_add(
                        dZ + grad_base + D_MODEL + d_offsets,
                        grad_out_1 * Y_state_1 * dz_gate_1,
                        sem="relaxed",
                        mask=d_mask,
                    )
                    tl.atomic_add(
                        dZ + grad_base + 2 * D_MODEL + d_offsets,
                        grad_out_2 * Y_state_2 * dz_gate_2,
                        sem="relaxed",
                        mask=d_mask,
                    )
                    tl.atomic_add(
                        dZ + grad_base + 3 * D_MODEL + d_offsets,
                        grad_out_3 * Y_state_3 * dz_gate_3,
                        sem="relaxed",
                        mask=d_mask,
                    )
                    grad_Y_1 = grad_out_1 * z_1 * z_sigmoid_1
                    grad_Y_2 = grad_out_2 * z_2 * z_sigmoid_2
                    grad_Y_3 = grad_out_3 * z_3 * z_sigmoid_3
                dC_real_1 = tl.sum(grad_Y_1 * H_real, axis=0)
                dC_real_2 = tl.sum(grad_Y_2 * H_real, axis=0)
                dC_real_3 = tl.sum(grad_Y_3 * H_real, axis=0)
                dC_imag_1 = tl.sum(grad_Y_1 * H_imag, axis=0)
                dC_imag_2 = tl.sum(grad_Y_2 * H_imag, axis=0)
                dC_imag_3 = tl.sum(grad_Y_3 * H_imag, axis=0)
                dH_real += grad_Y_1 * C_real_1 + grad_Y_2 * C_real_2 + grad_Y_3 * C_real_3
                dH_imag += grad_Y_1 * C_imag_1 + grad_Y_2 * C_imag_2 + grad_Y_3 * C_imag_3

            dalpha = tl.sum(dH_real * rotated_H_real + dH_imag * rotated_H_imag, axis=0)
            dbeta = tl.sum(dH_real * prev_outer_real + dH_imag * prev_outer_imag, axis=0)
            dgamma = tl.sum(dH_real * curr_outer_real + dH_imag * curr_outer_imag, axis=0)
            d_rotated_H_real = alpha * dH_real
            d_rotated_H_imag = alpha * dH_imag
            d_prev_outer_real = beta * dH_real
            d_prev_outer_imag = beta * dH_imag
            d_curr_outer_real = gamma * dH_real
            d_curr_outer_imag = gamma * dH_imag

            dB_real_0 = dB_real_next_0 + tl.sum(d_curr_outer_real * X_t_0, axis=0)
            dB_imag_0 = dB_imag_next_0 + tl.sum(d_curr_outer_imag * X_t_0, axis=0)
            dX_0 = dX_next_0 + d_curr_outer_real * B_real_0 + d_curr_outer_imag * B_imag_0
            tl.atomic_add(dX + base_bl_dr + d_offsets * RANK, dX_0, sem="relaxed", mask=d_mask)

            if step_idx > 0:
                dB_real_next_0 = tl.sum(d_prev_outer_real * prev_X_0, axis=0)
                dB_imag_next_0 = tl.sum(d_prev_outer_imag * prev_X_0, axis=0)
                dX_next_0 = d_prev_outer_real * prev_B_real_0 + d_prev_outer_imag * prev_B_imag_0
            else:
                dB_real_next_0 = tl.full((), 0.0, tl.float32)
                dB_imag_next_0 = tl.full((), 0.0, tl.float32)
                dX_next_0 = tl.zeros((BLOCK_D,), tl.float32)

            if RANK == 4:
                dB_real_1 = dB_real_next_1 + tl.sum(d_curr_outer_real * X_t_1, axis=0)
                dB_real_2 = dB_real_next_2 + tl.sum(d_curr_outer_real * X_t_2, axis=0)
                dB_real_3 = dB_real_next_3 + tl.sum(d_curr_outer_real * X_t_3, axis=0)
                dB_imag_1 = dB_imag_next_1 + tl.sum(d_curr_outer_imag * X_t_1, axis=0)
                dB_imag_2 = dB_imag_next_2 + tl.sum(d_curr_outer_imag * X_t_2, axis=0)
                dB_imag_3 = dB_imag_next_3 + tl.sum(d_curr_outer_imag * X_t_3, axis=0)
                dX_1 = dX_next_1 + d_curr_outer_real * B_real_1 + d_curr_outer_imag * B_imag_1
                dX_2 = dX_next_2 + d_curr_outer_real * B_real_2 + d_curr_outer_imag * B_imag_2
                dX_3 = dX_next_3 + d_curr_outer_real * B_real_3 + d_curr_outer_imag * B_imag_3
                tl.atomic_add(dX + base_bl_dr + d_offsets * RANK + 1, dX_1, sem="relaxed", mask=d_mask)
                tl.atomic_add(dX + base_bl_dr + d_offsets * RANK + 2, dX_2, sem="relaxed", mask=d_mask)
                tl.atomic_add(dX + base_bl_dr + d_offsets * RANK + 3, dX_3, sem="relaxed", mask=d_mask)

                if step_idx > 0:
                    dB_real_next_1 = tl.sum(d_prev_outer_real * prev_X_1, axis=0)
                    dB_real_next_2 = tl.sum(d_prev_outer_real * prev_X_2, axis=0)
                    dB_real_next_3 = tl.sum(d_prev_outer_real * prev_X_3, axis=0)
                    dB_imag_next_1 = tl.sum(d_prev_outer_imag * prev_X_1, axis=0)
                    dB_imag_next_2 = tl.sum(d_prev_outer_imag * prev_X_2, axis=0)
                    dB_imag_next_3 = tl.sum(d_prev_outer_imag * prev_X_3, axis=0)
                    dX_next_1 = d_prev_outer_real * prev_B_real_1 + d_prev_outer_imag * prev_B_imag_1
                    dX_next_2 = d_prev_outer_real * prev_B_real_2 + d_prev_outer_imag * prev_B_imag_2
                    dX_next_3 = d_prev_outer_real * prev_B_real_3 + d_prev_outer_imag * prev_B_imag_3
                else:
                    dB_real_next_1 = tl.full((), 0.0, tl.float32)
                    dB_real_next_2 = tl.full((), 0.0, tl.float32)
                    dB_real_next_3 = tl.full((), 0.0, tl.float32)
                    dB_imag_next_1 = tl.full((), 0.0, tl.float32)
                    dB_imag_next_2 = tl.full((), 0.0, tl.float32)
                    dB_imag_next_3 = tl.full((), 0.0, tl.float32)
                    dX_next_1 = tl.zeros((BLOCK_D,), tl.float32)
                    dX_next_2 = tl.zeros((BLOCK_D,), tl.float32)
                    dX_next_3 = tl.zeros((BLOCK_D,), tl.float32)

            dB_0 = dB_real_0 * angle_cos + dB_imag_0 * angle_sin
            dC_0 = dC_real_0 * angle_cos + dC_imag_0 * angle_sin
            d_angle_from_B = dB_real_0 * (-B_t_0 * angle_sin) + dB_imag_0 * (B_t_0 * angle_cos)
            d_angle_from_C = dC_real_0 * (-C_t_0 * angle_sin) + dC_imag_0 * (C_t_0 * angle_cos)
            tl.store(dB + base_bl_nr, dB_0)
            tl.store(dC + base_bl_nr, dC_0)

            if RANK == 4:
                dB_1 = dB_real_1 * angle_cos + dB_imag_1 * angle_sin
                dB_2 = dB_real_2 * angle_cos + dB_imag_2 * angle_sin
                dB_3 = dB_real_3 * angle_cos + dB_imag_3 * angle_sin
                dC_1 = dC_real_1 * angle_cos + dC_imag_1 * angle_sin
                dC_2 = dC_real_2 * angle_cos + dC_imag_2 * angle_sin
                dC_3 = dC_real_3 * angle_cos + dC_imag_3 * angle_sin
                d_angle_from_B += (
                    dB_real_1 * (-B_t_1 * angle_sin)
                    + dB_imag_1 * (B_t_1 * angle_cos)
                    + dB_real_2 * (-B_t_2 * angle_sin)
                    + dB_imag_2 * (B_t_2 * angle_cos)
                    + dB_real_3 * (-B_t_3 * angle_sin)
                    + dB_imag_3 * (B_t_3 * angle_cos)
                )
                d_angle_from_C += (
                    dC_real_1 * (-C_t_1 * angle_sin)
                    + dC_imag_1 * (C_t_1 * angle_cos)
                    + dC_real_2 * (-C_t_2 * angle_sin)
                    + dC_imag_2 * (C_t_2 * angle_cos)
                    + dC_real_3 * (-C_t_3 * angle_sin)
                    + dC_imag_3 * (C_t_3 * angle_cos)
                )
                tl.store(dB + base_bl_nr + 1, dB_1)
                tl.store(dB + base_bl_nr + 2, dB_2)
                tl.store(dB + base_bl_nr + 3, dB_3)
                tl.store(dC + base_bl_nr + 1, dC_1)
                tl.store(dC + base_bl_nr + 2, dC_2)
                tl.store(dC + base_bl_nr + 3, dC_3)

            d_angle_state = d_angle_from_B + d_angle_from_C + d_angle_state_carry
            dH_prev_real = d_rotated_H_real * step_cos + d_rotated_H_imag * step_sin
            dH_prev_imag = -d_rotated_H_real * step_sin + d_rotated_H_imag * step_cos
            dtheta_from_rotation = tl.sum(
                d_rotated_H_real * (-step_sin * H_prev_real - step_cos * H_prev_imag)
                + d_rotated_H_imag * (step_cos * H_prev_real - step_sin * H_prev_imag),
                axis=0,
            )

            dtheta = dtheta_from_rotation + d_angle_state
            dDelta_t = dtheta * angle_velocity_t
            dangle_velocity_t = dtheta * Delta_t

            dlambda_t = dbeta * (-Delta_t * alpha) + dgamma * Delta_t
            dDelta_t += dbeta * (1.0 - lambda_t) * alpha + dgamma * lambda_t
            dalpha += dbeta * (1.0 - lambda_t) * Delta_t

            alpha_unclamped = (Delta_t * A_t >= -20.0) & (Delta_t * A_t <= 20.0)
            d_alpha_arg = dalpha * alpha * alpha_unclamped
            dDelta_t += d_alpha_arg * A_t
            dA_t = d_alpha_arg * Delta_t

            tl.store(dDelta + base_bln, dDelta_t)
            tl.store(dA + base_bln, dA_t)
            tl.store(dlambd + base_bln, dlambda_t)
            tl.store(dangle_velocity + base_bln, dangle_velocity_t)

            dH_real_carry = dH_prev_real
            dH_imag_carry = dH_prev_imag
            d_angle_state_carry = d_angle_state


def _run_complex_mimo_trapezoidal_ssm_cuda(
    *,
    X: Tensor,
    Delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    lambd: Tensor,
    angle_velocity: Tensor,
    z: Tensor | None = None,
    d_model: int,
    d_state: int,
    rank: int,
) -> Tensor:
    if triton is None:
        raise RuntimeError("Triton is not available.")

    batch_size, seqlen, _, _ = X.shape
    X = X.contiguous()
    Delta = Delta.contiguous()
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    lambd = lambd.contiguous()
    angle_velocity = angle_velocity.contiguous()
    z_arg = z.contiguous() if z is not None else X
    y = torch.empty(
        batch_size,
        seqlen,
        rank,
        d_model,
        device=X.device,
        dtype=torch.float32,
    )
    block_d = 16
    grid = (batch_size, triton.cdiv(d_model, block_d))
    _complex_mimo_trapezoidal_ssm_kernel[grid](
        X,
        Delta,
        A,
        B,
        C,
        lambd,
        angle_velocity,
        z_arg,
        y,
        D_MODEL=d_model,
        SEQLEN=seqlen,
        D_STATE=d_state,
        RANK=rank,
        HAS_Z=z is not None,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return y.flatten(start_dim=2)


def _run_complex_mimo_trapezoidal_ssm_cuda_with_cache(
    *,
    X: Tensor,
    Delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    lambd: Tensor,
    angle_velocity: Tensor,
    z: Tensor | None = None,
    d_model: int,
    d_state: int,
    rank: int,
) -> tuple[Tensor, Tensor, Tensor]:
    if triton is None:
        raise RuntimeError("Triton is not available.")

    batch_size, seqlen, _, _ = X.shape
    X = X.contiguous()
    Delta = Delta.contiguous()
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    lambd = lambd.contiguous()
    angle_velocity = angle_velocity.contiguous()
    z_arg = z.contiguous() if z is not None else X
    y = torch.empty(
        batch_size,
        seqlen,
        rank,
        d_model,
        device=X.device,
        dtype=torch.float32,
    )
    h_real_cache = torch.empty(
        batch_size,
        seqlen,
        d_state,
        d_model,
        device=X.device,
        dtype=torch.float32,
    )
    h_imag_cache = torch.empty_like(h_real_cache)
    block_d = 16
    grid = (batch_size, triton.cdiv(d_model, block_d))
    _complex_mimo_trapezoidal_ssm_fwd_cache_kernel[grid](
        X,
        Delta,
        A,
        B,
        C,
        lambd,
        angle_velocity,
        z_arg,
        y,
        h_real_cache,
        h_imag_cache,
        D_MODEL=d_model,
        SEQLEN=seqlen,
        D_STATE=d_state,
        RANK=rank,
        HAS_Z=z is not None,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return y.flatten(start_dim=2), h_real_cache, h_imag_cache


def _run_complex_mimo_trapezoidal_ssm_backward_cuda(
    *,
    grad_y: Tensor,
    X: Tensor,
    Delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    lambd: Tensor,
    angle_velocity: Tensor,
    z: Tensor | None,
    h_real_cache: Tensor,
    h_imag_cache: Tensor,
    d_model: int,
    d_state: int,
    rank: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None]:
    if triton is None:
        raise RuntimeError("Triton is not available.")

    block_d = 1 << (d_model - 1).bit_length()
    if block_d > 1024:
        raise RuntimeError(f"Triton backward only supports hidden_dim <= 1024. Got {d_model}.")

    grad_y = grad_y.contiguous()
    X = X.contiguous()
    Delta = Delta.contiguous()
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    lambd = lambd.contiguous()
    angle_velocity = angle_velocity.contiguous()
    z_arg = z.contiguous() if z is not None else grad_y
    h_real_cache = h_real_cache.contiguous()
    h_imag_cache = h_imag_cache.contiguous()

    dX = torch.zeros_like(X)
    dDelta = torch.empty_like(Delta)
    dA = torch.empty_like(A)
    dB = torch.empty_like(B)
    dC = torch.empty_like(C)
    dlambd = torch.empty_like(lambd)
    dangle_velocity = torch.empty_like(angle_velocity)
    dZ = torch.zeros_like(z_arg) if z is not None else torch.empty_like(grad_y)

    batch_size = X.shape[0]
    seqlen = X.shape[1]
    grid = (batch_size, d_state)
    _complex_mimo_trapezoidal_ssm_bwd_kernel[grid](
        grad_y,
        X,
        Delta,
        A,
        B,
        C,
        lambd,
        angle_velocity,
        z_arg,
        h_real_cache,
        h_imag_cache,
        dX,
        dDelta,
        dA,
        dB,
        dC,
        dlambd,
        dangle_velocity,
        dZ,
        D_MODEL=d_model,
        SEQLEN=seqlen,
        D_STATE=d_state,
        RANK=rank,
        HAS_Z=z is not None,
        BLOCK_D=block_d,
        num_warps=8,
    )
    return dX, dDelta, dA, dB, dC, dlambd, dangle_velocity, dZ if z is not None else None


def _complex_mimo_trapezoidal_ssm_forward(
    *,
    X: Tensor,
    Delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    lambd: Tensor,
    angle_velocity: Tensor,
) -> Tensor:
    batch_size, seqlen, d_model, rank = X.shape
    d_state = Delta.shape[-1]
    H_real = torch.zeros(batch_size, d_state, d_model, device=X.device, dtype=torch.float32)
    H_imag = torch.zeros_like(H_real)
    prev_B_real = torch.zeros(batch_size, d_state, rank, device=X.device, dtype=torch.float32)
    prev_B_imag = torch.zeros_like(prev_B_real)
    prev_X = torch.zeros(batch_size, d_model, rank, device=X.device, dtype=torch.float32)
    angle_state = torch.zeros(batch_size, d_state, device=X.device, dtype=torch.float32)

    outputs = []
    for idx in range(seqlen):
        Delta_t = Delta[:, idx]
        A_t = A[:, idx]
        B_t = B[:, idx]
        C_t = C[:, idx]
        lambda_t = lambd[:, idx]
        X_t = X[:, idx]
        theta_t = Delta_t * angle_velocity[:, idx]

        alpha = torch.exp((Delta_t * A_t).clamp(min=-20.0, max=20.0))
        beta = (1.0 - lambda_t) * Delta_t * alpha
        gamma = lambda_t * Delta_t

        step_cos = torch.cos(theta_t).unsqueeze(-1)
        step_sin = torch.sin(theta_t).unsqueeze(-1)
        rotated_H_real = step_cos * H_real - step_sin * H_imag
        rotated_H_imag = step_sin * H_real + step_cos * H_imag

        angle_state = angle_state + theta_t
        angle_cos = torch.cos(angle_state).unsqueeze(-1)
        angle_sin = torch.sin(angle_state).unsqueeze(-1)
        B_real = B_t * angle_cos
        B_imag = B_t * angle_sin
        C_real = C_t * angle_cos
        C_imag = C_t * angle_sin

        prev_outer_real = torch.einsum("bnr,bdr->bnd", prev_B_real, prev_X)
        prev_outer_imag = torch.einsum("bnr,bdr->bnd", prev_B_imag, prev_X)
        curr_outer_real = torch.einsum("bnr,bdr->bnd", B_real, X_t)
        curr_outer_imag = torch.einsum("bnr,bdr->bnd", B_imag, X_t)

        H_real = (
            alpha.unsqueeze(-1) * rotated_H_real
            + beta.unsqueeze(-1) * prev_outer_real
            + gamma.unsqueeze(-1) * curr_outer_real
        )
        H_imag = (
            alpha.unsqueeze(-1) * rotated_H_imag
            + beta.unsqueeze(-1) * prev_outer_imag
            + gamma.unsqueeze(-1) * curr_outer_imag
        )

        Y_real = torch.einsum("bnr,bnd->brd", C_real, H_real)
        Y_imag = torch.einsum("bnr,bnd->brd", C_imag, H_imag)
        outputs.append((Y_real + Y_imag).flatten(start_dim=1))

        prev_B_real = B_real
        prev_B_imag = B_imag
        prev_X = X_t

    return torch.stack(outputs, dim=1)


def _complex_mimo_trapezoidal_ssm_backward(
    *,
    grad_y: Tensor,
    X: Tensor,
    Delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    lambd: Tensor,
    angle_velocity: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    batch_size, seqlen, d_model, rank = X.shape
    d_state = Delta.shape[-1]

    h_real_steps = []
    h_imag_steps = []
    h_prev_real_steps = []
    h_prev_imag_steps = []
    rotated_h_real_steps = []
    rotated_h_imag_steps = []
    prev_outer_real_steps = []
    prev_outer_imag_steps = []
    curr_outer_real_steps = []
    curr_outer_imag_steps = []
    b_real_steps = []
    b_imag_steps = []
    c_real_steps = []
    c_imag_steps = []
    angle_state_steps = []
    theta_steps = []
    alpha_steps = []
    beta_steps = []
    gamma_steps = []

    H_real = torch.zeros(batch_size, d_state, d_model, device=X.device, dtype=torch.float32)
    H_imag = torch.zeros_like(H_real)
    prev_B_real = torch.zeros(batch_size, d_state, rank, device=X.device, dtype=torch.float32)
    prev_B_imag = torch.zeros_like(prev_B_real)
    prev_X = torch.zeros(batch_size, d_model, rank, device=X.device, dtype=torch.float32)
    angle_state = torch.zeros(batch_size, d_state, device=X.device, dtype=torch.float32)

    for idx in range(seqlen):
        Delta_t = Delta[:, idx]
        A_t = A[:, idx]
        B_t = B[:, idx]
        C_t = C[:, idx]
        lambda_t = lambd[:, idx]
        X_t = X[:, idx]
        theta_t = Delta_t * angle_velocity[:, idx]

        alpha = torch.exp((Delta_t * A_t).clamp(min=-20.0, max=20.0))
        beta = (1.0 - lambda_t) * Delta_t * alpha
        gamma = lambda_t * Delta_t

        step_cos = torch.cos(theta_t).unsqueeze(-1)
        step_sin = torch.sin(theta_t).unsqueeze(-1)
        H_prev_real = H_real
        H_prev_imag = H_imag
        rotated_H_real = step_cos * H_prev_real - step_sin * H_prev_imag
        rotated_H_imag = step_sin * H_prev_real + step_cos * H_prev_imag

        angle_state = angle_state + theta_t
        angle_cos = torch.cos(angle_state).unsqueeze(-1)
        angle_sin = torch.sin(angle_state).unsqueeze(-1)
        B_real = B_t * angle_cos
        B_imag = B_t * angle_sin
        C_real = C_t * angle_cos
        C_imag = C_t * angle_sin

        prev_outer_real = torch.einsum("bnr,bdr->bnd", prev_B_real, prev_X)
        prev_outer_imag = torch.einsum("bnr,bdr->bnd", prev_B_imag, prev_X)
        curr_outer_real = torch.einsum("bnr,bdr->bnd", B_real, X_t)
        curr_outer_imag = torch.einsum("bnr,bdr->bnd", B_imag, X_t)

        H_real = (
            alpha.unsqueeze(-1) * rotated_H_real
            + beta.unsqueeze(-1) * prev_outer_real
            + gamma.unsqueeze(-1) * curr_outer_real
        )
        H_imag = (
            alpha.unsqueeze(-1) * rotated_H_imag
            + beta.unsqueeze(-1) * prev_outer_imag
            + gamma.unsqueeze(-1) * curr_outer_imag
        )

        h_real_steps.append(H_real)
        h_imag_steps.append(H_imag)
        h_prev_real_steps.append(H_prev_real)
        h_prev_imag_steps.append(H_prev_imag)
        rotated_h_real_steps.append(rotated_H_real)
        rotated_h_imag_steps.append(rotated_H_imag)
        prev_outer_real_steps.append(prev_outer_real)
        prev_outer_imag_steps.append(prev_outer_imag)
        curr_outer_real_steps.append(curr_outer_real)
        curr_outer_imag_steps.append(curr_outer_imag)
        b_real_steps.append(B_real)
        b_imag_steps.append(B_imag)
        c_real_steps.append(C_real)
        c_imag_steps.append(C_imag)
        angle_state_steps.append(angle_state)
        theta_steps.append(theta_t)
        alpha_steps.append(alpha)
        beta_steps.append(beta)
        gamma_steps.append(gamma)

        prev_B_real = B_real
        prev_B_imag = B_imag
        prev_X = X_t

    dX = torch.zeros_like(X)
    dDelta = torch.zeros_like(Delta)
    dA = torch.zeros_like(A)
    dB = torch.zeros_like(B)
    dC = torch.zeros_like(C)
    dlambd = torch.zeros_like(lambd)
    dangle_velocity = torch.zeros_like(angle_velocity)

    dH_real_carry = torch.zeros(batch_size, d_state, d_model, device=X.device, dtype=torch.float32)
    dH_imag_carry = torch.zeros_like(dH_real_carry)
    dB_real_next = torch.zeros(batch_size, d_state, rank, device=X.device, dtype=torch.float32)
    dB_imag_next = torch.zeros_like(dB_real_next)
    dX_next = torch.zeros(batch_size, d_model, rank, device=X.device, dtype=torch.float32)
    d_angle_state_carry = torch.zeros(batch_size, d_state, device=X.device, dtype=torch.float32)

    grad_y = grad_y.view(batch_size, seqlen, rank, d_model).float()

    for idx in range(seqlen - 1, -1, -1):
        grad_Y = grad_y[:, idx]
        X_t = X[:, idx]
        Delta_t = Delta[:, idx]
        A_t = A[:, idx]
        B_t = B[:, idx]
        C_t = C[:, idx]
        lambda_t = lambd[:, idx]
        angle_velocity_t = angle_velocity[:, idx]

        H_real = h_real_steps[idx]
        H_imag = h_imag_steps[idx]
        H_prev_real = h_prev_real_steps[idx]
        H_prev_imag = h_prev_imag_steps[idx]
        rotated_H_real = rotated_h_real_steps[idx]
        rotated_H_imag = rotated_h_imag_steps[idx]
        prev_outer_real = prev_outer_real_steps[idx]
        prev_outer_imag = prev_outer_imag_steps[idx]
        curr_outer_real = curr_outer_real_steps[idx]
        curr_outer_imag = curr_outer_imag_steps[idx]
        B_real = b_real_steps[idx]
        B_imag = b_imag_steps[idx]
        C_real = c_real_steps[idx]
        C_imag = c_imag_steps[idx]
        angle_state = angle_state_steps[idx]
        theta_t = theta_steps[idx]
        alpha = alpha_steps[idx]
        beta = beta_steps[idx]
        gamma = gamma_steps[idx]

        dC_real = torch.einsum("brd,bnd->bnr", grad_Y, H_real)
        dC_imag = torch.einsum("brd,bnd->bnr", grad_Y, H_imag)
        dH_real = dH_real_carry + torch.einsum("brd,bnr->bnd", grad_Y, C_real)
        dH_imag = dH_imag_carry + torch.einsum("brd,bnr->bnd", grad_Y, C_imag)

        dalpha = torch.sum(dH_real * rotated_H_real + dH_imag * rotated_H_imag, dim=-1)
        dbeta = torch.sum(dH_real * prev_outer_real + dH_imag * prev_outer_imag, dim=-1)
        dgamma = torch.sum(dH_real * curr_outer_real + dH_imag * curr_outer_imag, dim=-1)
        d_rotated_H_real = alpha.unsqueeze(-1) * dH_real
        d_rotated_H_imag = alpha.unsqueeze(-1) * dH_imag
        d_prev_outer_real = beta.unsqueeze(-1) * dH_real
        d_prev_outer_imag = beta.unsqueeze(-1) * dH_imag
        d_curr_outer_real = gamma.unsqueeze(-1) * dH_real
        d_curr_outer_imag = gamma.unsqueeze(-1) * dH_imag

        dB_real = dB_real_next + torch.einsum("bnd,bdr->bnr", d_curr_outer_real, X_t)
        dB_imag = dB_imag_next + torch.einsum("bnd,bdr->bnr", d_curr_outer_imag, X_t)
        dX[:, idx] = (
            dX_next
            + torch.einsum("bnd,bnr->bdr", d_curr_outer_real, B_real)
            + torch.einsum("bnd,bnr->bdr", d_curr_outer_imag, B_imag)
        )

        if idx > 0:
            prev_X_t = X[:, idx - 1]
            prev_B_real_t = b_real_steps[idx - 1]
            prev_B_imag_t = b_imag_steps[idx - 1]
            dB_real_next = torch.einsum("bnd,bdr->bnr", d_prev_outer_real, prev_X_t)
            dB_imag_next = torch.einsum("bnd,bdr->bnr", d_prev_outer_imag, prev_X_t)
            dX_next = (
                torch.einsum("bnd,bnr->bdr", d_prev_outer_real, prev_B_real_t)
                + torch.einsum("bnd,bnr->bdr", d_prev_outer_imag, prev_B_imag_t)
            )
        else:
            dB_real_next.zero_()
            dB_imag_next.zero_()
            dX_next.zero_()

        angle_cos = torch.cos(angle_state).unsqueeze(-1)
        angle_sin = torch.sin(angle_state).unsqueeze(-1)
        dB[:, idx] = dB_real * angle_cos + dB_imag * angle_sin
        dC[:, idx] = dC_real * angle_cos + dC_imag * angle_sin
        d_angle_from_B = torch.sum(dB_real * (-B_t * angle_sin) + dB_imag * (B_t * angle_cos), dim=-1)
        d_angle_from_C = torch.sum(dC_real * (-C_t * angle_sin) + dC_imag * (C_t * angle_cos), dim=-1)
        d_angle_state = d_angle_from_B + d_angle_from_C + d_angle_state_carry

        step_cos = torch.cos(theta_t).unsqueeze(-1)
        step_sin = torch.sin(theta_t).unsqueeze(-1)
        dH_prev_real = d_rotated_H_real * step_cos + d_rotated_H_imag * step_sin
        dH_prev_imag = -d_rotated_H_real * step_sin + d_rotated_H_imag * step_cos
        dtheta_from_rotation = torch.sum(
            d_rotated_H_real * (-step_sin * H_prev_real - step_cos * H_prev_imag)
            + d_rotated_H_imag * (step_cos * H_prev_real - step_sin * H_prev_imag),
            dim=-1,
        )

        dtheta = dtheta_from_rotation + d_angle_state
        dDelta[:, idx] += dtheta * angle_velocity_t
        dangle_velocity[:, idx] = dtheta * Delta_t

        dlambd[:, idx] += dbeta * (-Delta_t * alpha) + dgamma * Delta_t
        dDelta[:, idx] += dbeta * (1.0 - lambda_t) * alpha + dgamma * lambda_t
        dalpha = dalpha + dbeta * (1.0 - lambda_t) * Delta_t

        alpha_arg = Delta_t * A_t
        unclamped_alpha = (alpha_arg >= -20.0) & (alpha_arg <= 20.0)
        d_alpha_arg = dalpha * alpha * unclamped_alpha.to(dtype=dalpha.dtype)
        dDelta[:, idx] += d_alpha_arg * A_t
        dA[:, idx] = d_alpha_arg * Delta_t

        dH_real_carry = dH_prev_real
        dH_imag_carry = dH_prev_imag
        d_angle_state_carry = d_angle_state

    return dX, dDelta, dA, dB, dC, dlambd, dangle_velocity


class _ComplexMIMOTrapezoidalSSMFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        X: Tensor,
        Delta: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        lambd: Tensor,
        angle_velocity: Tensor,
        z: Tensor,
        has_z: bool,
        d_model: int,
        d_state: int,
        rank: int,
    ) -> Tensor:
        ctx.has_z = has_z
        ctx.d_model = d_model
        ctx.d_state = d_state
        ctx.rank = rank
        y, h_real_cache, h_imag_cache = _run_complex_mimo_trapezoidal_ssm_cuda_with_cache(
            X=X,
            Delta=Delta,
            A=A,
            B=B,
            C=C,
            lambd=lambd,
            angle_velocity=angle_velocity,
            z=z if has_z else None,
            d_model=d_model,
            d_state=d_state,
            rank=rank,
        )
        ctx.save_for_backward(X, Delta, A, B, C, lambd, angle_velocity, z, h_real_cache, h_imag_cache)
        return y

    @staticmethod
    def backward(ctx, grad_y: Tensor) -> tuple[Tensor | None, ...]:
        X, Delta, A, B, C, lambd, angle_velocity, z, h_real_cache, h_imag_cache = ctx.saved_tensors
        try:
            grads = _run_complex_mimo_trapezoidal_ssm_backward_cuda(
                grad_y=grad_y,
                X=X,
                Delta=Delta,
                A=A,
                B=B,
                C=C,
                lambd=lambd,
                angle_velocity=angle_velocity,
                z=z if ctx.has_z else None,
                h_real_cache=h_real_cache,
                h_imag_cache=h_imag_cache,
                d_model=ctx.d_model,
                d_state=ctx.d_state,
                rank=ctx.rank,
            )
        except Exception:
            dZ = None
            if ctx.has_z:
                raw_y = _complex_mimo_trapezoidal_ssm_forward(
                    X=X,
                    Delta=Delta,
                    A=A,
                    B=B,
                    C=C,
                    lambd=lambd,
                    angle_velocity=angle_velocity,
                )
                z_sigmoid = torch.sigmoid(z)
                dZ = grad_y * raw_y * z_sigmoid * (1.0 + z * (1.0 - z_sigmoid))
                grad_y = grad_y * z * z_sigmoid
            grads = _complex_mimo_trapezoidal_ssm_backward(
                grad_y=grad_y,
                X=X,
                Delta=Delta,
                A=A,
                B=B,
                C=C,
                lambd=lambd,
                angle_velocity=angle_velocity,
            )
            grads = (*grads, dZ)
        return (*grads, None, None, None, None)


class DiSPoMamba3ResidualBlock(nn.Module):
    """Residual block using gated MIMO trapezoidal step-scaled SSM mixing."""

    def __init__(self, config: DiSPoConfig, stream_dims: dict[str, int]):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.mixer = GatedMIMOTrapezoidalSSMMixer(config, stream_dims=stream_dims)
        self.drop = nn.Dropout(config.dropout)
        self.mlp_norm = nn.LayerNorm(config.hidden_dim)
        mlp_hidden = int(config.hidden_dim * config.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(mlp_hidden, config.hidden_dim),
        )

    def forward(
        self,
        x: Tensor,
        *,
        delta_rate: Tensor,
        eta: Tensor,
        stream_context: dict[str, Tensor],
    ) -> Tensor:
        x = x + self.drop(
            self.mixer(
                self.norm(x),
                delta_rate=delta_rate,
                eta=eta,
                stream_context=stream_context,
            )
        )
        x = x + self.drop(self.mlp(self.mlp_norm(x)))
        return x


class GatedMIMOTrapezoidalSSMMixer(nn.Module):
    """Gated MIMO exponential-trapezoidal SSM mixer.

    The mixer follows the requested recurrence:

        Delta_l = r_l * softplus(f_delta(u_l))
        alpha_l = exp(Delta_l * A_l)
        beta_l = (1 - lambda_l) * Delta_l * alpha_l
        gamma_l = lambda_l * Delta_l
        H_l = Diag(alpha_l) H_{l-1}
              + beta_l B_{l-1} X_{l-1}^T
              + gamma_l B_l X_l^T

    The PyTorch implementation is kept as the portable reference path. On CUDA,
    the complex angle-state path can use Triton forward and backward kernels for
    rank 1 or 4. When enabled, the angle-state path keeps the SSM state as
    explicit real and imaginary tensors instead of using PyTorch complex dtype.
    """

    def __init__(self, config: DiSPoConfig, stream_dims: dict[str, int]):
        super().__init__()
        self.d_model = config.hidden_dim
        self.d_state = config.d_state
        self.rank = config.mamba3_mimo_rank
        self.omega_min = config.mamba3_omega_min
        self.use_rotary_angle = config.mamba3_use_rotary_angle
        self.use_complex_ssm = config.mamba3_use_complex_ssm
        self.use_output_gate = config.mamba3_use_output_gate
        self.stream_names = tuple(stream_dims)
        self.use_cuda_fast_ssm = os.getenv("LEROBOT_DISPO_MAMBA3_FAST_SSM", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        self._cuda_fast_ssm_disabled = False

        if self.rank <= 0:
            raise ValueError(f"`mamba3_mimo_rank` must be positive. Got {self.rank}.")
        if not self.stream_names:
            raise ValueError("At least one stream is required for gated MIMO fusion.")

        self.stream_encoders = nn.ModuleDict(
            {name: nn.Linear(dim, self.d_model) for name, dim in stream_dims.items()}
        )
        self.stream_gate_mlps = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(self.d_model + 2, self.d_model),
                    nn.SiLU(),
                    nn.Linear(self.d_model, self.d_model),
                )
                for name in stream_dims
            }
        )
        self.stream_rank_proj = nn.Parameter(torch.empty(len(self.stream_names), self.rank))

        self.u_proj = nn.Linear(self.d_model * self.rank, self.d_model, bias=False)
        self.r_embed = nn.Linear(1, self.d_model)
        self.eta_embed = nn.Linear(1, self.d_model)

        self.delta_proj = nn.Linear(self.d_model, self.d_state)
        self.A_proj = nn.Linear(self.d_model, self.d_state)
        self.B_proj = nn.Linear(self.d_model, self.d_state * self.rank)
        self.C_proj = nn.Linear(self.d_model, self.d_state * self.rank)
        self.lambda_proj = nn.Linear(self.d_model, self.d_state)
        if self.use_rotary_angle:
            self.angle_proj = nn.Linear(self.d_model, self.d_state)
        else:
            self.angle_proj = None
        if self.use_output_gate:
            self.z_proj = nn.Linear(self.d_model, self.rank * self.d_model)
        else:
            self.z_proj = None

        self.output_proj = nn.Linear(self.rank * self.d_model, self.d_model)
        self._init_rank_projection()

    def _init_rank_projection(self) -> None:
        with torch.no_grad():
            self.stream_rank_proj.normal_(mean=0.0, std=max(self.rank, 1) ** -0.5)

    def _expand_rate(
        self,
        value: Tensor,
        *,
        name: str,
        batch_size: int,
        seqlen: int,
        device: torch.device,
    ) -> Tensor:
        value = value.to(device=device)
        if value.ndim == 1:
            value = value.unsqueeze(1).expand(-1, seqlen)
        elif value.ndim == 2 and value.shape[1] == 1:
            value = value.expand(-1, seqlen)
        elif value.ndim != 2:
            raise ValueError(f"`{name}` must have shape (B,), (B, 1), or (B, L). Got {tuple(value.shape)}.")

        if value.shape != (batch_size, seqlen):
            raise ValueError(
                f"`{name}` must have shape {(batch_size, seqlen)} after expansion. "
                f"Got {tuple(value.shape)}."
            )
        return value.float()

    def _expand_stream(self, stream_value: Tensor, *, name: str, seqlen: int) -> Tensor:
        if stream_value.ndim == 2:
            return stream_value.unsqueeze(1).expand(-1, seqlen, -1)
        if stream_value.ndim == 3:
            if stream_value.shape[1] == seqlen:
                return stream_value
            if stream_value.shape[1] == 1:
                return stream_value.expand(-1, seqlen, -1)
        raise ValueError(
            f"Stream `{name}` must have shape (B, F), (B, 1, F), or (B, L, F). "
            f"Got {tuple(stream_value.shape)} for L={seqlen}."
        )

    def _encode_and_gate_streams(
        self,
        hidden_states: Tensor,
        *,
        delta_rate: Tensor,
        eta: Tensor,
        stream_context: dict[str, Tensor],
    ) -> Tensor:
        _, seqlen, _ = hidden_states.shape
        gated_streams = []

        for name in self.stream_names:
            if name == NOISY_ACTION_STREAM:
                stream_value = hidden_states
            else:
                if name not in stream_context:
                    raise ValueError(
                        f"Missing stream `{name}` for Mamba3 gated MIMO block. "
                        f"Available streams: {sorted(stream_context)}."
                    )
                stream_value = stream_context[name]

            stream_value = self._expand_stream(stream_value, name=name, seqlen=seqlen)
            encoded = self.stream_encoders[name](stream_value.to(dtype=hidden_states.dtype))
            gate_input = torch.cat(
                [
                    encoded,
                    delta_rate.to(dtype=encoded.dtype).unsqueeze(-1),
                    eta.to(dtype=encoded.dtype).unsqueeze(-1),
                ],
                dim=-1,
            )
            gate = torch.sigmoid(self.stream_gate_mlps[name](gate_input))
            omega = self.omega_min + (1.0 - self.omega_min) * gate
            gated_streams.append(omega * encoded)

        return torch.stack(gated_streams, dim=-1)

    def _rotate_state_vector(self, value: Tensor, angle: Tensor) -> tuple[Tensor, Tensor]:
        cos_angle = torch.cos(angle).unsqueeze(-1)
        sin_angle = torch.sin(angle).unsqueeze(-1)
        return value * cos_angle, value * sin_angle

    def _forward_real_ssm(
        self,
        *,
        X: Tensor,
        Delta: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        lambd: Tensor,
    ) -> Tensor:
        batch_size, seqlen, _, _ = X.shape
        H = torch.zeros(
            batch_size,
            self.d_state,
            self.d_model,
            device=X.device,
            dtype=torch.float32,
        )
        prev_B = torch.zeros(
            batch_size,
            self.d_state,
            self.rank,
            device=X.device,
            dtype=torch.float32,
        )
        prev_X = torch.zeros(
            batch_size,
            self.d_model,
            self.rank,
            device=X.device,
            dtype=torch.float32,
        )

        outputs = []
        for idx in range(seqlen):
            Delta_t = Delta[:, idx]
            A_t = A[:, idx]
            B_t = B[:, idx]
            C_t = C[:, idx]
            lambda_t = lambd[:, idx]
            X_t = X[:, idx]

            alpha = torch.exp((Delta_t * A_t).clamp(min=-20.0, max=20.0))
            beta = (1.0 - lambda_t) * Delta_t * alpha
            gamma = lambda_t * Delta_t

            prev_outer = torch.einsum("bnr,bdr->bnd", prev_B, prev_X)
            curr_outer = torch.einsum("bnr,bdr->bnd", B_t, X_t)
            H = alpha.unsqueeze(-1) * H + beta.unsqueeze(-1) * prev_outer + gamma.unsqueeze(-1) * curr_outer

            Y = torch.einsum("bnr,bnd->brd", C_t, H)
            outputs.append(Y.flatten(start_dim=1))

            prev_B = B_t
            prev_X = X_t

        return torch.stack(outputs, dim=1)

    def _forward_complex_ssm(
        self,
        *,
        X: Tensor,
        Delta: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        lambd: Tensor,
        angle_velocity: Tensor,
    ) -> Tensor:
        batch_size, seqlen, _, _ = X.shape
        H_real = torch.zeros(
            batch_size,
            self.d_state,
            self.d_model,
            device=X.device,
            dtype=torch.float32,
        )
        H_imag = torch.zeros_like(H_real)
        prev_B_real = torch.zeros(
            batch_size,
            self.d_state,
            self.rank,
            device=X.device,
            dtype=torch.float32,
        )
        prev_B_imag = torch.zeros_like(prev_B_real)
        prev_X = torch.zeros(
            batch_size,
            self.d_model,
            self.rank,
            device=X.device,
            dtype=torch.float32,
        )
        angle_state = torch.zeros(
            batch_size,
            self.d_state,
            device=X.device,
            dtype=torch.float32,
        )

        outputs = []
        for idx in range(seqlen):
            Delta_t = Delta[:, idx]
            A_t = A[:, idx]
            B_t = B[:, idx]
            C_t = C[:, idx]
            lambda_t = lambd[:, idx]
            X_t = X[:, idx]
            theta_t = Delta_t * angle_velocity[:, idx]

            alpha = torch.exp((Delta_t * A_t).clamp(min=-20.0, max=20.0))
            beta = (1.0 - lambda_t) * Delta_t * alpha
            gamma = lambda_t * Delta_t

            step_cos = torch.cos(theta_t).unsqueeze(-1)
            step_sin = torch.sin(theta_t).unsqueeze(-1)
            rotated_H_real = step_cos * H_real - step_sin * H_imag
            rotated_H_imag = step_sin * H_real + step_cos * H_imag

            angle_state = angle_state + theta_t
            B_real, B_imag = self._rotate_state_vector(B_t, angle_state)
            C_real, C_imag = self._rotate_state_vector(C_t, angle_state)

            prev_outer_real = torch.einsum("bnr,bdr->bnd", prev_B_real, prev_X)
            prev_outer_imag = torch.einsum("bnr,bdr->bnd", prev_B_imag, prev_X)
            curr_outer_real = torch.einsum("bnr,bdr->bnd", B_real, X_t)
            curr_outer_imag = torch.einsum("bnr,bdr->bnd", B_imag, X_t)

            H_real = (
                alpha.unsqueeze(-1) * rotated_H_real
                + beta.unsqueeze(-1) * prev_outer_real
                + gamma.unsqueeze(-1) * curr_outer_real
            )
            H_imag = (
                alpha.unsqueeze(-1) * rotated_H_imag
                + beta.unsqueeze(-1) * prev_outer_imag
                + gamma.unsqueeze(-1) * curr_outer_imag
            )

            Y_real = torch.einsum("bnr,bnd->brd", C_real, H_real)
            Y_imag = torch.einsum("bnr,bnd->brd", C_imag, H_imag)
            outputs.append((Y_real + Y_imag).flatten(start_dim=1))

            prev_B_real = B_real
            prev_B_imag = B_imag
            prev_X = X_t

        return torch.stack(outputs, dim=1)

    def _try_forward_complex_ssm_cuda(
        self,
        *,
        X: Tensor,
        Delta: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        lambd: Tensor,
        angle_velocity: Tensor,
        z: Tensor | None = None,
    ) -> Tensor | None:
        if (
            triton is None
            or not self.use_cuda_fast_ssm
            or self._cuda_fast_ssm_disabled
            or not X.is_cuda
            or X.dtype != torch.float32
        ):
            return None

        batch_size, seqlen, d_model, rank = X.shape
        if (
            d_model != self.d_model
            or rank != self.rank
            or self.d_state not in {1, 2, 4, 8, 16, 32, 64}
            or self.rank not in {1, 4}
            or seqlen <= 0
        ):
            return None

        try:
            if torch.is_grad_enabled() and any(
                tensor.requires_grad
                for tensor in (X, Delta, A, B, C, lambd, angle_velocity, z)
                if tensor is not None
            ):
                z_arg = z if z is not None else X.new_empty(0)
                return _ComplexMIMOTrapezoidalSSMFunction.apply(
                    X,
                    Delta,
                    A,
                    B,
                    C,
                    lambd,
                    angle_velocity,
                    z_arg,
                    z is not None,
                    self.d_model,
                    self.d_state,
                    self.rank,
                )
            return _run_complex_mimo_trapezoidal_ssm_cuda(
                X=X,
                Delta=Delta,
                A=A,
                B=B,
                C=C,
                lambd=lambd,
                angle_velocity=angle_velocity,
                z=z,
                d_model=self.d_model,
                d_state=self.d_state,
                rank=self.rank,
            )
        except Exception:
            self._cuda_fast_ssm_disabled = True
            return None

    def forward(
        self,
        hidden_states: Tensor,
        *,
        delta_rate: Tensor,
        eta: Tensor,
        stream_context: dict[str, Tensor],
    ) -> Tensor:
        batch_size, seqlen, _ = hidden_states.shape
        delta_rate = self._expand_rate(
            delta_rate,
            name="delta_rate",
            batch_size=batch_size,
            seqlen=seqlen,
            device=hidden_states.device,
        )
        eta = self._expand_rate(
            eta,
            name="eta",
            batch_size=batch_size,
            seqlen=seqlen,
            device=hidden_states.device,
        )

        gated_streams = self._encode_and_gate_streams(
            hidden_states,
            delta_rate=delta_rate,
            eta=eta,
            stream_context=stream_context,
        )
        rank_proj = self.stream_rank_proj.to(device=hidden_states.device, dtype=gated_streams.dtype)
        X = torch.einsum("blds,sr->bldr", gated_streams, rank_proj)

        u = self.u_proj(X.flatten(start_dim=2))
        u = u + self.r_embed(delta_rate.unsqueeze(-1)) + self.eta_embed(eta.unsqueeze(-1))

        Delta = delta_rate.unsqueeze(-1) * F.softplus(self.delta_proj(u).float())
        A = -F.softplus(self.A_proj(u).float())
        B = self.B_proj(u).float().view(batch_size, seqlen, self.d_state, self.rank)
        C = self.C_proj(u).float().view(batch_size, seqlen, self.d_state, self.rank)
        lambd = torch.sigmoid(self.lambda_proj(u).float())
        z = self.z_proj(u).float() if self.use_output_gate else None
        X = X.float()
        if self.use_complex_ssm:
            angle_velocity = torch.tanh(self.angle_proj(u).float()) * math.pi
            y = self._try_forward_complex_ssm_cuda(
                X=X,
                Delta=Delta,
                A=A,
                B=B,
                C=C,
                lambd=lambd,
                angle_velocity=angle_velocity,
                z=z,
            )
            if y is None:
                y = self._forward_complex_ssm(
                    X=X,
                    Delta=Delta,
                    A=A,
                    B=B,
                    C=C,
                    lambd=lambd,
                    angle_velocity=angle_velocity,
                )
                if z is not None:
                    y = y * z * torch.sigmoid(z)
        else:
            y = self._forward_real_ssm(
                X=X,
                Delta=Delta,
                A=A,
                B=B,
                C=C,
                lambd=lambd,
            )
            if z is not None:
                y = y * z * torch.sigmoid(z)

        y = y.to(dtype=self.output_proj.weight.dtype)
        return self.output_proj(y)
