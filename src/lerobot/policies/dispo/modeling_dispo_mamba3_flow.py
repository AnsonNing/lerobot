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

import os
from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch.distributions import Beta

from lerobot.utils.constants import ACTION, OBS_IMAGES
from lerobot.utils.import_utils import require_package

from ..pretrained import PreTrainedPolicy
from ..utils import get_device_from_parameters, get_dtype_from_parameters, populate_queues
from .configuration_dispo import DiSPoMamba3FlowConfig
from .modeling_dispo import DiSPoDiffusionModel, DiSPoPolicy, _state_feature_keys


def _set_mamba3_triton_kernel_defaults() -> None:
    os.environ.setdefault("LEROBOT_DISPO_MAMBA3_FAST_SSM", "1")
    os.environ.setdefault("LEROBOT_DISPO_MAMBA3_FAST_BWD", "1")
    os.environ.setdefault("LEROBOT_DISPO_MAMBA3_BWD_BLOCK_D", "16")


class DiSPoMamba3FlowPolicy(PreTrainedPolicy):
    config_class = DiSPoMamba3FlowConfig
    name = "dispo_mamba3_flow"

    def __init__(self, config: DiSPoMamba3FlowConfig, **kwargs):
        _set_mamba3_triton_kernel_defaults()
        require_package("diffusers", extra="diffusion")
        super().__init__(config)
        config.validate_features()
        self.config = config
        self._state_feature_keys = _state_feature_keys(config)
        self.model = DiSPoMamba3FlowModel(config)
        self.reset()

    def get_optim_params(self) -> dict:
        return self.model.parameters()

    def reset(self):
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}
        for key in self._state_feature_keys:
            self._queues[key] = deque(maxlen=self.config.n_obs_steps)
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)

    def _prepare_image_batch(
        self,
        batch: dict[str, Tensor],
        ensure_obs_steps: bool = False,
    ) -> dict[str, Tensor]:
        return DiSPoPolicy._prepare_image_batch(self, batch, ensure_obs_steps=ensure_obs_steps)

    def _queued_batch(self) -> dict[str, Tensor]:
        return {
            key: torch.stack(list(queue), dim=1)
            for key, queue in self._queues.items()
            if key != ACTION and len(queue) > 0
        }

    def _action_chunk_for_queue(self, actions: Tensor) -> Tensor:
        return DiSPoPolicy._action_chunk_for_queue(self, actions)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        batch = self._prepare_image_batch(batch)
        self._queues = populate_queues(self._queues, batch)
        queued_batch = self._queued_batch()
        actions = self.model.generate_actions(queued_batch, noise=noise)
        return self._action_chunk_for_queue(actions)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        if ACTION in batch:
            batch = dict(batch)
            batch.pop(ACTION)

        batch = self._prepare_image_batch(batch)
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            queued_batch = self._queued_batch()
            actions = self.model.generate_actions(queued_batch, noise=noise)
            actions = self._action_chunk_for_queue(actions)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        return self._queues[ACTION].popleft()

    def forward(
        self,
        batch: dict[str, Tensor],
        reduction: str = "mean",
    ) -> tuple[Tensor, dict[str, float] | None]:
        batch = self._prepare_image_batch(batch, ensure_obs_steps=True)
        loss = self.model.compute_loss(batch, reduction=reduction)
        return loss, None


class DiSPoMamba3FlowModel(DiSPoDiffusionModel):
    """GR00T-style flow matching action generator for DiSPo-Mamba3."""

    def __init__(self, config: DiSPoMamba3FlowConfig):
        _set_mamba3_triton_kernel_defaults()
        if config.ssm_block_type != "mamba3_gated_mimo":
            raise ValueError("DiSPoMamba3FlowModel requires `ssm_block_type='mamba3_gated_mimo'`.")
        super().__init__(config)
        self.config = config
        self._beta_dist: Beta | None = None

    def _sample_flow_time(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if self._beta_dist is None:
            self._beta_dist = Beta(
                self.config.flow_noise_beta_alpha,
                self.config.flow_noise_beta_beta,
                validate_args=False,
            )
        sample = self._beta_dist.sample([batch_size]).to(device=device, dtype=dtype)
        return ((self.config.flow_noise_s - sample) / self.config.flow_noise_s).clamp(0.0, 1.0)

    def _flow_timestep(self, t: Tensor) -> Tensor:
        max_bucket = self.config.flow_num_timestep_buckets - 1
        return (t * max_bucket).long().clamp(min=0, max=max_bucket)

    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor,
        stream_cond: dict[str, Tensor] | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)
        actions = (
            noise
            if noise is not None
            else torch.randn(
                size=(batch_size, self.config.horizon, self.config.action_feature.shape[0]),
                dtype=dtype,
                device=device,
            )
        )
        num_steps = int(self.config.num_inference_steps)
        dt = 1.0 / num_steps
        delta_rate = self._eval_delta_rate(batch_size, self.config.horizon, actions.device)

        for step in range(num_steps):
            t_cont = torch.full(
                (batch_size,),
                step / float(num_steps),
                dtype=torch.float32,
                device=actions.device,
            )
            pred_velocity = self.denoiser(
                actions,
                self._flow_timestep(t_cont),
                global_cond=global_cond,
                stream_cond=stream_cond,
                delta_rate=delta_rate,
            )
            actions = actions + dt * pred_velocity

        return actions

    def compute_loss(self, batch: dict[str, Tensor], reduction: str = "mean") -> Tensor:
        if ACTION not in batch:
            raise ValueError(f"Missing `{ACTION}` in batch. Available keys: {list(batch)}")
        trajectory = batch[ACTION]
        if trajectory.shape[1] != self.config.horizon:
            raise ValueError(
                f"`{ACTION}` must contain horizon={self.config.horizon} steps. "
                f"Got shape {tuple(trajectory.shape)}."
            )

        global_cond, stream_cond = self._prepare_conditioning(batch)
        noise = torch.randn_like(trajectory)
        t = self._sample_flow_time(trajectory.shape[0], device=trajectory.device, dtype=trajectory.dtype)
        t_broadcast = t[:, None, None]
        noisy_trajectory = (1.0 - t_broadcast) * noise + t_broadcast * trajectory
        target_velocity = trajectory - noise

        timesteps = self._flow_timestep(t)
        delta_rate = self._sample_delta_rate(trajectory.shape[0], trajectory.shape[1], trajectory.device)
        pred_velocity = self.denoiser(
            noisy_trajectory,
            timesteps,
            global_cond=global_cond,
            stream_cond=stream_cond,
            delta_rate=delta_rate,
        )

        loss = F.mse_loss(pred_velocity, target_velocity, reduction="none")
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError("`action_is_pad` is required when `do_mask_loss_for_padding=True`.")
            valid = ~batch["action_is_pad"]
            loss = loss.mean(dim=-1)
            per_sample = (loss * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)
        else:
            per_sample = loss.flatten(start_dim=1).mean(dim=-1)

        if reduction == "none":
            return per_sample
        if reduction == "sum":
            return per_sample.sum()
        if reduction != "mean":
            raise ValueError(f"Unsupported reduction: {reduction}")
        return per_sample.mean()


__all__ = ["DiSPoMamba3FlowConfig", "DiSPoMamba3FlowModel", "DiSPoMamba3FlowPolicy"]
