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
from collections import deque

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.configs import FeatureType
from lerobot.utils.constants import ACTION, OBS_IMAGES
from lerobot.utils.import_utils import require_package

from ..diffusion.modeling_diffusion import (
    DiffusionRgbEncoder,
    DiffusionSinusoidalPosEmb,
    _make_noise_scheduler,
)
from ..pretrained import PreTrainedPolicy
from ..utils import get_device_from_parameters, get_dtype_from_parameters, populate_queues
from .configuration_dispo import DiSPoConfig
from .modeling_dispo_mamba3 import (
    FUSED_OBSERVATION_STREAM,
    GLOBAL_VISUAL_STREAM,
    GRANULARITY_CONDITION_STREAM,
    LOCAL_OR_WRIST_VISUAL_STREAM,
    NOISY_ACTION_STREAM,
    PROPRIO_STREAM,
    DiSPoMamba3ResidualBlock,
)


def _prod(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


def _state_feature_keys(config: DiSPoConfig) -> tuple[str, ...]:
    return tuple(
        key for key, ft in (config.input_features or {}).items() if ft.type is FeatureType.STATE
    )


class DiSPoPolicy(PreTrainedPolicy):
    config_class = DiSPoConfig
    name = "dispo"

    def __init__(self, config: DiSPoConfig, **kwargs):
        require_package("diffusers", extra="diffusion")
        super().__init__(config)
        config.validate_features()
        self.config = config
        self._state_feature_keys = _state_feature_keys(config)
        self.model = DiSPoDiffusionModel(config)
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
        if not self.config.image_features:
            return batch

        batch = dict(batch)
        image_tensors = []
        for key in self.config.image_features:
            if key in batch:
                image_tensors.append(batch[key])

        if not image_tensors:
            raise ValueError(
                "DiSPoPolicy expected at least one image feature in the batch, "
                f"but got keys: {list(batch)}"
            )

        image_ndims = {image.ndim for image in image_tensors}
        if len(image_ndims) != 1:
            raise ValueError(
                "All image features must have the same rank before they can be stacked. "
                f"Got ranks {sorted(image_ndims)}."
            )

        image_ndim = image_tensors[0].ndim
        if image_ndim == 4:
            images = torch.stack(image_tensors, dim=1)
            if ensure_obs_steps:
                images = images.unsqueeze(1).expand(-1, self.config.n_obs_steps, -1, -1, -1, -1)
            batch[OBS_IMAGES] = images
        elif image_ndim == 5:
            batch[OBS_IMAGES] = torch.stack(image_tensors, dim=2)
        elif image_ndim == 6:
            batch[OBS_IMAGES] = torch.cat(image_tensors, dim=2)
        else:
            raise ValueError(
                "DiSPoPolicy expects image tensors with shape (B, C, H, W) for rollout or "
                f"(B, S, C, H, W) for training. Got rank {image_ndim}."
            )
        return batch

    def _queued_batch(self) -> dict[str, Tensor]:
        return {
            key: torch.stack(list(queue), dim=1)
            for key, queue in self._queues.items()
            if key != ACTION and len(queue) > 0
        }

    def _action_chunk_for_queue(self, actions: Tensor) -> Tensor:
        if actions.shape[1] == self.config.horizon:
            start = self.config.n_obs_steps - 1
            return actions[:, start : start + self.config.n_action_steps]
        return actions[:, : self.config.n_action_steps]

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


class DiSPoDiffusionModel(nn.Module):
    def __init__(self, config: DiSPoConfig):
        super().__init__()
        self.config = config
        self.state_feature_keys = _state_feature_keys(config)

        global_cond_dim = 0
        stream_dims = {}
        proprio_dim = 0
        for key in self.state_feature_keys:
            dim = config.n_obs_steps * _prod(config.input_features[key].shape)
            global_cond_dim += dim
            proprio_dim += dim
        if proprio_dim > 0:
            stream_dims[PROPRIO_STREAM] = proprio_dim

        if config.image_features:
            num_images = len(config.image_features)
            if config.use_separate_rgb_encoder_per_camera:
                self.rgb_encoder = nn.ModuleList([DiffusionRgbEncoder(config) for _ in range(num_images)])
                image_feature_dim = self.rgb_encoder[0].feature_dim
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                image_feature_dim = self.rgb_encoder.feature_dim
            global_cond_dim += config.n_obs_steps * num_images * image_feature_dim
            stream_dims[GLOBAL_VISUAL_STREAM] = config.n_obs_steps * image_feature_dim
            if num_images > 1:
                stream_dims[LOCAL_OR_WRIST_VISUAL_STREAM] = (
                    config.n_obs_steps * (num_images - 1) * image_feature_dim
                )
        else:
            self.rgb_encoder = None

        if config.ssm_block_type == "mamba3_gated_mimo":
            if config.mamba3_single_stream_fallback and not stream_dims:
                stream_dims[FUSED_OBSERVATION_STREAM] = global_cond_dim
            stream_dims[NOISY_ACTION_STREAM] = config.hidden_dim
            stream_dims[GRANULARITY_CONDITION_STREAM] = config.diffusion_step_embed_dim + 2
        else:
            stream_dims = {}

        self.denoiser = DiSPoDenoiser(
            config,
            global_cond_dim=global_cond_dim,
            stream_dims=stream_dims,
        )
        if config.compile_model:
            self.denoiser = torch.compile(self.denoiser, mode=config.compile_mode)

        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )
        self.num_inference_steps = (
            self.noise_scheduler.config.num_train_timesteps
            if config.num_inference_steps is None
            else config.num_inference_steps
        )

    def _batch_size_from_observation(self, batch: dict[str, Tensor]) -> int:
        if self.state_feature_keys:
            return batch[self.state_feature_keys[0]].shape[0]
        if OBS_IMAGES in batch:
            return batch[OBS_IMAGES].shape[0]
        raise ValueError(f"Cannot infer batch size from keys: {list(batch)}")

    def _prepare_conditioning(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        cond_feats = []
        stream_cond = {}
        batch_size = self._batch_size_from_observation(batch)
        proprio_feats = []

        for key in self.state_feature_keys:
            if key not in batch:
                raise ValueError(f"Missing state feature `{key}` in batch. Available keys: {list(batch)}")
            state = batch[key]
            if state.ndim < 3:
                state = state.unsqueeze(1)
            if state.shape[1] != self.config.n_obs_steps:
                raise ValueError(
                    f"`{key}` must contain {self.config.n_obs_steps} observation steps. "
                    f"Got shape {tuple(state.shape)}."
                )
            state = state.flatten(start_dim=2)
            cond_feats.append(state)
            proprio_feats.append(state)

        if proprio_feats:
            stream_cond[PROPRIO_STREAM] = torch.cat(proprio_feats, dim=-1).flatten(start_dim=1)

        if self.config.image_features:
            if OBS_IMAGES not in batch:
                raise ValueError(f"Missing `{OBS_IMAGES}` in batch. Available keys: {list(batch)}")
            images = batch[OBS_IMAGES]
            if images.ndim == 5:
                images = images.unsqueeze(1)
            if images.ndim != 6:
                raise ValueError(
                    f"`{OBS_IMAGES}` must have shape (B, N, C, H, W) or (B, S, N, C, H, W). "
                    f"Got {tuple(images.shape)}."
                )
            if images.shape[1] != self.config.n_obs_steps:
                raise ValueError(
                    f"`{OBS_IMAGES}` must contain {self.config.n_obs_steps} observation steps. "
                    f"Got shape {tuple(images.shape)}."
                )

            if self.config.use_separate_rgb_encoder_per_camera:
                images_per_camera = einops.rearrange(images, "b s n c h w -> n (b s) c h w")
                image_feats = [
                    encoder(camera_images)
                    for encoder, camera_images in zip(self.rgb_encoder, images_per_camera, strict=True)
                ]
                image_feats = [
                    einops.rearrange(
                        camera_feats,
                        "(b s) f -> b s f",
                        b=batch_size,
                        s=self.config.n_obs_steps,
                    )
                    for camera_feats in image_feats
                ]
                image_feats_by_camera = torch.stack(image_feats, dim=2)
            else:
                flat_images = einops.rearrange(images, "b s n c h w -> (b s n) c h w")
                image_feats = self.rgb_encoder(flat_images)
                image_feats_by_camera = einops.rearrange(
                    image_feats,
                    "(b s n) f -> b s n f",
                    b=batch_size,
                    s=self.config.n_obs_steps,
                    n=images.shape[2],
                )

            cond_feats.append(image_feats_by_camera.flatten(start_dim=2))
            stream_cond[GLOBAL_VISUAL_STREAM] = image_feats_by_camera[:, :, 0].flatten(start_dim=1)
            if image_feats_by_camera.shape[2] > 1:
                stream_cond[LOCAL_OR_WRIST_VISUAL_STREAM] = image_feats_by_camera[:, :, 1:].flatten(
                    start_dim=1
                )

        if not cond_feats:
            raise ValueError("DiSPoPolicy requires at least one observation feature for conditioning.")

        global_cond = torch.cat(cond_feats, dim=-1).flatten(start_dim=1)
        if self.config.mamba3_single_stream_fallback and FUSED_OBSERVATION_STREAM not in stream_cond:
            stream_cond[FUSED_OBSERVATION_STREAM] = global_cond
        return global_cond, stream_cond

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        global_cond, _ = self._prepare_conditioning(batch)
        return global_cond

    def _sample_delta_rate(self, batch_size: int, seqlen: int, device: torch.device) -> Tensor:
        choices = torch.as_tensor(self.config.train_delta_rate_choices, device=device, dtype=torch.float32)
        indices = torch.randint(choices.numel(), (batch_size,), device=device)
        return choices[indices].unsqueeze(-1).expand(batch_size, seqlen)

    def _eval_delta_rate(self, batch_size: int, seqlen: int, device: torch.device) -> Tensor:
        return torch.full((batch_size, seqlen), self.config.eval_delta_rate, device=device)

    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor,
        stream_cond: dict[str, Tensor] | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)
        sample = (
            noise
            if noise is not None
            else torch.randn(
                size=(batch_size, self.config.horizon, self.config.action_feature.shape[0]),
                dtype=dtype,
                device=device,
            )
        )
        delta_rate = self._eval_delta_rate(batch_size, self.config.horizon, sample.device)

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            model_output = self.denoiser(
                sample,
                torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                global_cond=global_cond,
                stream_cond=stream_cond,
                delta_rate=delta_rate,
            )
            sample = self.noise_scheduler.step(model_output, t, sample).prev_sample
        return sample

    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        batch_size = self._batch_size_from_observation(batch)
        global_cond, stream_cond = self._prepare_conditioning(batch)
        actions = self.conditional_sample(
            batch_size,
            global_cond=global_cond,
            stream_cond=stream_cond,
            noise=noise,
        )

        start = self.config.n_obs_steps - 1
        end = start + self.config.n_action_steps
        return actions[:, start:end]

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
        eps = torch.randn_like(trajectory)
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)
        delta_rate = self._sample_delta_rate(trajectory.shape[0], trajectory.shape[1], trajectory.device)
        pred = self.denoiser(
            noisy_trajectory,
            timesteps,
            global_cond=global_cond,
            stream_cond=stream_cond,
            delta_rate=delta_rate,
        )

        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none")
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


class DiSPoDenoiser(nn.Module):
    def __init__(self, config: DiSPoConfig, global_cond_dim: int, stream_dims: dict[str, int] | None = None):
        super().__init__()
        self.config = config
        self.uses_mamba3_gated_mimo = config.ssm_block_type == "mamba3_gated_mimo"
        self.stream_dims = stream_dims or {}
        self.input_proj = nn.Linear(config.action_feature.shape[0], config.hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.horizon, config.hidden_dim))
        self.diffusion_step_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )
        self.cond_encoder = nn.Sequential(
            nn.Linear(config.diffusion_step_embed_dim + global_cond_dim, config.hidden_dim * 4),
            nn.Mish(),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim * 2),
        )
        if self.uses_mamba3_gated_mimo:
            self.blocks = nn.ModuleList(
                [DiSPoMamba3ResidualBlock(config, stream_dims=self.stream_dims) for _ in range(config.depth)]
            )
        else:
            self.blocks = nn.ModuleList([DiSPoResidualBlock(config) for _ in range(config.depth)])
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.output_proj = nn.Linear(config.hidden_dim, config.action_feature.shape[0])
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None and not hasattr(module.bias, "_no_reinit"):
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        sample: Tensor,
        timestep: Tensor | int,
        global_cond: Tensor,
        stream_cond: dict[str, Tensor] | None = None,
        delta_rate: Tensor | None = None,
    ) -> Tensor:
        if not torch.is_tensor(timestep):
            timestep = torch.full(sample.shape[:1], timestep, dtype=torch.long, device=sample.device)
        elif timestep.ndim == 0:
            timestep = timestep[None].expand(sample.shape[0]).to(sample.device)
        else:
            timestep = timestep.to(sample.device)

        x = self.input_proj(sample) + self.pos_emb[:, : sample.shape[1]]
        timestep_emb = self.diffusion_step_encoder(timestep)
        cond = torch.cat([timestep_emb, global_cond], dim=-1)
        scale, bias = self.cond_encoder(cond).chunk(2, dim=-1)
        x = x * (1 + scale.unsqueeze(1)) + bias.unsqueeze(1)

        if delta_rate is None:
            delta_rate = torch.ones(sample.shape[:2], device=sample.device, dtype=torch.float32)
        eta = timestep.float() / max(self.config.num_train_timesteps - 1, 1)
        eta = eta.unsqueeze(1).expand(-1, sample.shape[1])

        if self.uses_mamba3_gated_mimo:
            stream_context = dict(stream_cond or {})
            if FUSED_OBSERVATION_STREAM in self.stream_dims and FUSED_OBSERVATION_STREAM not in stream_context:
                stream_context[FUSED_OBSERVATION_STREAM] = global_cond
            granularity = torch.cat(
                [
                    timestep_emb.unsqueeze(1).expand(-1, sample.shape[1], -1),
                    delta_rate.to(dtype=timestep_emb.dtype).unsqueeze(-1),
                    eta.to(dtype=timestep_emb.dtype).unsqueeze(-1),
                ],
                dim=-1,
            )
            stream_context[GRANULARITY_CONDITION_STREAM] = granularity
            for block in self.blocks:
                x = block(
                    x,
                    delta_rate=delta_rate,
                    eta=eta,
                    stream_context=stream_context,
                )
        else:
            for block in self.blocks:
                x = block(x, delta_rate=delta_rate)

        return self.output_proj(self.norm(x))


class DiSPoResidualBlock(nn.Module):
    def __init__(self, config: DiSPoConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.mixer = PureTorchDiSPoMixer(config)
        self.drop = nn.Dropout(config.dropout)
        self.mlp_norm = nn.LayerNorm(config.hidden_dim)
        mlp_hidden = int(config.hidden_dim * config.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(mlp_hidden, config.hidden_dim),
        )

    def forward(self, x: Tensor, delta_rate: Tensor | None = None) -> Tensor:
        x = x + self.drop(self.mixer(self.norm(x), delta_rate=delta_rate))
        x = x + self.drop(self.mlp(self.mlp_norm(x)))
        return x


class PureTorchDiSPoMixer(nn.Module):
    """Mamba/DiSPo-style selective SSM mixer implemented without custom CUDA."""

    def __init__(self, config: DiSPoConfig):
        super().__init__()
        self.d_model = config.hidden_dim
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.expand = config.expand
        self.d_inner = config.hidden_dim * config.expand
        self.dt_rank = math.ceil(config.hidden_dim / 16) if config.dt_rank is None else config.dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        self.activation = nn.SiLU()

        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt_init_std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

    def forward(self, hidden_states: Tensor, delta_rate: Tensor | None = None) -> Tensor:
        batch, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)

        x = einops.rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[..., :seqlen]
        x = self.activation(x)
        x = einops.rearrange(x, "b d l -> b l d")

        x_dbl = self.x_proj(x)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        if delta_rate is not None:
            dt = dt * delta_rate.to(device=dt.device, dtype=dt.dtype).unsqueeze(-1)

        A = -torch.exp(self.A_log.float())
        D = self.D.to(dtype=x.dtype)
        state = torch.zeros(batch, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(seqlen):
            dt_t = dt[:, t].float()
            x_t = x[:, t]
            B_t = B[:, t].float()
            C_t = C[:, t].float()
            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0)).to(dtype=x.dtype)
            dB = (dt_t.unsqueeze(-1) * B_t.unsqueeze(1)).to(dtype=x.dtype)
            state = state * dA + x_t.unsqueeze(-1) * dB
            y_t = (state.float() * C_t.unsqueeze(1)).sum(dim=-1).to(dtype=x.dtype)
            y_t = y_t + D * x_t
            y_t = y_t * self.activation(z[:, t])
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        return self.out_proj(y)
