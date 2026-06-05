#!/usr/bin/env python

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

from typing import TYPE_CHECKING

import einops
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn

from lerobot.utils.constants import (
    OBS_ENV_STATE,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from lerobot.utils.import_utils import _transformers_available, require_package

from ..pretrained import PreTrainedPolicy
from .configuration_diffusion import DiffusionV2Config
from .modeling_diffusion import (
    DiffusionConditionalUnet1d,
    DiffusionModel,
    DiffusionPolicy,
    _make_noise_scheduler,
)

if TYPE_CHECKING or _transformers_available:
    from transformers import CLIPTextModel, CLIPVisionModel
else:
    CLIPTextModel = None
    CLIPVisionModel = None


class DiffusionV2Policy(DiffusionPolicy):
    """Diffusion Policy using frozen CLIP image and language conditioning."""

    config_class = DiffusionV2Config
    name = "diffusion_v2"

    def __init__(self, config: DiffusionV2Config, **kwargs):
        require_package("diffusers", extra="diffusion")
        require_package("transformers", extra="diffusion")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self._queues = None
        self.diffusion = DiffusionV2Model(config)
        self.reset()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        queued_batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        # Language is fixed per episode; unlike observations it should not be temporally queued.
        for key in (OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK):
            if key in batch:
                queued_batch[key] = batch[key]
        return self.diffusion.generate_actions(queued_batch, noise=noise)


class DiffusionV2Model(DiffusionModel):
    """Diffusion denoiser with the same global CLIP conditioning style as streaming_flow_v3."""

    def __init__(self, config: DiffusionV2Config):
        nn.Module.__init__(self)
        self.config = config

        global_cond_dim = config.n_obs_steps * config.robot_state_feature.shape[0]
        if config.image_features:
            num_images = len(config.image_features)
            if config.use_separate_rgb_encoder_per_camera:
                self.rgb_encoder = nn.ModuleList(
                    [FrozenCLIPTemporalImageEncoder(config) for _ in range(num_images)]
                )
                global_cond_dim += num_images * self.rgb_encoder[0].feature_dim
            else:
                self.rgb_encoder = FrozenCLIPTemporalImageEncoder(config)
                global_cond_dim += num_images * self.rgb_encoder.feature_dim
        else:
            self.rgb_encoder = None

        if config.env_state_feature:
            global_cond_dim += config.n_obs_steps * config.env_state_feature.shape[0]

        self.text_encoder = FrozenCLIPTextConditionEncoder(config)
        global_cond_dim += config.n_obs_steps * self.text_encoder.feature_dim

        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim)
        if config.compile_model:
            self.unet = torch.compile(self.unet, mode=config.compile_mode)

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

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        cond_feats = []
        state = batch[OBS_STATE]
        if state.shape[1] != self.config.n_obs_steps:
            raise ValueError(
                f"`{OBS_STATE}` must contain {self.config.n_obs_steps} observation steps. "
                f"Got shape {tuple(state.shape)}."
            )
        cond_feats.append(state.flatten(start_dim=1))

        if self.config.image_features:
            images = batch[OBS_IMAGES]
            if images.ndim != 6:
                raise ValueError(
                    f"`{OBS_IMAGES}` must have shape (B, S, N, C, H, W). Got {tuple(images.shape)}."
                )
            if self.config.use_separate_rgb_encoder_per_camera:
                images_per_camera = einops.rearrange(images, "b s n c h w -> n b s c h w")
                image_feats = [
                    encoder(camera_images)
                    for encoder, camera_images in zip(self.rgb_encoder, images_per_camera, strict=True)
                ]
                cond_feats.append(torch.cat(image_feats, dim=-1))
            else:
                flat_images = einops.rearrange(images, "b s n c h w -> (b n) s c h w")
                image_feats = self.rgb_encoder(flat_images)
                cond_feats.append(
                    einops.rearrange(
                        image_feats,
                        "(b n) f -> b (n f)",
                        b=images.shape[0],
                        n=images.shape[2],
                    )
                )

        if self.config.env_state_feature:
            cond_feats.append(batch[OBS_ENV_STATE].flatten(start_dim=1))

        if OBS_LANGUAGE_TOKENS not in batch or OBS_LANGUAGE_ATTENTION_MASK not in batch:
            raise ValueError(
                "DiffusionV2 requires tokenized task text for CLIP text conditioning. "
                f"Expected `{OBS_LANGUAGE_TOKENS}` and `{OBS_LANGUAGE_ATTENTION_MASK}`. "
                f"Available keys: {list(batch)}"
            )
        text_feats = self.text_encoder(
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
        )
        cond_feats.append(text_feats.unsqueeze(1).expand(-1, self.config.n_obs_steps, -1).flatten(start_dim=1))

        global_cond = torch.cat(cond_feats, dim=-1)
        return F.normalize(global_cond, dim=-1) if global_cond.shape[-1] > 1 else global_cond


class FrozenCLIPTemporalImageEncoder(nn.Module):
    """Frozen CLIP vision encoder with trainable temporal feature projection."""

    def __init__(self, config: DiffusionV2Config):
        super().__init__()
        self.config = config
        self.model = CLIPVisionModel.from_pretrained(config.vision_encoder_name)
        self.feature_dim = config.image_feature_dim

        if config.freeze_clip:
            self.model.requires_grad_(False)
            self.model.eval()

        self.resize = (
            torchvision.transforms.Resize(
                config.clip_image_resize_shape,
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                antialias=True,
            )
            if config.clip_image_resize_shape is not None
            else None
        )
        if config.clip_image_crop_shape is not None:
            self.center_crop = torchvision.transforms.CenterCrop(config.clip_image_crop_shape)
            self.maybe_random_crop = (
                torchvision.transforms.RandomCrop(config.clip_image_crop_shape)
                if config.clip_image_crop_is_random
                else self.center_crop
            )
        else:
            self.center_crop = None
            self.maybe_random_crop = None

        self.register_buffer(
            "clip_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "clip_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(1, 3, 1, 1),
        )

        temporal_dim = self.model.config.hidden_size * (3 if config.n_obs_steps >= 3 else 2)
        self.proj = nn.Sequential(nn.LayerNorm(temporal_dim), nn.Linear(temporal_dim, self.feature_dim))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_clip:
            self.model.eval()
        return self

    def forward(self, obs: Tensor) -> Tensor:
        bsize, obs_steps, channels, height, width = obs.shape
        obs = obs.reshape(bsize * obs_steps, channels, height, width).float()
        if self.resize is not None:
            obs = self.resize(obs)
        if self.center_crop is not None:
            obs = self.maybe_random_crop(obs) if self.training else self.center_crop(obs)
        obs = (obs - self.clip_mean.to(dtype=obs.dtype)) / self.clip_std.to(dtype=obs.dtype)

        if self.config.freeze_clip:
            with torch.no_grad():
                features = self.model(pixel_values=obs).last_hidden_state[:, 0]
        else:
            features = self.model(pixel_values=obs).last_hidden_state[:, 0]
        features = features.reshape(bsize, obs_steps, -1)

        if obs_steps >= 3:
            fused = torch.cat(
                [
                    features[:, -1],
                    features[:, -1] - features[:, -2],
                    features[:, -1] - 2 * features[:, -2] + features[:, -3],
                ],
                dim=-1,
            )
        elif obs_steps == 2:
            fused = torch.cat([features[:, -1], features[:, -1] - features[:, 0]], dim=-1)
        else:
            fused = torch.cat([features[:, -1], torch.zeros_like(features[:, -1])], dim=-1)
        return self.proj(fused)


class FrozenCLIPTextConditionEncoder(nn.Module):
    """Frozen CLIP text encoder with trainable projection."""

    def __init__(self, config: DiffusionV2Config):
        super().__init__()
        self.config = config
        self.text_encoder = CLIPTextModel.from_pretrained(config.text_encoder_name)
        self.feature_dim = config.text_projection_dim
        if config.freeze_clip:
            self.text_encoder.requires_grad_(False)
            self.text_encoder.eval()
        self.proj = nn.Sequential(
            nn.LayerNorm(self.text_encoder.config.hidden_size),
            nn.Linear(self.text_encoder.config.hidden_size, self.feature_dim),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_clip:
            self.text_encoder.eval()
        return self

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        input_ids = input_ids.long()
        attention_mask = attention_mask.long()
        if input_ids.ndim > 2:
            input_ids = input_ids.reshape(input_ids.shape[0], -1)
        if attention_mask.ndim > 2:
            attention_mask = attention_mask.reshape(attention_mask.shape[0], -1)
        if self.config.freeze_clip:
            with torch.no_grad():
                features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        else:
            features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        return self.proj(features)
