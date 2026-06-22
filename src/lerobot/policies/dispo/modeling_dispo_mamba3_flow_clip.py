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

from collections import deque
from typing import TYPE_CHECKING

import einops
import torch
import torchvision
from torch import Tensor, nn

from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)
from lerobot.utils.import_utils import _transformers_available, require_package

from ..pretrained import PreTrainedPolicy
from ..utils import populate_queues
from .configuration_dispo import DiSPoMamba3FlowClipConfig
from .modeling_dispo import (
    DiSPoDenoiser,
    DiSPoPolicy,
    _prod,
    _state_feature_keys,
)
from .modeling_dispo_mamba3 import (
    FUSED_OBSERVATION_STREAM,
    GLOBAL_VISUAL_STREAM,
    GRANULARITY_CONDITION_STREAM,
    LOCAL_OR_WRIST_VISUAL_STREAM,
    NOISY_ACTION_STREAM,
    PROPRIO_STREAM,
    TASK_TEXT_STREAM,
)
from .modeling_dispo_mamba3_flow import DiSPoMamba3FlowModel, _set_mamba3_triton_kernel_defaults

if TYPE_CHECKING or _transformers_available:
    from transformers import CLIPTextModel, CLIPVisionModel
else:
    CLIPTextModel = None
    CLIPVisionModel = None


class DiSPoMamba3FlowClipPolicy(PreTrainedPolicy):
    """DiSPo-Mamba3 flow policy with CLIP image/text conditioning."""

    config_class = DiSPoMamba3FlowClipConfig
    name = "dispo_mamba3_flow_clip"

    def __init__(self, config: DiSPoMamba3FlowClipConfig, **kwargs):
        _set_mamba3_triton_kernel_defaults()
        require_package("diffusers", extra="diffusion")
        require_package("transformers", extra="multi_task_dit")
        super().__init__(config)
        config.validate_features()
        self.config = config
        self._state_feature_keys = _state_feature_keys(config)
        self.model = DiSPoMamba3FlowClipModel(config)
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

    def _queued_batch(self, batch: dict[str, Tensor] | None = None) -> dict[str, Tensor]:
        queued = {
            key: torch.stack(list(queue), dim=1)
            for key, queue in self._queues.items()
            if key != ACTION and len(queue) > 0
        }
        if batch is not None:
            for key in (OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK):
                if key in batch:
                    queued[key] = batch[key]
        return queued

    def _action_chunk_for_queue(self, actions: Tensor) -> Tensor:
        return DiSPoPolicy._action_chunk_for_queue(self, actions)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        batch = self._prepare_image_batch(batch)
        self._queues = populate_queues(self._queues, batch)
        queued_batch = self._queued_batch(batch)
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
            queued_batch = self._queued_batch(batch)
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


class DiSPoMamba3FlowClipModel(DiSPoMamba3FlowModel):
    """Flow matching DiSPo-Mamba3 model using CLIP image and language streams."""

    def __init__(self, config: DiSPoMamba3FlowClipConfig):
        _set_mamba3_triton_kernel_defaults()
        nn.Module.__init__(self)
        if config.ssm_block_type != "mamba3_gated_mimo":
            raise ValueError("DiSPoMamba3FlowClipModel requires `ssm_block_type='mamba3_gated_mimo'`.")
        self.config = config
        self.state_feature_keys = _state_feature_keys(config)

        global_cond_dim = 0
        stream_dims: dict[str, int] = {}
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
                self.rgb_encoder = nn.ModuleList([CLIPSummaryImageEncoder(config) for _ in range(num_images)])
                image_feature_dim = self.rgb_encoder[0].feature_dim
            else:
                self.rgb_encoder = CLIPSummaryImageEncoder(config)
                image_feature_dim = self.rgb_encoder.feature_dim
            global_cond_dim += num_images * image_feature_dim
            stream_dims[GLOBAL_VISUAL_STREAM] = image_feature_dim
            if num_images > 1:
                stream_dims[LOCAL_OR_WRIST_VISUAL_STREAM] = (num_images - 1) * image_feature_dim
        else:
            self.rgb_encoder = None

        self.text_encoder = CLIPTextConditionEncoder(config)
        global_cond_dim += self.text_encoder.feature_dim
        stream_dims[TASK_TEXT_STREAM] = self.text_encoder.feature_dim

        stream_dims[NOISY_ACTION_STREAM] = config.hidden_dim
        stream_dims[GRANULARITY_CONDITION_STREAM] = config.diffusion_step_embed_dim + 2
        if config.mamba3_single_stream_fallback and not stream_dims:
            stream_dims[FUSED_OBSERVATION_STREAM] = global_cond_dim

        self.denoiser = DiSPoDenoiser(config, global_cond_dim=global_cond_dim, stream_dims=stream_dims)
        if config.compile_model:
            self.denoiser = torch.compile(self.denoiser, mode=config.compile_mode)
        self._beta_dist = None

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
            cond_feats.append(state.flatten(start_dim=1))
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
                images_per_camera = einops.rearrange(images, "b s n c h w -> n b s c h w")
                image_feats = [
                    encoder(camera_images)
                    for encoder, camera_images in zip(self.rgb_encoder, images_per_camera, strict=True)
                ]
                image_feats_by_camera = torch.stack(image_feats, dim=1)
            else:
                flat_images = einops.rearrange(images, "b s n c h w -> (b n) s c h w")
                image_feats = self.rgb_encoder(flat_images)
                image_feats_by_camera = einops.rearrange(
                    image_feats,
                    "(b n) f -> b n f",
                    b=batch_size,
                    n=images.shape[2],
                )

            cond_feats.append(image_feats_by_camera.flatten(start_dim=1))
            stream_cond[GLOBAL_VISUAL_STREAM] = image_feats_by_camera[:, 0]
            if image_feats_by_camera.shape[1] > 1:
                stream_cond[LOCAL_OR_WRIST_VISUAL_STREAM] = image_feats_by_camera[:, 1:].flatten(
                    start_dim=1
                )

        if OBS_LANGUAGE_TOKENS not in batch or OBS_LANGUAGE_ATTENTION_MASK not in batch:
            raise ValueError(
                "DiSPoMamba3FlowClipModel requires tokenized task text. "
                f"Expected `{OBS_LANGUAGE_TOKENS}` and `{OBS_LANGUAGE_ATTENTION_MASK}`. "
                f"Available keys: {list(batch)}."
            )
        text_feats = self.text_encoder(batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK])
        cond_feats.append(text_feats)
        stream_cond[TASK_TEXT_STREAM] = text_feats

        if not cond_feats:
            raise ValueError("DiSPoMamba3FlowClipPolicy requires observation conditioning.")

        global_cond = torch.cat(cond_feats, dim=-1)
        if self.config.mamba3_single_stream_fallback and FUSED_OBSERVATION_STREAM not in stream_cond:
            stream_cond[FUSED_OBSERVATION_STREAM] = global_cond
        return global_cond, stream_cond


class CLIPSummaryImageEncoder(nn.Module):
    """Frozen CLIP vision encoder returning one fused feature per camera."""

    def __init__(self, config: DiSPoMamba3FlowClipConfig):
        super().__init__()
        self.config = config
        self.model = CLIPVisionModel.from_pretrained(config.vision_encoder_name)
        self.feature_dim = config.clip_image_feature_dim
        clip_dim = self.model.config.hidden_size
        proj_in_dim = clip_dim * 2 if config.n_obs_steps < 3 else clip_dim * 3
        self.summary_proj = nn.Sequential(nn.LayerNorm(proj_in_dim), nn.Linear(proj_in_dim, self.feature_dim))
        self.resize = (
            torchvision.transforms.Resize(
                config.clip_image_resize_shape,
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                antialias=True,
            )
            if config.clip_image_resize_shape is not None
            else None
        )
        self.center_crop = (
            torchvision.transforms.CenterCrop(config.clip_image_crop_shape)
            if config.clip_image_crop_shape is not None
            else None
        )
        self.maybe_random_crop = (
            torchvision.transforms.RandomCrop(config.clip_image_crop_shape)
            if config.clip_image_crop_shape is not None and config.clip_image_crop_is_random
            else self.center_crop
        )
        self.register_buffer(
            "clip_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "clip_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(1, 3, 1, 1),
        )
        if config.clip_freeze_image_encoder:
            self.model.requires_grad_(False)
            self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.clip_freeze_image_encoder:
            self.model.eval()
        return self

    def forward(self, obs: Tensor) -> Tensor:
        if obs.ndim != 5:
            raise ValueError(f"CLIPSummaryImageEncoder expects (B, S, C, H, W). Got {tuple(obs.shape)}.")
        if obs.shape[2] not in (1, 3, 4) and obs.shape[-1] in (1, 3, 4):
            obs = obs.permute(0, 1, 4, 2, 3).contiguous()

        batch_size, obs_steps, channels, height, width = obs.shape
        del height, width
        pixels = obs.reshape(batch_size * obs_steps, channels, *obs.shape[-2:])
        if pixels.dtype == torch.uint8:
            pixels = pixels.float() / 255.0
        else:
            pixels = pixels.float()
        if pixels.shape[1] == 1:
            pixels = pixels.expand(-1, 3, -1, -1)
        elif pixels.shape[1] == 4:
            pixels = pixels[:, :3]
        pixels = pixels.clamp(0.0, 1.0)
        if self.resize is not None:
            pixels = self.resize(pixels)
        if self.center_crop is not None:
            pixels = self.maybe_random_crop(pixels) if self.training else self.center_crop(pixels)
        pixels = (pixels - self.clip_mean.to(dtype=pixels.dtype)) / self.clip_std.to(dtype=pixels.dtype)

        if self.config.clip_freeze_image_encoder:
            with torch.no_grad():
                hidden = self.model(pixel_values=pixels, output_hidden_states=False).last_hidden_state
        else:
            hidden = self.model(pixel_values=pixels, output_hidden_states=False).last_hidden_state

        cls = hidden[:, 0].reshape(batch_size, obs_steps, -1)
        if obs_steps >= 3:
            fused = torch.cat(
                [cls[:, -1], cls[:, -1] - cls[:, -2], cls[:, -1] - 2 * cls[:, -2] + cls[:, -3]],
                dim=-1,
            )
        elif obs_steps == 2:
            fused = torch.cat([cls[:, -1], cls[:, -1] - cls[:, 0]], dim=-1)
        else:
            fused = torch.cat([cls[:, -1], torch.zeros_like(cls[:, -1])], dim=-1)
        return self.summary_proj(fused)


class CLIPTextConditionEncoder(nn.Module):
    """Frozen CLIP text encoder returning a task-conditioning vector."""

    def __init__(self, config: DiSPoMamba3FlowClipConfig):
        super().__init__()
        self.config = config
        self.text_encoder = CLIPTextModel.from_pretrained(config.text_encoder_name)
        self.feature_dim = config.clip_text_projection_dim
        text_dim = self.text_encoder.config.hidden_size
        self.summary_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, self.feature_dim))
        if config.clip_freeze_text_encoder:
            self.text_encoder.requires_grad_(False)
            self.text_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.clip_freeze_text_encoder:
            self.text_encoder.eval()
        return self

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        input_ids = input_ids.long().reshape(input_ids.shape[0], -1)
        attention_mask = attention_mask.long().reshape(attention_mask.shape[0], -1)
        if self.config.clip_freeze_text_encoder:
            with torch.no_grad():
                outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.summary_proj(outputs.pooler_output)


__all__ = [
    "DiSPoMamba3FlowClipConfig",
    "DiSPoMamba3FlowClipModel",
    "DiSPoMamba3FlowClipPolicy",
]
