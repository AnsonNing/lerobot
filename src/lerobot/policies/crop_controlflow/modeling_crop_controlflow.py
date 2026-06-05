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
from copy import deepcopy
from dataclasses import dataclass

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.streaming_flow.modeling_streaming_flow_v3 import (
    SinusoidalPosEmb,
    StreamingFlowPolicy as CLIPStreamingFlowPolicy,
    linearly_interpolate_trajectory,
    sample_cfm_inputs_and_targets,
)
from lerobot.policies.streaming_flow.modeling_streaming_flow_v5 import (
    CLIPPatchTokenImageEncoder,
    CLIPTokenTextConditionEncoder,
    DirectFrequencyPredictor,
    StepScalingLayer,
    modulate,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from lerobot.utils.import_utils import require_package

from .configuration_crop_controlflow import CropControlFlowConfig


@dataclass
class TokenConditioning:
    raw_cond: Tensor
    normalized_cond: Tensor
    context_tokens: Tensor
    context_mask: Tensor
    image_token_start: int
    image_token_end: int
    num_image_cameras: int
    num_obs_steps: int
    visual_grid_size: int
    crop_feature_start: int = -1
    crop_feature_end: int = -1
    bbox_feature_start: int = -1
    bbox_feature_end: int = -1


class CropControlFlowPolicy(CLIPStreamingFlowPolicy):
    """Streaming Flow v5 with online action-attention crop conditioning."""

    config_class = CropControlFlowConfig
    name = "crop_controlflow"

    def __init__(self, config: CropControlFlowConfig, **kwargs):
        require_package("transformers", extra="multi_task_dit")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.register_buffer("_action_min", torch.empty(0), persistent=True)
        self.register_buffer("_action_max", torch.empty(0), persistent=True)
        self.register_buffer("_ema_step", torch.zeros((), dtype=torch.long), persistent=True)
        self._init_normalization_buffers(kwargs.get("dataset_stats"))
        self.model = CropControlFlowModel(config)
        self.ema_model = deepcopy(self.model) if config.use_ema else None
        if self.ema_model is not None:
            self.ema_model.requires_grad_(False)
        self.reset()

    def get_optim_params(self) -> list[dict]:
        non_vision_params = []
        vision_encoder_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if (
                ("rgb_encoder" in name or "crop_rgb_encoder" in name)
                and ".model." in name
            ):
                vision_encoder_params.append(param)
            else:
                non_vision_params.append(param)

        groups: list[dict] = [{"params": non_vision_params}]
        if vision_encoder_params:
            groups.append(
                {
                    "params": vision_encoder_params,
                    "lr": self.config.optimizer_lr * self.config.vision_encoder_lr_multiplier,
                }
            )
        return groups

    def update(self) -> None:
        if self.ema_model is None:
            return

        with torch.no_grad():
            self._ema_step += 1
            decay = self._get_ema_decay(int(self._ema_step.item()))
            one_minus_decay = 1.0 - decay
            for ema_param, param in zip(
                self.ema_model.parameters(), self.model.parameters(), strict=True
            ):
                if param.requires_grad:
                    ema_param.lerp_(param.detach().to(dtype=ema_param.dtype), one_minus_decay)
                else:
                    ema_param.copy_(param.detach().to(dtype=ema_param.dtype))

            for ema_buffer, buffer in zip(self.ema_model.buffers(), self.model.buffers(), strict=True):
                ema_buffer.copy_(buffer)


class CropControlFlowModel(nn.Module):
    """CLIP token conditioning plus an action-attention local crop refinement pass."""

    def __init__(self, config: CropControlFlowConfig):
        super().__init__()
        self.config = config
        hidden_dim = config.transformer_hidden_dim
        self.visual_grid_size = math.isqrt(config.transformer_visual_tokens_per_frame)
        if self.visual_grid_size * self.visual_grid_size != config.transformer_visual_tokens_per_frame:
            raise ValueError(
                "`transformer_visual_tokens_per_frame` must be a square number for crop heatmap reshaping. "
                f"Got {config.transformer_visual_tokens_per_frame}."
            )
        self.roi_bbox_dim = 9

        global_cond_dim = 0
        if config.image_features:
            num_images = len(config.image_features)
            if config.use_separate_rgb_encoder_per_camera:
                self.rgb_encoder = nn.ModuleList(
                    [CLIPPatchTokenImageEncoder(config) for _ in range(num_images)]
                )
                self.crop_rgb_encoder = CLIPPatchTokenImageEncoder(config)
                image_feature_dim = self.rgb_encoder[0].feature_dim
            else:
                self.rgb_encoder = CLIPPatchTokenImageEncoder(config)
                self.crop_rgb_encoder = None
                image_feature_dim = self.rgb_encoder.feature_dim
            global_cond_dim += num_images * image_feature_dim
            if config.crop_controlflow_enabled:
                global_cond_dim += image_feature_dim
        else:
            self.rgb_encoder = None
            self.crop_rgb_encoder = None

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

        if config.crop_controlflow_enabled:
            bbox_dim = config.crop_controlflow_bbox_feature_dim
            self.bbox_cond_proj = nn.Sequential(
                nn.LayerNorm(self.roi_bbox_dim), nn.Linear(self.roi_bbox_dim, bbox_dim)
            )
            self.bbox_token_proj = nn.Sequential(
                nn.LayerNorm(self.roi_bbox_dim), nn.Linear(self.roi_bbox_dim, hidden_dim)
            )
            global_cond_dim += bbox_dim
        else:
            self.bbox_cond_proj = None
            self.bbox_token_proj = None

        self.context_type_embedding = nn.Parameter(torch.zeros(1, 5, hidden_dim))
        nn.init.normal_(self.context_type_embedding, std=0.02)
        self.global_cond_dim = global_cond_dim
        self.velocity_model = AdaptiveCropControlFlowTransformer(
            config=config,
            global_cond_dim=global_cond_dim,
        )

        if config.compile_model:
            self.velocity_model = torch.compile(self.velocity_model, mode=config.compile_mode)

    def _encode_images(self, images: Tensor) -> tuple[Tensor, Tensor]:
        if self.config.use_separate_rgb_encoder_per_camera:
            encoded = [
                encoder(camera_images)
                for encoder, camera_images in zip(
                    self.rgb_encoder,
                    einops.rearrange(images, "b s n c h w -> n b s c h w"),
                    strict=True,
                )
            ]
            image_feats = torch.cat([features for features, _ in encoded], dim=-1)
            image_tokens = torch.cat([tokens for _, tokens in encoded], dim=1)
        else:
            flat_images = einops.rearrange(images, "b s n c h w -> (b n) s c h w")
            image_feats, image_tokens = self.rgb_encoder(flat_images)
            image_feats = einops.rearrange(image_feats, "(b n) f -> b (n f)", b=images.shape[0])
            image_tokens = einops.rearrange(image_tokens, "(b n) l h -> b (n l) h", b=images.shape[0])
        return image_feats, image_tokens

    def _encode_crop_images(self, crop_images: Tensor) -> tuple[Tensor, Tensor]:
        if self.config.use_separate_rgb_encoder_per_camera:
            crop_feats, crop_tokens = self.crop_rgb_encoder(crop_images[:, :, 0])
        else:
            crop_feats, crop_tokens = self._encode_images(crop_images)
        return crop_feats, crop_tokens

    def _prepare_token_conditioning(
        self,
        batch: dict[str, Tensor],
        crop_images: Tensor | None = None,
        roi_bbox: Tensor | None = None,
    ) -> TokenConditioning:
        cond_feats: list[Tensor] = []
        context_tokens: list[Tensor] = []
        context_masks: list[Tensor] = []
        batch_size = None
        image_token_start = 0
        image_token_end = 0
        num_image_cameras = 0
        num_obs_steps = self.config.n_obs_steps
        crop_feature_start = -1
        crop_feature_end = -1
        bbox_feature_start = -1
        bbox_feature_end = -1

        if self.config.robot_state_feature is not None:
            if OBS_STATE not in batch:
                raise ValueError(f"Missing `{OBS_STATE}` in batch. Available keys: {list(batch)}")
            state = batch[OBS_STATE]
            if state.ndim == 2:
                state = state.unsqueeze(1)
            if state.ndim != 3 or state.shape[1] != self.config.n_obs_steps:
                raise ValueError(
                    f"`{OBS_STATE}` must have shape (B, {self.config.n_obs_steps}, D). "
                    f"Got {tuple(state.shape)}."
                )
            batch_size = state.shape[0]
            cond_feats.append(state.flatten(start_dim=1))
            state_tokens = self.state_token_proj(state.float()) + self.context_type_embedding[:, 2:3]
            context_tokens.append(state_tokens)
            context_masks.append(torch.ones(state.shape[:2], dtype=torch.bool, device=state.device))

        if self.config.image_features:
            if OBS_IMAGES not in batch:
                raise ValueError(f"Missing `{OBS_IMAGES}` in batch. Available keys: {list(batch)}")
            images = batch[OBS_IMAGES]
            if images.ndim == 5:
                images = images.unsqueeze(1)
            if images.ndim != 6:
                raise ValueError(
                    f"`{OBS_IMAGES}` must have shape (B, S, N, C, H, W). Got {tuple(images.shape)}."
                )
            batch_size = images.shape[0]
            num_image_cameras = images.shape[2]
            num_obs_steps = images.shape[1]

            image_feats, image_tokens = self._encode_images(images)
            crop_feats = None
            crop_tokens = None
            if crop_images is not None:
                crop_feats, crop_tokens = self._encode_crop_images(crop_images)
            if self.config.crop_controlflow_enabled:
                crop_feature_start = sum(feat.shape[-1] for feat in cond_feats) + image_feats.shape[-1]
                crop_feature = (
                    crop_feats
                    if crop_feats is not None
                    else torch.zeros_like(image_feats[:, : self.rgb_feature_dim])
                )
                crop_feature_end = crop_feature_start + crop_feature.shape[-1]
                image_feats = torch.cat([image_feats, crop_feature], dim=-1)

            cond_feats.append(image_feats)
            image_token_start = sum(tokens.shape[1] for tokens in context_tokens)
            image_token_end = image_token_start + image_tokens.shape[1]
            context_tokens.append(image_tokens + self.context_type_embedding[:, 0:1])
            context_masks.append(
                torch.ones(image_tokens.shape[:2], dtype=torch.bool, device=image_tokens.device)
            )
            if crop_tokens is not None:
                crop_tokens = crop_tokens + self.context_type_embedding[:, 4:5]
                context_tokens.append(crop_tokens)
                context_masks.append(
                    torch.ones(crop_tokens.shape[:2], dtype=torch.bool, device=crop_tokens.device)
                )

        if self.text_encoder is not None:
            if OBS_LANGUAGE_TOKENS not in batch or OBS_LANGUAGE_ATTENTION_MASK not in batch:
                raise ValueError(
                    "CropControlFlow requires tokenized task text when CLIP text conditioning is enabled."
                )
            text_feats, text_tokens, text_mask = self.text_encoder(
                batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
            )
            batch_size = text_feats.shape[0]
            cond_feats.append(
                text_feats.unsqueeze(1).expand(-1, self.config.n_obs_steps, -1).flatten(start_dim=1)
            )
            context_tokens.append(text_tokens + self.context_type_embedding[:, 1:2])
            context_masks.append(text_mask)

        if self.config.crop_controlflow_enabled:
            if batch_size is None:
                raise ValueError("CropControlFlow could not infer batch size before bbox conditioning.")
            if roi_bbox is None:
                roi_bbox = torch.zeros(
                    (batch_size, self.roi_bbox_dim),
                    device=cond_feats[0].device,
                    dtype=torch.float32,
                )
            else:
                roi_bbox = roi_bbox.to(device=cond_feats[0].device, dtype=torch.float32)
                if roi_bbox.shape[-1] != self.roi_bbox_dim:
                    raise ValueError(
                        f"`roi_bbox` must have shape (B, {self.roi_bbox_dim}). Got {tuple(roi_bbox.shape)}."
                    )
            bbox_feature_start = sum(feat.shape[-1] for feat in cond_feats)
            bbox_feature = self.bbox_cond_proj(roi_bbox)
            bbox_feature_end = bbox_feature_start + bbox_feature.shape[-1]
            cond_feats.append(bbox_feature)
            if self.config.crop_controlflow_use_bbox_token and crop_images is not None:
                bbox_token = self.bbox_token_proj(roi_bbox).unsqueeze(1)
                bbox_token = bbox_token + self.context_type_embedding[:, 3:4]
                context_tokens.append(bbox_token)
                context_masks.append(torch.ones((batch_size, 1), dtype=torch.bool, device=bbox_token.device))

        if not cond_feats or batch_size is None:
            raise ValueError("CropControlFlow received no conditioning features.")

        raw_cond = torch.cat(cond_feats, dim=-1)
        normalized_cond = F.normalize(raw_cond, dim=-1) if raw_cond.shape[-1] > 1 else raw_cond
        return TokenConditioning(
            raw_cond=raw_cond,
            normalized_cond=normalized_cond,
            context_tokens=torch.cat(context_tokens, dim=1),
            context_mask=torch.cat(context_masks, dim=1),
            image_token_start=image_token_start,
            image_token_end=image_token_end,
            num_image_cameras=num_image_cameras,
            num_obs_steps=num_obs_steps,
            visual_grid_size=self.visual_grid_size,
            crop_feature_start=crop_feature_start,
            crop_feature_end=crop_feature_end,
            bbox_feature_start=bbox_feature_start,
            bbox_feature_end=bbox_feature_end,
        )

    def _prepare_crop_conditioning_from_base(
        self,
        base_conditioning: TokenConditioning,
        crop_images: Tensor,
        roi_bbox: Tensor,
    ) -> TokenConditioning:
        if base_conditioning.crop_feature_start < 0 or base_conditioning.bbox_feature_start < 0:
            raise ValueError("Base conditioning does not contain crop/bbox feature slots.")

        crop_feats, crop_tokens = self._encode_crop_images(crop_images)
        roi_bbox = roi_bbox.to(device=base_conditioning.raw_cond.device, dtype=torch.float32)
        if roi_bbox.shape[-1] != self.roi_bbox_dim:
            raise ValueError(
                f"`roi_bbox` must have shape (B, {self.roi_bbox_dim}). Got {tuple(roi_bbox.shape)}."
            )

        raw_cond = base_conditioning.raw_cond.clone()
        raw_cond[:, base_conditioning.crop_feature_start : base_conditioning.crop_feature_end] = crop_feats
        raw_cond[:, base_conditioning.bbox_feature_start : base_conditioning.bbox_feature_end] = (
            self.bbox_cond_proj(roi_bbox)
        )
        normalized_cond = F.normalize(raw_cond, dim=-1) if raw_cond.shape[-1] > 1 else raw_cond

        crop_tokens = crop_tokens + self.context_type_embedding[:, 4:5]
        crop_mask = torch.ones(crop_tokens.shape[:2], dtype=torch.bool, device=crop_tokens.device)
        context_tokens = [base_conditioning.context_tokens, crop_tokens]
        context_masks = [base_conditioning.context_mask, crop_mask]
        if self.config.crop_controlflow_use_bbox_token:
            bbox_token = self.bbox_token_proj(roi_bbox).unsqueeze(1)
            bbox_token = bbox_token + self.context_type_embedding[:, 3:4]
            context_tokens.append(bbox_token)
            context_masks.append(
                torch.ones((roi_bbox.shape[0], 1), dtype=torch.bool, device=bbox_token.device)
            )

        return TokenConditioning(
            raw_cond=raw_cond,
            normalized_cond=normalized_cond,
            context_tokens=torch.cat(context_tokens, dim=1),
            context_mask=torch.cat(context_masks, dim=1),
            image_token_start=base_conditioning.image_token_start,
            image_token_end=base_conditioning.image_token_end,
            num_image_cameras=base_conditioning.num_image_cameras,
            num_obs_steps=base_conditioning.num_obs_steps,
            visual_grid_size=base_conditioning.visual_grid_size,
            crop_feature_start=base_conditioning.crop_feature_start,
            crop_feature_end=base_conditioning.crop_feature_end,
            bbox_feature_start=base_conditioning.bbox_feature_start,
            bbox_feature_end=base_conditioning.bbox_feature_end,
        )

    @property
    def rgb_feature_dim(self) -> int:
        if self.config.use_separate_rgb_encoder_per_camera:
            return self.rgb_encoder[0].feature_dim
        return self.rgb_encoder.feature_dim

    def _predict_frequency(self, raw_cond: Tensor, clamp: bool) -> tuple[Tensor, Tensor, Tensor]:
        if not self.config.sfp_use_adaptive_freq:
            freq = torch.ones(raw_cond.shape[0], device=raw_cond.device, dtype=torch.float32)
            return freq, freq, freq
        raw_freq = self.velocity_model.granularity_predictor(raw_cond.float())
        raw_freq = raw_freq.reshape(raw_cond.shape[0], -1)[:, 0]
        clamped_freq = torch.clamp(raw_freq, min=self.config.sfp_freq_min, max=self.config.sfp_freq_max)
        freq = clamped_freq if clamp else raw_freq
        return raw_freq, freq, clamped_freq

    def _initial_action(
        self,
        batch: dict[str, Tensor],
        init_action: Tensor | None = None,
    ) -> Tensor:
        batch_size = batch[OBS_IMAGES].shape[0]
        device = batch[OBS_IMAGES].device
        dtype = batch[OBS_IMAGES].dtype

        if init_action is not None:
            init_action = init_action.to(device=device, dtype=dtype)
            if init_action.ndim == 2:
                init_action = init_action.unsqueeze(1)
            if init_action.ndim == 3 and init_action.shape[1] > 1:
                init_action = init_action[:, :1]
            if init_action.shape[0] == 1 and batch_size > 1:
                init_action = init_action.expand(batch_size, -1, -1)
            elif init_action.shape[0] != batch_size:
                raise ValueError(
                    "Initial action state batch size does not match observation batch size. "
                    f"Got init_action.shape={tuple(init_action.shape)} and batch_size={batch_size}."
                )
            return init_action[:, :, : self.config.action_feature.shape[0]].contiguous()

        return torch.zeros(
            (batch_size, 1, self.config.action_feature.shape[0]),
            dtype=dtype,
            device=device,
        )

    def _normalize_attention_heatmap(self, heatmap: Tensor) -> Tensor:
        heatmap = heatmap.float().clamp_min(0)
        if self.config.crop_controlflow_attention_normalization == "none":
            return heatmap

        baseline = heatmap.mean(dim=(-2, -1), keepdim=True)
        contrast = (heatmap - baseline).clamp_min(0)
        contrast_mass = contrast.flatten(3).sum(dim=-1, keepdim=True)
        raw_mass = heatmap.flatten(3).sum(dim=-1, keepdim=True).clamp_min(1e-8)
        use_contrast = contrast_mass > self.config.crop_controlflow_attention_contrast_eps * raw_mass
        return torch.where(use_contrast[..., None], contrast, heatmap)

    def _attention_heatmap(
        self,
        cross_attentions: list[Tensor],
        conditioning: TokenConditioning,
    ) -> Tensor:
        if not cross_attentions:
            raise RuntimeError("CropControlFlow requested crop attention, but no cross-attentions were returned.")
        grid = conditioning.visual_grid_size
        tokens_per_frame = grid * grid
        image_token_count = conditioning.image_token_end - conditioning.image_token_start
        expected_tokens = conditioning.num_image_cameras * conditioning.num_obs_steps * tokens_per_frame
        if image_token_count != expected_tokens:
            raise ValueError(
                "Image attention tokens do not match the expected camera/step/grid layout. "
                f"Got {image_token_count} image tokens, expected {expected_tokens} "
                f"({conditioning.num_image_cameras} cameras x {conditioning.num_obs_steps} steps "
                f"x {grid} x {grid})."
            )

        num_layers = min(self.config.crop_controlflow_attention_num_layers, len(cross_attentions))
        layer_heatmaps = []
        for attn in cross_attentions[-num_layers:]:
            image_attn = attn[..., conditioning.image_token_start : conditioning.image_token_end]
            image_attn = image_attn.mean(dim=1).mean(dim=1)
            image_attn = image_attn.reshape(
                image_attn.shape[0],
                conditioning.num_image_cameras,
                conditioning.num_obs_steps,
                grid,
                grid,
            )
            image_attn = self._normalize_attention_heatmap(image_attn)
            layer_heatmaps.append(image_attn)

        heatmap = torch.stack(layer_heatmaps, dim=0).mean(dim=0).clamp_min(0)
        smoothing_kernel = self.config.crop_controlflow_attention_smoothing_kernel
        if smoothing_kernel > 1:
            pad = smoothing_kernel // 2
            batch_size, num_cameras, num_steps, grid_h, grid_w = heatmap.shape
            flat_heatmap = heatmap.reshape(batch_size * num_cameras * num_steps, 1, grid_h, grid_w)
            heatmap = F.avg_pool2d(
                F.pad(flat_heatmap, (pad, pad, pad, pad), mode="replicate"),
                kernel_size=smoothing_kernel,
                stride=1,
            ).reshape(batch_size, num_cameras, num_steps, grid_h, grid_w)
        return heatmap.detach() if self.config.crop_controlflow_detach_attention else heatmap

    def _bbox_from_heatmap(self, heatmap: Tensor, image_height: int, image_width: int) -> Tensor:
        heatmap = heatmap.float().clamp_min(0)
        if heatmap.ndim == 3:
            heatmap = heatmap[:, None, None]
        if heatmap.ndim != 5:
            raise ValueError(
                "`heatmap` must have shape (B, C, S, H, W). "
                f"Got {tuple(heatmap.shape)}."
            )
        batch_size, num_cameras, num_steps, grid_h, grid_w = heatmap.shape
        heatmap_sum = heatmap.flatten(1).sum(dim=1).clamp_min(1e-6)
        heatmap = heatmap / heatmap_sum[:, None, None, None, None]

        # Same spirit as mllms_know: aggregate heatmap mass in a sliding window,
        # then crop around the window with the largest task/action-conditioned mass.
        flat_heatmap = heatmap.reshape(batch_size * num_cameras * num_steps, 1, grid_h, grid_w)
        best_score = torch.full((batch_size,), -float("inf"), device=heatmap.device, dtype=heatmap.dtype)
        best_center_x = torch.full((batch_size,), 0.5, device=heatmap.device, dtype=heatmap.dtype)
        best_center_y = torch.full((batch_size,), 0.5, device=heatmap.device, dtype=heatmap.dtype)
        best_crop_w = torch.ones((batch_size,), device=heatmap.device, dtype=heatmap.dtype)
        best_crop_h = torch.ones((batch_size,), device=heatmap.device, dtype=heatmap.dtype)
        best_camera = torch.zeros((batch_size,), device=heatmap.device, dtype=torch.long)
        best_step = torch.zeros((batch_size,), device=heatmap.device, dtype=torch.long)
        best_scale = torch.full(
            (batch_size,),
            float(self.config.crop_controlflow_crop_size_ratio),
            device=heatmap.device,
            dtype=heatmap.dtype,
        )
        fallback_mask = torch.zeros((batch_size,), device=heatmap.device, dtype=torch.bool)

        for ratio in self.config.crop_controlflow_crop_size_ratios:
            crop_size = ratio * min(image_height, image_width)
            crop_w = min(float(crop_size / image_width), 1.0)
            crop_h = min(float(crop_size / image_height), 1.0)
            block_w = max(1, min(grid_w, int(round(crop_w * grid_w))))
            block_h = max(1, min(grid_h, int(round(crop_h * grid_h))))

            kernel = torch.ones((1, 1, block_h, block_w), device=heatmap.device, dtype=heatmap.dtype)
            sliding = F.conv2d(flat_heatmap, kernel).reshape(batch_size, num_cameras, num_steps, -1)
            candidate_mass, flat_idx = sliding.flatten(1).max(dim=1)
            uniform_window_mass = float(block_h * block_w) / float(num_cameras * num_steps * grid_h * grid_w)
            candidate_score = candidate_mass / max(uniform_window_mass, 1e-8)
            update_mask = candidate_score > best_score

            spatial_positions = sliding.shape[-1]
            camera_idx = torch.div(flat_idx, num_steps * spatial_positions, rounding_mode="floor")
            camera_remainder = flat_idx % (num_steps * spatial_positions)
            step_idx = torch.div(camera_remainder, spatial_positions, rounding_mode="floor")
            spatial_idx = camera_remainder % spatial_positions
            out_w = grid_w - block_w + 1
            top = torch.div(spatial_idx, out_w, rounding_mode="floor").to(dtype=heatmap.dtype)
            left = (spatial_idx % out_w).to(dtype=heatmap.dtype)
            center_x = (left + block_w / 2.0) / grid_w
            center_y = (top + block_h / 2.0) / grid_h

            best_score = torch.where(update_mask, candidate_score, best_score)
            best_center_x = torch.where(update_mask, center_x, best_center_x)
            best_center_y = torch.where(update_mask, center_y, best_center_y)
            best_crop_w = torch.where(update_mask, torch.full_like(best_crop_w, crop_w), best_crop_w)
            best_crop_h = torch.where(update_mask, torch.full_like(best_crop_h, crop_h), best_crop_h)
            best_camera = torch.where(update_mask, camera_idx.long(), best_camera)
            best_step = torch.where(update_mask, step_idx.long(), best_step)
            best_scale = torch.where(update_mask, torch.full_like(best_scale, float(ratio)), best_scale)

        if self.config.crop_controlflow_fallback_to_center_on_low_confidence:
            spatial_sum = heatmap.flatten(3).sum(dim=-1).clamp_min(1e-6)
            spatial_heatmap = heatmap / spatial_sum[..., None, None]
            heatmap_entropy = -(spatial_heatmap * spatial_heatmap.clamp_min(1e-8).log()).flatten(3).sum(dim=3)
            heatmap_entropy = heatmap_entropy / math.log(max(grid_h * grid_w, 2))
            batch_indices = torch.arange(batch_size, device=heatmap.device)
            selected_entropy = heatmap_entropy[batch_indices, best_camera, best_step]
            fallback_mask = (
                (selected_entropy > self.config.crop_controlflow_max_heatmap_entropy)
                | (best_score < self.config.crop_controlflow_min_window_mass_ratio)
            )
            fallback_camera = min(
                max(self.config.crop_controlflow_attention_camera_index, 0), num_cameras - 1
            )
            fallback_step = self.config.crop_controlflow_attention_obs_step
            if fallback_step < 0:
                fallback_step = num_steps + fallback_step
            fallback_step = min(max(fallback_step, 0), num_steps - 1)
            fallback_scale = float(self.config.crop_controlflow_crop_size_ratio)
            fallback_crop_size = fallback_scale * min(image_height, image_width)
            best_center_x = torch.where(
                fallback_mask, torch.full_like(best_center_x, 0.5), best_center_x
            )
            best_center_y = torch.where(
                fallback_mask, torch.full_like(best_center_y, 0.5), best_center_y
            )
            best_crop_w = torch.where(
                fallback_mask,
                torch.full_like(best_crop_w, min(fallback_crop_size / image_width, 1.0)),
                best_crop_w,
            )
            best_crop_h = torch.where(
                fallback_mask,
                torch.full_like(best_crop_h, min(fallback_crop_size / image_height, 1.0)),
                best_crop_h,
            )
            best_camera = torch.where(
                fallback_mask, torch.full_like(best_camera, fallback_camera), best_camera
            )
            best_step = torch.where(
                fallback_mask, torch.full_like(best_step, fallback_step), best_step
            )
            best_scale = torch.where(fallback_mask, torch.full_like(best_scale, fallback_scale), best_scale)

        half_w = best_crop_w / 2.0
        half_h = best_crop_h / 2.0
        best_center_x = torch.where(
            half_w < 0.5,
            best_center_x.clamp(half_w, 1.0 - half_w),
            torch.full_like(best_center_x, 0.5),
        )
        best_center_y = torch.where(
            half_h < 0.5,
            best_center_y.clamp(half_h, 1.0 - half_h),
            torch.full_like(best_center_y, 0.5),
        )

        x1 = (best_center_x - half_w).clamp(0.0, 1.0)
        y1 = (best_center_y - half_h).clamp(0.0, 1.0)
        x2 = (best_center_x + half_w).clamp(0.0, 1.0)
        y2 = (best_center_y + half_h).clamp(0.0, 1.0)
        score = best_score
        if self.config.crop_controlflow_fallback_to_center_on_low_confidence:
            score = torch.where(fallback_mask, torch.zeros_like(score), score)
        score = torch.log1p(score.clamp_min(0.0)).clamp_max(
            self.config.crop_controlflow_score_log_max
        ) / self.config.crop_controlflow_score_log_max
        camera_norm = best_camera.to(dtype=heatmap.dtype) / max(num_cameras - 1, 1)
        step_norm = best_step.to(dtype=heatmap.dtype) / max(num_steps - 1, 1)
        fallback_flag = fallback_mask.to(dtype=heatmap.dtype)
        return torch.stack(
            [x1, y1, x2, y2, score, camera_norm, step_norm, best_scale, fallback_flag],
            dim=1,
        )

    def _crop_images(self, images: Tensor, bbox: Tensor) -> Tensor:
        if images.ndim == 5:
            images = images.unsqueeze(1)
        batch_size, obs_steps, num_cameras, channels, height, width = images.shape
        if bbox.shape[-1] >= 6:
            camera_idx = (bbox[:, 5] * max(num_cameras - 1, 1)).round().long()
            camera_idx = camera_idx.clamp(0, num_cameras - 1)
        else:
            fallback_idx = min(max(self.config.crop_controlflow_attention_camera_index, 0), num_cameras - 1)
            camera_idx = torch.full((batch_size,), fallback_idx, device=images.device, dtype=torch.long)
        camera_idx = camera_idx.to(device=images.device)
        gather_index = camera_idx[:, None, None, None, None, None].expand(
            -1, obs_steps, 1, channels, height, width
        )
        source = images.gather(2, gather_index).squeeze(2)
        flat_source = source.reshape(batch_size * obs_steps, channels, height, width)
        repeated_bbox = bbox[:, :4].repeat_interleave(obs_steps, dim=0)

        x1, y1, x2, y2 = repeated_bbox.unbind(dim=1)
        theta = torch.zeros((batch_size * obs_steps, 2, 3), device=images.device, dtype=images.dtype)
        theta[:, 0, 0] = (x2 - x1).to(dtype=images.dtype)
        theta[:, 1, 1] = (y2 - y1).to(dtype=images.dtype)
        theta[:, 0, 2] = (x1 + x2 - 1.0).to(dtype=images.dtype)
        theta[:, 1, 2] = (y1 + y2 - 1.0).to(dtype=images.dtype)
        grid = F.affine_grid(theta, size=flat_source.shape, align_corners=False)
        crop = F.grid_sample(flat_source, grid, mode="bilinear", padding_mode="border", align_corners=False)
        return crop.reshape(batch_size, obs_steps, 1, channels, height, width)

    def _velocity_norm(self, velocity: Tensor) -> Tensor:
        return velocity.reshape(velocity.shape[0], -1).norm(dim=1)

    def _velocity_cosine(self, first: Tensor, second: Tensor) -> Tensor:
        return F.cosine_similarity(first.reshape(first.shape[0], -1), second.reshape(second.shape[0], -1), dim=1)

    def _use_crop_during_training(self) -> bool:
        return self.config.crop_controlflow_use_crop_during_training

    def _global_loss_weight(self) -> float:
        return float(self.config.crop_controlflow_global_loss_weight)

    def _extra_loss_logs(self) -> dict[str, float]:
        return {}

    def _build_crop_conditioning(
        self,
        batch: dict[str, Tensor],
        action: Tensor,
        timestep: Tensor,
        base_conditioning: TokenConditioning,
        base_freq: Tensor,
    ) -> tuple[TokenConditioning, Tensor, Tensor, Tensor]:
        base_velocity, cross_attentions = self.velocity_model(
            sample=action.float(),
            timestep=timestep,
            global_cond=base_conditioning.normalized_cond,
            context_tokens=base_conditioning.context_tokens,
            context_mask=base_conditioning.context_mask,
            freq=base_freq,
            return_cross_attention=True,
        )
        heatmap = self._attention_heatmap(cross_attentions, base_conditioning)
        images = batch[OBS_IMAGES]
        if images.ndim == 5:
            images = images.unsqueeze(1)
        bbox = self._bbox_from_heatmap(heatmap, image_height=images.shape[-2], image_width=images.shape[-1])
        bbox_for_cond = bbox.detach()
        with torch.no_grad():
            crop_images = self._crop_images(images, bbox_for_cond)
        return (
            self._prepare_crop_conditioning_from_base(base_conditioning, crop_images, bbox_for_cond),
            bbox_for_cond,
            heatmap,
            base_velocity.detach(),
        )

    def integrate_actions(
        self,
        batch: dict[str, Tensor],
        init_action: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, float]]:
        base_conditioning = self._prepare_token_conditioning(batch)
        raw_freq, freq, clamped_freq = self._predict_frequency(
            base_conditioning.raw_cond,
            clamp=self.config.sfp_clamp_freq_during_eval,
        )
        base_freq_for_log = freq
        action = self._initial_action(batch, init_action=init_action)
        timestep0 = torch.zeros((action.shape[0],), device=action.device, dtype=torch.float32)

        bbox = None
        base_first_velocity = None
        if self.config.crop_controlflow_enabled and self.config.crop_controlflow_use_crop_during_eval:
            conditioning, bbox, _, base_first_velocity = self._build_crop_conditioning(
                batch, action, timestep0, base_conditioning, freq
            )
            raw_freq, freq, clamped_freq = self._predict_frequency(
                conditioning.raw_cond,
                clamp=self.config.sfp_clamp_freq_during_eval,
            )
        else:
            conditioning = base_conditioning

        dt = 1.0 / max(self.config.chunk_size - self.config.n_obs_steps, 1)
        action_chunk = []
        initial_action = action.detach()
        first_velocity_mean_abs = 0.0
        crop_first_velocity = None

        for step_idx in range(self.config.n_action_steps):
            timestep = torch.full(
                (action.shape[0],), step_idx * dt, device=action.device, dtype=torch.float32
            )
            velocity = self.velocity_model(
                sample=action.float(),
                timestep=timestep,
                global_cond=conditioning.normalized_cond,
                context_tokens=conditioning.context_tokens,
                context_mask=conditioning.context_mask,
                freq=freq,
            )
            if step_idx == 0:
                first_velocity_mean_abs = float(velocity.abs().mean().detach().cpu())
                crop_first_velocity = velocity.detach()
            action = action + velocity * dt
            action_chunk.append(action.squeeze(1))

        stacked_chunk = torch.stack(action_chunk, dim=1)
        final_action = action.detach()
        info = {
            "pred_freq": float(freq.mean().detach().cpu()),
            "raw_pred_freq": float(raw_freq.mean().detach().cpu()),
            "clamped_pred_freq": float(clamped_freq.mean().detach().cpu()),
            "base_pred_freq": float(base_freq_for_log.mean().detach().cpu()),
            "crop_pred_freq": float(freq.mean().detach().cpu()),
            "delta_freq_abs": float((freq - base_freq_for_log).abs().mean().detach().cpu()),
            "delta_freq_rel": float(
                ((freq - base_freq_for_log).abs() / base_freq_for_log.abs().clamp_min(1e-6))
                .mean()
                .detach()
                .cpu()
            ),
            "dt": float(dt),
            "init_action_mean_abs": float(initial_action.abs().mean().detach().cpu()),
            "final_action_mean_abs": float(final_action.abs().mean().detach().cpu()),
            "chunk_delta_mean_abs": float(
                (stacked_chunk[:, 1:] - stacked_chunk[:, :-1]).abs().mean().detach().cpu()
            )
            if stacked_chunk.shape[1] > 1
            else 0.0,
            "first_velocity_mean_abs": first_velocity_mean_abs,
        }
        if base_first_velocity is not None and crop_first_velocity is not None:
            info["base_first_velocity_norm"] = float(self._velocity_norm(base_first_velocity).mean().cpu())
            info["crop_first_velocity_norm"] = float(self._velocity_norm(crop_first_velocity).mean().cpu())
            info["delta_velocity_norm"] = float(
                self._velocity_norm(crop_first_velocity - base_first_velocity).mean().cpu()
            )
            info["velocity_cosine"] = float(
                self._velocity_cosine(base_first_velocity, crop_first_velocity).mean().cpu()
            )
        elif crop_first_velocity is not None:
            velocity_norm = float(self._velocity_norm(crop_first_velocity).mean().cpu())
            info["base_first_velocity_norm"] = velocity_norm
            info["crop_first_velocity_norm"] = velocity_norm
            info["delta_velocity_norm"] = 0.0
            info["velocity_cosine"] = 1.0
        if bbox is not None:
            info["roi_bbox_mean"] = float(bbox[:, :4].mean().detach().cpu())
            info["roi_score_mean"] = float(bbox[:, 4].mean().detach().cpu())
            info["roi_camera_mean"] = float(bbox[:, 5].mean().detach().cpu())
            info["roi_step_mean"] = float(bbox[:, 6].mean().detach().cpu())
            info["roi_scale_mean"] = float(bbox[:, 7].mean().detach().cpu())
            info["fallback_ratio"] = float(bbox[:, 8].mean().detach().cpu())
        return stacked_chunk, final_action, info

    def compute_loss(
        self,
        batch: dict[str, Tensor],
        reduction: str = "mean",
    ) -> tuple[Tensor, dict[str, float] | None]:
        if ACTION not in batch:
            raise ValueError(f"Missing `{ACTION}` in batch. Available keys: {list(batch)}")

        base_conditioning = self._prepare_token_conditioning(batch)
        base_raw_freq, base_freq, base_clamped_freq = self._predict_frequency(
            base_conditioning.raw_cond,
            clamp=self.config.sfp_clamp_freq_during_training,
        )
        start_idx = min(max(self.config.n_obs_steps - 1, 0), batch[ACTION].shape[1] - 1)
        trajectory = batch[ACTION][:, start_idx:, :]
        num_queries = max(1, int(self.config.sfp_num_train_points))
        time_shape = (trajectory.shape[0], num_queries) if num_queries > 1 else (trajectory.shape[0],)
        time = torch.rand(time_shape, device=trajectory.device, dtype=torch.float32) * 0.999 + 0.001
        xi_t, dxi_dt = linearly_interpolate_trajectory(trajectory, time)
        noised_action, target_velocity = sample_cfm_inputs_and_targets(
            xi_t, dxi_dt, time, k=self.config.sfp_k, sigma0=self.config.sfp_sigma0
        )

        crop_first_velocity = None
        use_crop_during_training = self.config.crop_controlflow_enabled and self._use_crop_during_training()
        global_loss_weight = self._global_loss_weight()

        if use_crop_during_training:
            rough_action = self._initial_action(batch)
            rough_time = torch.zeros((rough_action.shape[0],), device=rough_action.device, dtype=torch.float32)
            conditioning, bbox, _, base_first_velocity = self._build_crop_conditioning(
                batch, rough_action, rough_time, base_conditioning, base_freq
            )
            raw_freq, freq, clamped_freq = self._predict_frequency(
                conditioning.raw_cond,
                clamp=self.config.sfp_clamp_freq_during_training,
            )
            with torch.no_grad():
                crop_first_velocity = self.velocity_model(
                    sample=rough_action.float(),
                    timestep=rough_time,
                    global_cond=conditioning.normalized_cond,
                    context_tokens=conditioning.context_tokens,
                    context_mask=conditioning.context_mask,
                    freq=freq,
                )
        else:
            conditioning = base_conditioning
            bbox = None
            base_first_velocity = None
            raw_freq, freq, clamped_freq = base_raw_freq, base_freq, base_clamped_freq

        if num_queries > 1:
            batch_size, _, action_dim = noised_action.shape
            flat_noised_action = noised_action.reshape(batch_size * num_queries, action_dim)
            flat_target_velocity = target_velocity.reshape(batch_size * num_queries, action_dim)
            flat_time = time.reshape(batch_size * num_queries)
            flat_cond = conditioning.normalized_cond.repeat_interleave(num_queries, dim=0)
            flat_freq = freq.repeat_interleave(num_queries, dim=0)
            flat_tokens = conditioning.context_tokens.repeat_interleave(num_queries, dim=0)
            flat_mask = conditioning.context_mask.repeat_interleave(num_queries, dim=0)
        else:
            flat_noised_action = noised_action
            flat_target_velocity = target_velocity
            flat_time = time
            flat_cond = conditioning.normalized_cond
            flat_freq = freq
            flat_tokens = conditioning.context_tokens
            flat_mask = conditioning.context_mask

        pred_velocity = self.velocity_model(
            sample=flat_noised_action.unsqueeze(1),
            timestep=flat_time,
            global_cond=flat_cond,
            context_tokens=flat_tokens,
            context_mask=flat_mask,
            freq=flat_freq,
        )
        per_sample_loss = F.mse_loss(
            pred_velocity, flat_target_velocity.unsqueeze(1), reduction="none"
        ).mean(dim=(1, 2)).reshape(trajectory.shape[0], num_queries).mean(dim=1)
        crop_loss_for_log = per_sample_loss.detach()

        base_pred_velocity = None
        base_loss_for_log = None
        if global_loss_weight > 0.0:
            if num_queries > 1:
                flat_base_cond = base_conditioning.normalized_cond.repeat_interleave(num_queries, dim=0)
                flat_base_freq = base_freq.repeat_interleave(num_queries, dim=0)
                flat_base_tokens = base_conditioning.context_tokens.repeat_interleave(num_queries, dim=0)
                flat_base_mask = base_conditioning.context_mask.repeat_interleave(num_queries, dim=0)
            else:
                flat_base_cond = base_conditioning.normalized_cond
                flat_base_freq = base_freq
                flat_base_tokens = base_conditioning.context_tokens
                flat_base_mask = base_conditioning.context_mask
            base_pred_velocity = self.velocity_model(
                sample=flat_noised_action.unsqueeze(1),
                timestep=flat_time,
                global_cond=flat_base_cond,
                context_tokens=flat_base_tokens,
                context_mask=flat_base_mask,
                freq=flat_base_freq,
            )
            base_loss = F.mse_loss(
                base_pred_velocity, flat_target_velocity.unsqueeze(1), reduction="none"
            ).mean(dim=(1, 2)).reshape(trajectory.shape[0], num_queries).mean(dim=1)
            base_loss_for_log = base_loss.detach()
            per_sample_loss = per_sample_loss + global_loss_weight * base_loss
        elif bbox is not None:
            if num_queries > 1:
                flat_base_cond = base_conditioning.normalized_cond.repeat_interleave(num_queries, dim=0)
                flat_base_freq = base_freq.repeat_interleave(num_queries, dim=0)
                flat_base_tokens = base_conditioning.context_tokens.repeat_interleave(num_queries, dim=0)
                flat_base_mask = base_conditioning.context_mask.repeat_interleave(num_queries, dim=0)
            else:
                flat_base_cond = base_conditioning.normalized_cond
                flat_base_freq = base_freq
                flat_base_tokens = base_conditioning.context_tokens
                flat_base_mask = base_conditioning.context_mask
            with torch.no_grad():
                base_pred_velocity = self.velocity_model(
                    sample=flat_noised_action.unsqueeze(1),
                    timestep=flat_time,
                    global_cond=flat_base_cond,
                    context_tokens=flat_base_tokens,
                    context_mask=flat_base_mask,
                    freq=flat_base_freq,
                )
                base_loss_for_log = F.mse_loss(
                    base_pred_velocity,
                    flat_target_velocity.unsqueeze(1),
                    reduction="none",
                ).mean(dim=(1, 2)).reshape(trajectory.shape[0], num_queries).mean(dim=1)

        if self.config.sfp_freq_reg_weight > 0.0:
            per_sample_loss = per_sample_loss + self.config.sfp_freq_reg_weight * (freq - 1.0).pow(2)
        if self.config.do_mask_loss_for_padding and "action_is_pad" in batch:
            valid_mask = (~batch["action_is_pad"][:, start_idx:]).all(dim=1)
            per_sample_loss = per_sample_loss * valid_mask.to(dtype=per_sample_loss.dtype)
            mean_loss = per_sample_loss.sum() / valid_mask.sum().clamp_min(1)
        else:
            mean_loss = per_sample_loss.mean()
        output_dict = {
            "loss": float(mean_loss.detach().item()),
            "crop_loss": float(crop_loss_for_log.mean().detach().item()),
            "pred_freq": float(freq.mean().detach().item()),
            "raw_pred_freq": float(raw_freq.mean().detach().item()),
            "clamped_pred_freq": float(clamped_freq.mean().detach().item()),
            "rough_pred_freq": float(base_freq.mean().detach().item()),
            "base_pred_freq": float(base_freq.mean().detach().item()),
            "crop_pred_freq": float(freq.mean().detach().item()),
            "delta_freq_abs": float((freq - base_freq).abs().mean().detach().item()),
            "delta_freq_rel": float(
                ((freq - base_freq).abs() / base_freq.abs().clamp_min(1e-6)).mean().detach().item()
            ),
            "sfp_num_train_points": float(num_queries),
            "crop_controlflow_global_loss_weight": float(global_loss_weight),
            "crop_controlflow_train_crop_enabled": float(use_crop_during_training),
        }
        if base_loss_for_log is not None:
            output_dict["global_loss"] = float(base_loss_for_log.mean().detach().item())
            output_dict["global_minus_crop_loss"] = float(
                (base_loss_for_log - crop_loss_for_log).mean().detach().item()
            )
        output_dict.update(self._extra_loss_logs())
        if base_first_velocity is not None and crop_first_velocity is not None:
            output_dict["base_first_velocity_norm"] = float(
                self._velocity_norm(base_first_velocity).mean().detach().item()
            )
            output_dict["crop_first_velocity_norm"] = float(
                self._velocity_norm(crop_first_velocity).mean().detach().item()
            )
            output_dict["delta_velocity_norm"] = float(
                self._velocity_norm(crop_first_velocity - base_first_velocity).mean().detach().item()
            )
            output_dict["velocity_cosine"] = float(
                self._velocity_cosine(base_first_velocity, crop_first_velocity).mean().detach().item()
            )
        elif base_pred_velocity is not None:
            velocity_norm = float(self._velocity_norm(base_pred_velocity.detach()).mean().detach().item())
            output_dict["base_first_velocity_norm"] = velocity_norm
            output_dict["crop_first_velocity_norm"] = velocity_norm
            output_dict["delta_velocity_norm"] = 0.0
            output_dict["velocity_cosine"] = 1.0
        if bbox is not None:
            output_dict["roi_bbox_mean"] = float(bbox[:, :4].mean().detach().item())
            output_dict["roi_score_mean"] = float(bbox[:, 4].mean().detach().item())
            output_dict["roi_camera_mean"] = float(bbox[:, 5].mean().detach().item())
            output_dict["roi_step_mean"] = float(bbox[:, 6].mean().detach().item())
            output_dict["roi_scale_mean"] = float(bbox[:, 7].mean().detach().item())
            output_dict["fallback_ratio"] = float(bbox[:, 8].mean().detach().item())
        return (per_sample_loss, output_dict) if reduction == "none" else (mean_loss, output_dict)


class AdaLNZeroCropCrossAttentionBlock(nn.Module):
    """Transformer decoder block that can expose action-to-memory cross-attention."""

    def __init__(self, config: CropControlFlowConfig):
        super().__init__()
        hidden_dim = config.transformer_hidden_dim
        self.norm_self = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm_cross = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm_mlp = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, config.transformer_num_heads, dropout=config.transformer_dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, config.transformer_num_heads, dropout=config.transformer_dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, config.transformer_ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(config.transformer_dropout),
            nn.Linear(config.transformer_ffn_dim, hidden_dim),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 9 * hidden_dim))
        self.dropout = nn.Dropout(config.transformer_dropout)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        memory_padding_mask: Tensor,
        condition: Tensor,
        delta: Tensor,
        need_cross_attention: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        modulation = self.adaLN_modulation(condition).chunk(9, dim=-1)
        (
            shift_self,
            scale_self,
            gate_self,
            shift_cross,
            scale_cross,
            gate_cross,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = modulation
        residual_scale = delta[:, None, None]
        self_input = modulate(self.norm_self(x), shift_self[:, None], scale_self[:, None])
        self_out, _ = self.self_attn(self_input, self_input, self_input, need_weights=False)
        x = x + residual_scale * gate_self[:, None] * self.dropout(self_out)
        cross_input = modulate(self.norm_cross(x), shift_cross[:, None], scale_cross[:, None])
        cross_out, cross_weights = self.cross_attn(
            cross_input,
            memory,
            memory,
            key_padding_mask=memory_padding_mask,
            need_weights=need_cross_attention,
            average_attn_weights=False,
        )
        x = x + residual_scale * gate_cross[:, None] * self.dropout(cross_out)
        mlp_input = modulate(self.norm_mlp(x), shift_mlp[:, None], scale_mlp[:, None])
        x = x + residual_scale * gate_mlp[:, None] * self.dropout(self.mlp(mlp_input))
        return x, cross_weights


class AdaptiveCropControlFlowTransformer(nn.Module):
    """Token-grounded velocity expert with optional cross-attention attribution."""

    def __init__(self, config: CropControlFlowConfig, global_cond_dim: int):
        super().__init__()
        self.config = config
        action_dim = config.action_feature.shape[0]
        hidden_dim = config.transformer_hidden_dim
        self.action_in_proj = nn.Linear(action_dim, hidden_dim)
        self.action_out_proj = nn.Linear(hidden_dim, action_dim)
        self.pooled_context_proj = nn.Sequential(
            nn.LayerNorm(global_cond_dim),
            nn.Linear(global_cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(config.embedding_dim, scale=config.timestep_embedding_scale),
            nn.Linear(config.embedding_dim, config.embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(config.embedding_dim * 4, hidden_dim),
        )
        self.freq_encoder = nn.Sequential(
            SinusoidalPosEmb(config.embedding_dim, scale=config.frequency_embedding_scale),
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.SiLU(),
            nn.Linear(config.embedding_dim, hidden_dim),
        )
        self.action_condition_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.modulation_condition = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.context_type_embedding = nn.Parameter(torch.zeros(1, 3, hidden_dim))
        self.blocks = nn.ModuleList(
            [AdaLNZeroCropCrossAttentionBlock(config) for _ in range(config.transformer_num_layers)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.granularity_predictor = DirectFrequencyPredictor(global_cond_dim, config)
        self.step_scaling = StepScalingLayer(hidden_dim * 2)
        nn.init.normal_(self.context_type_embedding, std=0.02)

    def forward(
        self,
        sample: Tensor,
        timestep: Tensor,
        global_cond: Tensor,
        context_tokens: Tensor,
        context_mask: Tensor,
        freq: Tensor | None = None,
        smooth_freq: bool = False,
        return_cross_attention: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        del smooth_freq
        if sample.ndim == 2:
            sample = sample.unsqueeze(1)
        batch_size, query_len, _ = sample.shape
        timestep = torch.as_tensor(timestep, dtype=torch.float32, device=sample.device).view(-1)
        if timestep.numel() == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)
        if freq is None:
            freq = self.granularity_predictor(global_cond)
        freq = torch.as_tensor(freq, dtype=torch.float32, device=sample.device).reshape(-1)
        if freq.numel() == 1 and batch_size > 1:
            freq = freq.expand(batch_size)
        elif freq.numel() != batch_size:
            freq = freq.reshape(batch_size, -1)[:, 0]
        freq = freq.to(device=sample.device, dtype=torch.float32)

        pooled_context = self.pooled_context_proj(global_cond.float())
        time_token = self.diffusion_step_encoder(timestep)
        freq_token = self.freq_encoder(freq)
        action_token = self.action_in_proj(sample.float())
        hidden = self.action_condition_proj(
            torch.cat(
                [
                    action_token,
                    time_token[:, None].expand(-1, query_len, -1),
                    freq_token[:, None].expand(-1, query_len, -1),
                ],
                dim=-1,
            )
        )
        condition = self.modulation_condition(torch.cat([pooled_context, time_token, freq_token], dim=-1))
        global_memory = torch.stack([pooled_context, time_token, freq_token], dim=1)
        global_memory = global_memory + self.context_type_embedding.to(dtype=global_memory.dtype)
        memory = torch.cat([context_tokens.to(dtype=global_memory.dtype), global_memory], dim=1)
        valid_global = torch.ones((batch_size, 3), dtype=torch.bool, device=sample.device)
        memory_padding_mask = ~torch.cat([context_mask.bool(), valid_global], dim=1)
        delta = self.step_scaling(torch.cat([time_token, freq_token], dim=-1), freq)
        cross_attentions: list[Tensor] = []
        num_attention_layers = min(self.config.crop_controlflow_attention_num_layers, len(self.blocks))
        first_attention_layer = len(self.blocks) - num_attention_layers
        for block_idx, block in enumerate(self.blocks):
            need_cross_attention = return_cross_attention and block_idx >= first_attention_layer
            hidden, cross_weights = block(
                hidden,
                memory,
                memory_padding_mask,
                condition,
                delta,
                need_cross_attention=need_cross_attention,
            )
            if need_cross_attention and cross_weights is not None:
                cross_attentions.append(cross_weights)
        output = self.action_out_proj(self.output_norm(hidden))
        return (output, cross_attentions) if return_cross_attention else output
