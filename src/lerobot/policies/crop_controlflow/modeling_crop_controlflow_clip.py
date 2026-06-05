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

from copy import deepcopy

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.streaming_flow.modeling_streaming_flow_v3 import (
    StreamingFlowPolicy as CLIPStreamingFlowPolicy,
)
from lerobot.utils.constants import OBS_IMAGES
from lerobot.utils.import_utils import require_package

from .configuration_crop_controlflow import CropControlFlowClipConfig
from .modeling_crop_controlflow import CropControlFlowModel, TokenConditioning


class CropControlFlowClipPolicy(CLIPStreamingFlowPolicy):
    """CropControlFlow with a learned CLIP image/text ROI selector."""

    config_class = CropControlFlowClipConfig
    name = "crop_controlflow_clip"

    def __init__(self, config: CropControlFlowClipConfig, **kwargs):
        require_package("transformers", extra="multi_task_dit")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.register_buffer("_action_min", torch.empty(0), persistent=True)
        self.register_buffer("_action_max", torch.empty(0), persistent=True)
        self.register_buffer("_ema_step", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer("_crop_clip_train_step", torch.zeros((), dtype=torch.long), persistent=True)
        self._init_normalization_buffers(kwargs.get("dataset_stats"))
        self.model = CropControlFlowClipModel(config)
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

    def _schedule_state(self, step: int) -> tuple[str, bool, float]:
        if not self.config.crop_controlflow_clip_schedule_enabled:
            return (
                "joint",
                self.config.crop_controlflow_use_crop_during_training,
                float(self.config.crop_controlflow_global_loss_weight),
            )

        stage1_steps = self.config.crop_controlflow_clip_stage1_global_steps
        stage2_steps = self.config.crop_controlflow_clip_stage2_warmup_steps
        if step < stage1_steps:
            return "global", False, 0.0
        if step < stage1_steps + stage2_steps:
            return (
                "roi_warmup",
                self.config.crop_controlflow_use_crop_during_training,
                float(self.config.crop_controlflow_clip_stage2_global_loss_weight),
            )
        return (
            "joint",
            self.config.crop_controlflow_use_crop_during_training,
            float(self.config.crop_controlflow_clip_stage3_global_loss_weight),
        )

    def forward(
        self,
        batch: dict[str, Tensor],
        reduction: str = "mean",
    ) -> tuple[Tensor, dict[str, float] | None]:
        batch = self._prepare_image_batch(batch)
        if self.training:
            train_step = int(self._crop_clip_train_step.item())
            stage, crop_enabled, global_loss_weight = self._schedule_state(train_step)
            self.model.set_training_schedule(
                stage=stage,
                step=train_step,
                crop_enabled=crop_enabled,
                global_loss_weight=global_loss_weight,
            )
            self.model.configure_trainable_params(stage)
            loss, output_dict = self.model.compute_loss(batch, reduction=reduction)
            return loss, output_dict

        self.model.set_training_schedule(
            stage="eval",
            step=int(self._crop_clip_train_step.item()),
            crop_enabled=self.config.crop_controlflow_use_crop_during_eval,
            global_loss_weight=float(self.config.crop_controlflow_clip_stage3_global_loss_weight),
        )
        self.model.configure_trainable_params("joint")
        return self.model.compute_loss(batch, reduction=reduction)

    def update(self) -> None:
        self._crop_clip_train_step += 1
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


class CLIPLearnedROIHead(nn.Module):
    """Predict a differentiable ROI from CLIP patch tokens plus policy context."""

    def __init__(self, config: CropControlFlowClipConfig, global_cond_dim: int):
        super().__init__()
        hidden_dim = config.transformer_hidden_dim
        roi_hidden_dim = config.crop_controlflow_clip_roi_hidden_dim
        action_dim = config.action_feature.shape[0]
        self.config = config
        self.token_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, roi_hidden_dim),
        )
        self.global_cond_proj = nn.Sequential(
            nn.LayerNorm(global_cond_dim),
            nn.Linear(global_cond_dim, roi_hidden_dim),
        )
        self.action_proj = nn.Linear(action_dim, roi_hidden_dim)
        self.time_proj = nn.Linear(1, roi_hidden_dim)
        self.freq_proj = nn.Linear(1, roi_hidden_dim)
        self.score_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(roi_hidden_dim, roi_hidden_dim),
            nn.SiLU(),
            nn.Linear(roi_hidden_dim, 1),
        )
        self.size_head = nn.Sequential(
            nn.LayerNorm(roi_hidden_dim),
            nn.Linear(roi_hidden_dim, roi_hidden_dim),
            nn.SiLU(),
            nn.Linear(roi_hidden_dim, 2),
        )
        self._init_size_bias()

    def _init_size_bias(self) -> None:
        min_ratio = self.config.crop_controlflow_clip_roi_min_crop_ratio
        max_ratio = self.config.crop_controlflow_clip_roi_max_crop_ratio
        init_ratio = self.config.crop_controlflow_clip_roi_init_crop_ratio
        init_unit = (init_ratio - min_ratio) / max(max_ratio - min_ratio, 1e-6)
        init_unit = min(max(init_unit, 1e-4), 1.0 - 1e-4)
        init_logit = torch.logit(torch.tensor(init_unit)).item()
        last = self.size_head[-1]
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, init_logit)

    def forward(
        self,
        image_tokens: Tensor,
        global_cond: Tensor,
        rough_action: Tensor,
        timestep: Tensor,
        freq: Tensor,
        num_cameras: int,
        num_obs_steps: int,
        grid: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = image_tokens.shape[0]
        if rough_action.ndim == 3:
            rough_action = rough_action[:, 0]
        rough_action = rough_action.reshape(batch_size, -1)
        timestep = torch.as_tensor(timestep, device=image_tokens.device, dtype=torch.float32).reshape(-1, 1)
        if timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size, 1)
        freq = torch.as_tensor(freq, device=image_tokens.device, dtype=torch.float32).reshape(-1, 1)
        if freq.shape[0] == 1 and batch_size > 1:
            freq = freq.expand(batch_size, 1)

        selector_tokens = image_tokens
        selector_cond = global_cond
        if self.config.crop_controlflow_clip_roi_detach_clip_features:
            selector_tokens = selector_tokens.detach()
            selector_cond = selector_cond.detach()

        cond = (
            self.global_cond_proj(selector_cond.float())
            + self.action_proj(rough_action.float())
            + self.time_proj(timestep)
            + self.freq_proj(freq)
        )
        token_hidden = self.token_proj(selector_tokens.float())
        logits = self.score_proj(token_hidden + cond[:, None]).squeeze(-1)
        logits = logits / self.config.crop_controlflow_clip_roi_temperature
        logits = logits.reshape(batch_size, num_cameras, num_obs_steps, grid, grid)
        heatmap_flat = F.softmax(logits.flatten(3).float(), dim=-1)
        heatmap = heatmap_flat.reshape(batch_size, num_cameras, num_obs_steps, grid, grid)
        map_score = logits.float().flatten(3).amax(dim=-1)
        camera_step_prob = F.softmax(map_score.flatten(1), dim=1).reshape(
            batch_size,
            num_cameras,
            num_obs_steps,
        )
        spatial_heatmap = (camera_step_prob[..., None, None] * heatmap).sum(dim=(1, 2))
        spatial_heatmap = spatial_heatmap / spatial_heatmap.flatten(1).sum(dim=1, keepdim=True).clamp_min(
            1e-8
        )[:, None]

        x_coords = torch.linspace(
            0.5 / grid,
            1.0 - 0.5 / grid,
            grid,
            device=image_tokens.device,
            dtype=heatmap.dtype,
        )
        y_coords = torch.linspace(
            0.5 / grid,
            1.0 - 0.5 / grid,
            grid,
            device=image_tokens.device,
            dtype=heatmap.dtype,
        )
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        center_x = (spatial_heatmap * xx[None]).sum(dim=(1, 2))
        center_y = (spatial_heatmap * yy[None]).sum(dim=(1, 2))

        full_heatmap_flat = (camera_step_prob[..., None, None] * heatmap).flatten(1)
        full_heatmap_flat = full_heatmap_flat / full_heatmap_flat.sum(dim=1, keepdim=True).clamp_min(1e-8)
        roi_summary = (full_heatmap_flat[:, :, None] * token_hidden).sum(dim=1)
        size_unit = torch.sigmoid(self.size_head(roi_summary + cond))
        min_ratio = self.config.crop_controlflow_clip_roi_min_crop_ratio
        max_ratio = self.config.crop_controlflow_clip_roi_max_crop_ratio
        crop_size = min_ratio + size_unit * (max_ratio - min_ratio)
        crop_w = crop_size[:, 0]
        crop_h = crop_size[:, 1]
        half_w = crop_w / 2.0
        half_h = crop_h / 2.0
        center_x = torch.where(
            half_w < 0.5,
            center_x.clamp(half_w, 1.0 - half_w),
            torch.full_like(center_x, 0.5),
        )
        center_y = torch.where(
            half_h < 0.5,
            center_y.clamp(half_h, 1.0 - half_h),
            torch.full_like(center_y, 0.5),
        )

        x1 = (center_x - half_w).clamp(0.0, 1.0)
        y1 = (center_y - half_h).clamp(0.0, 1.0)
        x2 = (center_x + half_w).clamp(0.0, 1.0)
        y2 = (center_y + half_h).clamp(0.0, 1.0)

        spatial_flat = spatial_heatmap.flatten(1)
        entropy = -(spatial_flat * spatial_flat.clamp_min(1e-8).log()).sum(dim=1)
        max_entropy = spatial_flat.new_tensor(float(spatial_flat.shape[1])).log().clamp_min(1e-6)
        confidence = (1.0 - entropy / max_entropy).clamp(0.0, 1.0)
        camera_coords = torch.arange(num_cameras, device=heatmap.device, dtype=heatmap.dtype)
        step_coords = torch.arange(num_obs_steps, device=heatmap.device, dtype=heatmap.dtype)
        camera_norm = camera_step_prob.sum(dim=2).mul(camera_coords[None]).sum(dim=1) / max(
            num_cameras - 1,
            1,
        )
        step_norm = camera_step_prob.sum(dim=1).mul(step_coords[None]).sum(dim=1) / max(
            num_obs_steps - 1,
            1,
        )
        scale = (crop_w + crop_h) / 2.0
        fallback_flag = torch.zeros_like(confidence)
        bbox = torch.stack(
            [x1, y1, x2, y2, confidence, camera_norm, step_norm, scale, fallback_flag],
            dim=1,
        )
        return heatmap, bbox, camera_step_prob


class CropControlFlowClipModel(CropControlFlowModel):
    """CropControlFlow using learned CLIP ROI tokens instead of action-attention ROI."""

    def __init__(self, config: CropControlFlowClipConfig):
        super().__init__(config)
        if not config.image_features:
            raise ValueError("CropControlFlowClip requires image features.")
        self._schedule_stage = "joint"
        self._schedule_step = 0
        self._schedule_crop_enabled = config.crop_controlflow_use_crop_during_training
        self._schedule_global_loss_weight = float(config.crop_controlflow_global_loss_weight)
        self._active_trainable_stage: str | None = None
        self._last_roi_log: dict[str, float] = {}
        self.clip_roi_head = CLIPLearnedROIHead(config, self.global_cond_dim)
        self.clip_crop_summary_proj = nn.Sequential(
            nn.LayerNorm(config.transformer_hidden_dim),
            nn.Linear(config.transformer_hidden_dim, self.rgb_feature_dim),
        )
        if self.crop_rgb_encoder is not None:
            self.crop_rgb_encoder = None

    def set_training_schedule(
        self,
        stage: str,
        step: int,
        crop_enabled: bool,
        global_loss_weight: float,
    ) -> None:
        self._schedule_stage = stage
        self._schedule_step = int(step)
        self._schedule_crop_enabled = bool(crop_enabled)
        self._schedule_global_loss_weight = float(global_loss_weight)

    def _use_crop_during_training(self) -> bool:
        return self._schedule_crop_enabled

    def _global_loss_weight(self) -> float:
        return self._schedule_global_loss_weight

    def _extra_loss_logs(self) -> dict[str, float]:
        stage_id = {"global": 1.0, "roi_warmup": 2.0, "joint": 3.0, "eval": 4.0}.get(
            self._schedule_stage,
            0.0,
        )
        logs = {
            "crop_controlflow_clip_schedule_stage_id": stage_id,
            "crop_controlflow_clip_schedule_step": float(self._schedule_step),
        }
        logs.update(self._last_roi_log)
        return logs

    def _set_module_requires_grad(self, module: nn.Module | None, requires_grad: bool) -> None:
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = requires_grad

    def _set_all_requires_grad(self, requires_grad: bool) -> None:
        for param in self.parameters():
            param.requires_grad = requires_grad

    def _enforce_clip_freeze(self) -> None:
        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                for encoder in self.rgb_encoder:
                    if not self.config.sfp_finetune_clip_image:
                        encoder.model.requires_grad_(False)
                        encoder.model.eval()
            elif self.rgb_encoder is not None and not self.config.sfp_finetune_clip_image:
                self.rgb_encoder.model.requires_grad_(False)
                self.rgb_encoder.model.eval()
        if self.text_encoder is not None and self.config.sfp_freeze_clip:
            self.text_encoder.text_encoder.requires_grad_(False)
            self.text_encoder.text_encoder.eval()

    def configure_trainable_params(self, stage: str) -> None:
        if self._active_trainable_stage == stage:
            self._enforce_clip_freeze()
            return
        self._active_trainable_stage = stage

        if stage in {"global", "joint", "eval"}:
            self._set_all_requires_grad(True)
            if stage == "global":
                self._set_module_requires_grad(self.clip_roi_head, False)
                self._set_module_requires_grad(self.clip_crop_summary_proj, False)
                self._set_module_requires_grad(self.bbox_cond_proj, False)
                self._set_module_requires_grad(self.bbox_token_proj, False)
        elif stage == "roi_warmup":
            self._set_all_requires_grad(False)
            self._set_module_requires_grad(self.clip_roi_head, True)
            self._set_module_requires_grad(self.clip_crop_summary_proj, True)
            self._set_module_requires_grad(self.bbox_cond_proj, True)
            self._set_module_requires_grad(self.bbox_token_proj, True)
            self.context_type_embedding.requires_grad = True

            last_n = min(
                self.config.crop_controlflow_clip_stage2_train_last_n_velocity_layers,
                len(self.velocity_model.blocks),
            )
            if last_n > 0:
                for block in self.velocity_model.blocks[-last_n:]:
                    self._set_module_requires_grad(block, True)
            self._set_module_requires_grad(self.velocity_model.output_norm, True)
            self._set_module_requires_grad(self.velocity_model.action_out_proj, True)
        else:
            raise ValueError(f"Unknown CropControlFlowClip training stage: {stage}.")

        self._enforce_clip_freeze()

    def _prepare_clip_crop_conditioning_from_base(
        self,
        base_conditioning: TokenConditioning,
        crop_tokens: Tensor,
        crop_feats: Tensor,
        roi_bbox: Tensor,
    ) -> TokenConditioning:
        if base_conditioning.crop_feature_start < 0 or base_conditioning.bbox_feature_start < 0:
            raise ValueError("Base conditioning does not contain crop/bbox feature slots.")
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

    def _base_image_tokens(self, conditioning: TokenConditioning) -> Tensor:
        grid = conditioning.visual_grid_size
        expected_tokens = (
            conditioning.num_image_cameras
            * conditioning.num_obs_steps
            * grid
            * grid
        )
        image_token_count = conditioning.image_token_end - conditioning.image_token_start
        if image_token_count != expected_tokens:
            raise ValueError(
                "Image tokens do not match the expected camera/step/grid layout. "
                f"Got {image_token_count} image tokens, expected {expected_tokens}."
            )
        return conditioning.context_tokens[:, conditioning.image_token_start : conditioning.image_token_end]

    def _encode_single_camera_crop_images_with_input_grad(
        self,
        crop_images: Tensor,
        encoder: nn.Module | None = None,
    ) -> tuple[Tensor, Tensor]:
        if crop_images.ndim != 5:
            raise ValueError(
                "`crop_images` must have shape (B, S, C, H, W). "
                f"Got {tuple(crop_images.shape)}."
            )
        if encoder is None:
            encoder = self.rgb_encoder[0] if self.config.use_separate_rgb_encoder_per_camera else self.rgb_encoder
        batch_size, obs_steps, channels, height, width = crop_images.shape
        pixels = crop_images.reshape(batch_size * obs_steps, channels, height, width).float()
        if encoder.resize is not None:
            pixels = encoder.resize(pixels)
        if encoder.center_crop is not None:
            pixels = encoder.center_crop(pixels)
        pixels = (pixels - encoder.clip_mean.to(dtype=pixels.dtype)) / encoder.clip_std.to(dtype=pixels.dtype)

        hidden = encoder.model(pixel_values=pixels, output_hidden_states=False).last_hidden_state
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

        patches = hidden[:, 1:]
        patch_grid_size = int(patches.shape[1] ** 0.5)
        if patch_grid_size * patch_grid_size != patches.shape[1]:
            raise ValueError(f"CLIP visual token count must form a square grid. Got {patches.shape[1]}.")
        patches = einops.rearrange(
            patches,
            "(b s) (h w) d -> (b s) d h w",
            b=batch_size,
            s=obs_steps,
            h=patch_grid_size,
            w=patch_grid_size,
        )
        patches = F.adaptive_avg_pool2d(patches, (self.visual_grid_size, self.visual_grid_size))
        crop_tokens = einops.rearrange(
            patches,
            "(b s) d h w -> b (s h w) d",
            b=batch_size,
            s=obs_steps,
        )
        crop_tokens = encoder.token_proj(crop_tokens)
        crop_feats = self.clip_crop_summary_proj(crop_tokens.float().mean(dim=1))
        return crop_feats, crop_tokens

    def _clip_roi_crop_tokens(
        self,
        base_conditioning: TokenConditioning,
        action: Tensor,
        timestep: Tensor,
        base_freq: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        image_tokens = self._base_image_tokens(base_conditioning)
        grid = base_conditioning.visual_grid_size
        batch_size = image_tokens.shape[0]
        num_cameras = base_conditioning.num_image_cameras
        num_obs_steps = base_conditioning.num_obs_steps
        heatmap, bbox, camera_step_prob = self.clip_roi_head(
            image_tokens=image_tokens,
            global_cond=base_conditioning.raw_cond,
            rough_action=action,
            timestep=timestep,
            freq=base_freq,
            num_cameras=num_cameras,
            num_obs_steps=num_obs_steps,
            grid=grid,
        )

        camera_weights = camera_step_prob.sum(dim=2)
        with torch.no_grad():
            spatial_heatmap = (camera_step_prob[..., None, None] * heatmap).sum(dim=(1, 2))
            spatial_heatmap = spatial_heatmap / spatial_heatmap.flatten(1).sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1e-8)[:, None]
            roi_entropy = -(spatial_heatmap * spatial_heatmap.clamp_min(1e-8).log()).flatten(1).sum(dim=1)
            roi_entropy = roi_entropy / heatmap.new_tensor(float(grid * grid)).log().clamp_min(1e-6)
            self._last_roi_log = {
                "roi_confidence": float(bbox[:, 4].mean().detach().item()),
                "roi_entropy": float(roi_entropy.mean().detach().item()),
                "roi_camera": float(bbox[:, 5].mean().detach().item()),
                "roi_step": float(bbox[:, 6].mean().detach().item()),
                "roi_scale": float(bbox[:, 7].mean().detach().item()),
            }
        return bbox, heatmap, camera_weights

    def _crop_single_camera_images(
        self,
        images: Tensor,
        bbox: Tensor,
    ) -> Tensor:
        if images.ndim != 5:
            raise ValueError(
                "`images` must have shape (B, S, C, H, W) for single-camera crop. "
                f"Got {tuple(images.shape)}."
            )
        batch_size, obs_steps, channels, height, width = images.shape
        flat_source = images.reshape(batch_size * obs_steps, channels, height, width)
        repeated_bbox = bbox[:, :4].repeat_interleave(obs_steps, dim=0)

        x1, y1, x2, y2 = repeated_bbox.unbind(dim=1)
        theta = torch.zeros((batch_size * obs_steps, 2, 3), device=images.device, dtype=images.dtype)
        theta[:, 0, 0] = (x2 - x1).to(dtype=images.dtype)
        theta[:, 1, 1] = (y2 - y1).to(dtype=images.dtype)
        theta[:, 0, 2] = (x1 + x2 - 1.0).to(dtype=images.dtype)
        theta[:, 1, 2] = (y1 + y2 - 1.0).to(dtype=images.dtype)
        grid = F.affine_grid(theta, size=flat_source.shape, align_corners=False)
        crop = F.grid_sample(flat_source, grid, mode="bilinear", padding_mode="border", align_corners=False)
        return crop.reshape(batch_size, obs_steps, channels, height, width)

    def _encode_weighted_camera_crops(
        self,
        images: Tensor,
        bbox: Tensor,
        camera_weights: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if images.ndim == 5:
            images = images.unsqueeze(1)
        batch_size, _, num_cameras, _, _, _ = images.shape
        if camera_weights.shape != (batch_size, num_cameras):
            raise ValueError(
                "`camera_weights` must have shape (B, num_cameras). "
                f"Got {tuple(camera_weights.shape)} for {num_cameras} cameras."
            )

        crop_feats_by_camera = []
        crop_tokens_by_camera = []
        for camera_idx in range(num_cameras):
            encoder = (
                self.rgb_encoder[camera_idx]
                if self.config.use_separate_rgb_encoder_per_camera
                else self.rgb_encoder
            )
            camera_crop = self._crop_single_camera_images(images[:, :, camera_idx], bbox)
            crop_feats, crop_tokens = self._encode_single_camera_crop_images_with_input_grad(
                camera_crop,
                encoder=encoder,
            )
            crop_feats_by_camera.append(crop_feats)
            crop_tokens_by_camera.append(crop_tokens)

        stacked_feats = torch.stack(crop_feats_by_camera, dim=1)
        stacked_tokens = torch.stack(crop_tokens_by_camera, dim=1)
        weights = camera_weights.to(dtype=stacked_feats.dtype)
        crop_feats = (stacked_feats * weights[:, :, None]).sum(dim=1)
        crop_tokens = (stacked_tokens * weights[:, :, None, None].to(dtype=stacked_tokens.dtype)).sum(dim=1)
        return crop_feats, crop_tokens

    def _build_crop_conditioning(
        self,
        batch: dict[str, Tensor],
        action: Tensor,
        timestep: Tensor,
        base_conditioning: TokenConditioning,
        base_freq: Tensor,
    ) -> tuple[TokenConditioning, Tensor, Tensor, Tensor]:
        base_velocity = self.velocity_model(
            sample=action.float(),
            timestep=timestep,
            global_cond=base_conditioning.normalized_cond,
            context_tokens=base_conditioning.context_tokens,
            context_mask=base_conditioning.context_mask,
            freq=base_freq,
        )
        bbox, heatmap, camera_weights = self._clip_roi_crop_tokens(
            base_conditioning=base_conditioning,
            action=action,
            timestep=timestep,
            base_freq=base_freq,
        )
        images = batch[OBS_IMAGES]
        if images.ndim == 5:
            images = images.unsqueeze(1)
        crop_feats, crop_tokens = self._encode_weighted_camera_crops(images, bbox, camera_weights)
        return (
            self._prepare_clip_crop_conditioning_from_base(
                base_conditioning=base_conditioning,
                crop_tokens=crop_tokens,
                crop_feats=crop_feats,
                roi_bbox=bbox,
            ),
            bbox,
            heatmap,
            base_velocity.detach(),
        )
