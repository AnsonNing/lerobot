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
from typing import TYPE_CHECKING

import einops
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn

from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from lerobot.utils.import_utils import _transformers_available, require_package

from ..pretrained import PreTrainedPolicy
from .configuration_streaming_flow import StreamingFlowV5Config
from .modeling_streaming_flow_v3 import (
    SinusoidalPosEmb,
    StreamingFlowModel as CLIPStreamingFlowModel,
    StreamingFlowPolicy as CLIPStreamingFlowPolicy,
    linearly_interpolate_trajectory,
    sample_cfm_inputs_and_targets,
)

if TYPE_CHECKING or _transformers_available:
    from transformers import CLIPTextModel, CLIPVisionModel
else:
    CLIPTextModel = None
    CLIPVisionModel = None


class StreamingFlowPolicy(CLIPStreamingFlowPolicy):
    """Transformer SFP with frequency-scaled residual velocity-field updates."""

    config_class = StreamingFlowV5Config
    name = "streaming_flow_v5"

    def __init__(self, config: StreamingFlowV5Config, **kwargs):
        require_package("transformers", extra="multi_task_dit")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.register_buffer("_action_min", torch.empty(0), persistent=True)
        self.register_buffer("_action_max", torch.empty(0), persistent=True)
        self.register_buffer("_ema_step", torch.zeros((), dtype=torch.long), persistent=True)
        self._init_normalization_buffers(kwargs.get("dataset_stats"))
        self.model = StreamingFlowModel(config)
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
            if "rgb_encoder" in name and ".model." in name:
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


class StreamingFlowModel(CLIPStreamingFlowModel):
    """CLIP token conditioning coupled to a streaming transformer velocity expert."""

    def __init__(self, config: StreamingFlowV5Config):
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
        self.velocity_model = AdaptiveStreamingFlowTransformer(config=config, global_cond_dim=global_cond_dim)

        if config.compile_model:
            self.velocity_model = torch.compile(self.velocity_model, mode=config.compile_mode)

    def _prepare_token_conditioning(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        cond_feats: list[Tensor] = []
        context_tokens: list[Tensor] = []
        context_masks: list[Tensor] = []
        batch_size = None

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
            cond_feats.append(image_feats)
            image_tokens = image_tokens + self.context_type_embedding[:, 0:1]
            context_tokens.append(image_tokens)
            context_masks.append(
                torch.ones(image_tokens.shape[:2], dtype=torch.bool, device=image_tokens.device)
            )

        if self.text_encoder is not None:
            if OBS_LANGUAGE_TOKENS not in batch or OBS_LANGUAGE_ATTENTION_MASK not in batch:
                raise ValueError(
                    "StreamingFlowV5 requires tokenized task text when CLIP text conditioning is enabled."
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

        if not cond_feats or batch_size is None:
            raise ValueError("StreamingFlowV5 received no conditioning features.")

        raw_cond = torch.cat(cond_feats, dim=-1)
        normalized_cond = F.normalize(raw_cond, dim=-1) if raw_cond.shape[-1] > 1 else raw_cond
        return raw_cond, normalized_cond, torch.cat(context_tokens, dim=1), torch.cat(context_masks, dim=1)

    def _predict_frequency(self, raw_cond: Tensor, clamp: bool) -> tuple[Tensor, Tensor, Tensor]:
        if not self.config.sfp_use_adaptive_freq:
            freq = torch.ones(raw_cond.shape[0], device=raw_cond.device, dtype=torch.float32)
            return freq, freq, freq
        raw_freq = self.velocity_model.granularity_predictor(raw_cond.float())
        clamped_freq = torch.clamp(raw_freq, min=self.config.sfp_freq_min, max=self.config.sfp_freq_max)
        freq = clamped_freq if clamp else raw_freq
        return raw_freq, freq, clamped_freq

    def integrate_actions(
        self,
        batch: dict[str, Tensor],
        init_action: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, float]]:
        raw_cond, normalized_cond, context_tokens, context_mask = self._prepare_token_conditioning(batch)
        raw_freq, freq, clamped_freq = self._predict_frequency(
            raw_cond, clamp=self.config.sfp_clamp_freq_during_eval
        )
        action = self._initial_action(batch, init_action=init_action)
        dt = 1.0 / max(self.config.chunk_size - self.config.n_obs_steps, 1)
        action_chunk = []
        initial_action = action.detach()
        first_velocity_mean_abs = 0.0

        for step_idx in range(self.config.n_action_steps):
            timestep = torch.full(
                (action.shape[0],), step_idx * dt, device=action.device, dtype=torch.float32
            )
            velocity = self.velocity_model(
                sample=action.float(),
                timestep=timestep,
                global_cond=normalized_cond,
                context_tokens=context_tokens,
                context_mask=context_mask,
                freq=freq,
            )
            if step_idx == 0:
                first_velocity_mean_abs = float(velocity.abs().mean().detach().cpu())
            action = action + velocity * dt
            action_chunk.append(action.squeeze(1))

        stacked_chunk = torch.stack(action_chunk, dim=1)
        final_action = action.detach()
        return (
            stacked_chunk,
            final_action,
            {
                "pred_freq": float(freq.mean().detach().cpu()),
                "raw_pred_freq": float(raw_freq.mean().detach().cpu()),
                "clamped_pred_freq": float(clamped_freq.mean().detach().cpu()),
                "dt": float(dt),
                "init_action_mean_abs": float(initial_action.abs().mean().detach().cpu()),
                "final_action_mean_abs": float(final_action.abs().mean().detach().cpu()),
                "chunk_delta_mean_abs": float(
                    (stacked_chunk[:, 1:] - stacked_chunk[:, :-1]).abs().mean().detach().cpu()
                )
                if stacked_chunk.shape[1] > 1
                else 0.0,
                "first_velocity_mean_abs": first_velocity_mean_abs,
            },
        )

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
        start_idx = min(max(self.config.n_obs_steps - 1, 0), batch[ACTION].shape[1] - 1)
        trajectory = batch[ACTION][:, start_idx:, :]
        num_queries = max(1, int(self.config.sfp_num_train_points))
        time_shape = (trajectory.shape[0], num_queries) if num_queries > 1 else (trajectory.shape[0],)
        time = torch.rand(time_shape, device=trajectory.device, dtype=torch.float32) * 0.999 + 0.001
        xi_t, dxi_dt = linearly_interpolate_trajectory(trajectory, time)
        noised_action, target_velocity = sample_cfm_inputs_and_targets(
            xi_t, dxi_dt, time, k=self.config.sfp_k, sigma0=self.config.sfp_sigma0
        )

        if num_queries > 1:
            batch_size, _, action_dim = noised_action.shape
            flat_noised_action = noised_action.reshape(batch_size * num_queries, action_dim)
            flat_target_velocity = target_velocity.reshape(batch_size * num_queries, action_dim)
            flat_time = time.reshape(batch_size * num_queries)
            flat_cond = normalized_cond.repeat_interleave(num_queries, dim=0)
            flat_freq = freq.repeat_interleave(num_queries, dim=0)
            flat_tokens = context_tokens.repeat_interleave(num_queries, dim=0)
            flat_mask = context_mask.repeat_interleave(num_queries, dim=0)
        else:
            flat_noised_action = noised_action
            flat_target_velocity = target_velocity
            flat_time = time
            flat_cond = normalized_cond
            flat_freq = freq
            flat_tokens = context_tokens
            flat_mask = context_mask

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
            "pred_freq": float(freq.mean().detach().item()),
            "raw_pred_freq": float(raw_freq.mean().detach().item()),
            "clamped_pred_freq": float(clamped_freq.mean().detach().item()),
            "sfp_num_train_points": float(num_queries),
        }
        return (per_sample_loss, output_dict) if reduction == "none" else (mean_loss, output_dict)


class CLIPPatchTokenImageEncoder(nn.Module):
    """CLIP vision encoder returning pooled features and compressed patch tokens."""

    def __init__(self, config: StreamingFlowV5Config):
        super().__init__()
        self.config = config
        self.model = CLIPVisionModel.from_pretrained(config.vision_encoder_name)
        self.feature_dim = config.image_feature_dim
        clip_dim = self.model.config.hidden_size
        proj_in_dim = clip_dim * 2 if config.n_obs_steps < 3 else clip_dim * 3
        self.summary_proj = nn.Sequential(nn.LayerNorm(proj_in_dim), nn.Linear(proj_in_dim, self.feature_dim))
        self.token_proj = nn.Sequential(
            nn.LayerNorm(clip_dim), nn.Linear(clip_dim, config.transformer_hidden_dim)
        )
        self.visual_grid_size = math.isqrt(config.transformer_visual_tokens_per_frame)
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
        if not config.sfp_finetune_clip_image:
            self.model.requires_grad_(False)
            self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.config.sfp_finetune_clip_image:
            self.model.eval()
        return self

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, obs_steps, channels, height, width = obs.shape
        pixels = obs.reshape(batch_size * obs_steps, channels, height, width).float()
        if self.resize is not None:
            pixels = self.resize(pixels)
        if self.center_crop is not None:
            pixels = self.maybe_random_crop(pixels) if self.training else self.center_crop(pixels)
        pixels = (pixels - self.clip_mean.to(dtype=pixels.dtype)) / self.clip_std.to(dtype=pixels.dtype)
        if not self.config.sfp_finetune_clip_image:
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

        patches = hidden[:, 1:]
        patch_grid_size = math.isqrt(patches.shape[1])
        if patch_grid_size**2 != patches.shape[1]:
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
        patches = einops.rearrange(patches, "(b s) d h w -> b (s h w) d", b=batch_size, s=obs_steps)
        return self.summary_proj(fused), self.token_proj(patches)


class CLIPTokenTextConditionEncoder(nn.Module):
    """Frozen CLIP text encoder returning pooled and token-level representations."""

    def __init__(self, config: StreamingFlowV5Config):
        super().__init__()
        self.config = config
        self.text_encoder = CLIPTextModel.from_pretrained(config.text_encoder_name)
        self.feature_dim = config.clip_text_projection_dim
        text_dim = self.text_encoder.config.hidden_size
        self.summary_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, self.feature_dim))
        self.token_proj = nn.Sequential(
            nn.LayerNorm(text_dim), nn.Linear(text_dim, config.transformer_hidden_dim)
        )
        if config.sfp_freeze_clip:
            self.text_encoder.requires_grad_(False)
            self.text_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.sfp_freeze_clip:
            self.text_encoder.eval()
        return self

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        input_ids = input_ids.long().reshape(input_ids.shape[0], -1)
        attention_mask = attention_mask.long().reshape(attention_mask.shape[0], -1)
        if self.config.sfp_freeze_clip:
            with torch.no_grad():
                outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return (
            self.summary_proj(outputs.pooler_output),
            self.token_proj(outputs.last_hidden_state),
            attention_mask.bool(),
        )


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1.0 + scale) + shift


class AdaLNZeroCrossAttentionBlock(nn.Module):
    """Transformer decoder block with frequency-scaled residual updates."""

    def __init__(self, config: StreamingFlowV5Config):
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
    ) -> Tensor:
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
        cross_out, _ = self.cross_attn(
            cross_input, memory, memory, key_padding_mask=memory_padding_mask, need_weights=False
        )
        x = x + residual_scale * gate_cross[:, None] * self.dropout(cross_out)
        mlp_input = modulate(self.norm_mlp(x), shift_mlp[:, None], scale_mlp[:, None])
        x = x + residual_scale * gate_mlp[:, None] * self.dropout(self.mlp(mlp_input))
        return x


class StepScalingLayer(nn.Module):
    """Make frequency an explicit multiplicative control over velocity-field updates."""

    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, 1),
        )
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.constant_(self.fc[-1].bias, math.log(math.expm1(1.0)))

    def forward(self, embedding: Tensor, freq: Tensor) -> Tensor:
        learned_scale = F.softplus(self.fc(embedding)).squeeze(-1)
        return freq * learned_scale


class DirectFrequencyPredictor(nn.Module):
    """Predict frequency values directly, initialized at a usable frequency."""

    def __init__(self, cond_dim: int, config: StreamingFlowV5Config):
        super().__init__()
        hidden_dim = max(128, cond_dim // 4)
        self.net = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.xavier_uniform_(self.net[1].weight, gain=0.8)
        nn.init.zeros_(self.net[1].bias)
        nn.init.zeros_(self.net[3].weight)
        nn.init.constant_(self.net[3].bias, config.sfp_freq_init)

    def forward(self, cond: Tensor) -> Tensor:
        return self.net(cond).squeeze(-1)


class AdaptiveStreamingFlowTransformer(nn.Module):
    """Token-grounded velocity expert with frequency-scaled transformer residual updates."""

    def __init__(self, config: StreamingFlowV5Config, global_cond_dim: int):
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
            [AdaLNZeroCrossAttentionBlock(config) for _ in range(config.transformer_num_layers)]
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
    ) -> Tensor:
        del smooth_freq
        if sample.ndim == 2:
            sample = sample.unsqueeze(1)
        batch_size, query_len, _ = sample.shape
        timestep = torch.as_tensor(timestep, dtype=torch.float32, device=sample.device).view(-1)
        if timestep.numel() == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)
        if freq is None:
            freq = self.granularity_predictor(global_cond)
        elif freq.ndim > 1:
            freq = freq.squeeze(-1)
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
        for block in self.blocks:
            hidden = block(hidden, memory, memory_padding_mask, condition, delta)
        return self.action_out_proj(self.output_norm(hidden))
