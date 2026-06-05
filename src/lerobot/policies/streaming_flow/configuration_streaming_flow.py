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
from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig, DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("streaming_flow")
@dataclass
class StreamingFlowConfig(PreTrainedConfig):
    """Configuration for an image-based Streaming Flow Policy.

    The policy follows the rollout logic from the user's notebook: at inference time it
    integrates a velocity field over actions, chunk by chunk, while carrying the last
    predicted action state across chunk boundaries.
    """

    n_obs_steps: int = 2
    chunk_size: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # The same edge-frame dropping heuristic used by diffusion-style chunked policies.
    drop_n_last_frames: int = 7

    # Vision encoder.
    vision_backbone: str = "resnet18"
    resize_shape: tuple[int, int] | None = None
    crop_ratio: float = 1.0
    crop_shape: tuple[int, int] | None = None
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    freeze_vision_encoder: bool = False
    use_group_norm: bool = True
    image_feature_dim: int = 512
    image_dropout: float = 0.1
    use_separate_rgb_encoder_per_camera: bool = False
    sfp_use_imagenet_visual_norm: bool = True

    # Streaming flow backbone.
    down_dims: tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    embedding_dim: int = 256
    updownsample_type: str = "linear"  # "linear" or "conv"
    timestep_embedding_scale: float = 100.0
    frequency_embedding_scale: float = 0.1

    # Flow-matching / SFP controls.
    sfp_sigma0: float = 0.4
    sfp_k: float = 10.0
    sfp_num_train_points: int = 4
    sfp_use_adaptive_freq: bool = True
    sfp_freq_min: float = 0.2
    sfp_freq_max: float = 5.0
    sfp_clamp_freq_during_training: bool = False
    sfp_clamp_freq_during_eval: bool = True
    sfp_freq_reg_weight: float = 0.0

    # First action state used when a rollout starts without a prior predicted chunk.
    # `auto` preserves the PushT notebook center for 2-D actions and otherwise
    # starts from a normalized raw zero action, which is suitable for delta controls.
    rollout_initial_action_mode: str = "auto"  # "auto", "zero", "constant", or "state"
    rollout_initial_action: tuple[float, ...] | None = None

    # Optional CLIP conditioning used by streaming_flow_v3.
    sfp_use_clip_image_conditioning: bool = False
    sfp_use_clip_text_conditioning: bool = False
    sfp_freeze_clip: bool = True
    vision_encoder_name: str = "openai/clip-vit-base-patch16"
    text_encoder_name: str = "openai/clip-vit-base-patch16"
    tokenizer_max_length: int = 77
    tokenizer_padding: str = "max_length"
    tokenizer_padding_side: str = "right"
    tokenizer_truncation: bool = True
    clip_image_resize_shape: tuple[int, int] | None = None
    clip_image_crop_shape: tuple[int, int] | None = (224, 224)
    clip_image_crop_is_random: bool = True
    clip_text_projection_dim: int = 256

    # Loss.
    do_mask_loss_for_padding: bool = False

    # Optimization.
    use_amp: bool = True
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    optimizer_grad_clip_norm: float = 1.0
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500
    use_ema: bool = True
    ema_decay: float = 0.9999
    ema_min_decay: float = 0.0
    ema_update_after_step: int = 0

    def __post_init__(self):
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                "The chunk size is the upper bound for the number of action steps per model invocation. "
                f"Got {self.n_action_steps=} and {self.chunk_size=}."
            )
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if self.updownsample_type not in {"linear", "conv"}:
            raise ValueError(
                f"`updownsample_type` must be one of ['linear', 'conv']. Got {self.updownsample_type}."
            )
        if self.resize_shape is not None and (
            len(self.resize_shape) != 2 or any(d <= 0 for d in self.resize_shape)
        ):
            raise ValueError(f"`resize_shape` must be a pair of positive integers. Got {self.resize_shape}.")
        if not (0 < self.crop_ratio <= 1.0):
            raise ValueError(f"`crop_ratio` must be in (0, 1]. Got {self.crop_ratio}.")
        if self.resize_shape is not None:
            if self.crop_ratio < 1.0:
                self.crop_shape = (
                    int(self.resize_shape[0] * self.crop_ratio),
                    int(self.resize_shape[1] * self.crop_ratio),
                )
            else:
                self.crop_shape = None
        if self.crop_shape is not None and (self.crop_shape[0] <= 0 or self.crop_shape[1] <= 0):
            raise ValueError(f"`crop_shape` must have positive dimensions. Got {self.crop_shape}.")
        if self.sfp_sigma0 < 0.0:
            raise ValueError(f"`sfp_sigma0` must be non-negative. Got {self.sfp_sigma0}.")
        if self.sfp_k < 0.0:
            raise ValueError(f"`sfp_k` must be non-negative. Got {self.sfp_k}.")
        if self.sfp_num_train_points <= 0:
            raise ValueError(
                f"`sfp_num_train_points` must be positive. Got {self.sfp_num_train_points}."
            )
        if self.sfp_freq_min <= 0.0:
            raise ValueError(f"`sfp_freq_min` must be > 0. Got {self.sfp_freq_min}.")
        if self.sfp_freq_max < self.sfp_freq_min:
            raise ValueError(
                f"`sfp_freq_max` must be >= `sfp_freq_min`. Got {self.sfp_freq_min=} and {self.sfp_freq_max=}."
            )
        if self.rollout_initial_action_mode not in {"auto", "zero", "constant", "state"}:
            raise ValueError(
                "`rollout_initial_action_mode` must be one of ['auto', 'constant', 'state', 'zero']. "
                f"Got {self.rollout_initial_action_mode}."
            )
        if self.rollout_initial_action_mode == "constant" and self.rollout_initial_action is None:
            raise ValueError(
                "`rollout_initial_action` must be provided when `rollout_initial_action_mode='constant'`."
            )
        if self.clip_text_projection_dim <= 0:
            raise ValueError(f"`clip_text_projection_dim` must be positive. Got {self.clip_text_projection_dim}.")

        self.drop_n_last_frames = self.chunk_size - self.n_action_steps - self.n_obs_steps + 1

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        has_images = len(self.image_features) > 0
        if not has_images:
            raise ValueError("StreamingFlowPolicy follows the notebook image-SFP setup and requires images.")

        if self.resize_shape is None and self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the image shapes. Got {self.crop_shape} "
                        f"and image shape {image_ft.shape} for `{key}`."
                    )

        if len(self.image_features) > 0:
            first_key, first_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_ft.shape:
                    raise ValueError(
                        f"`{key}` shape {image_ft.shape} does not match `{first_key}` shape {first_ft.shape}."
                    )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None


@PreTrainedConfig.register_subclass("streaming_flow_v2")
@dataclass
class StreamingFlowV2Config(StreamingFlowConfig):
    """Streaming Flow variant with multi-point SFP supervision during training."""


@PreTrainedConfig.register_subclass("streaming_flow_v3")
@dataclass
class StreamingFlowV3Config(StreamingFlowV2Config):
    """Streaming Flow variant with frozen CLIP image and text conditioning."""

    sfp_use_clip_image_conditioning: bool = True
    sfp_use_clip_text_conditioning: bool = True
    sfp_freeze_clip: bool = True
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )


@PreTrainedConfig.register_subclass("streaming_flow_v4")
@dataclass
class StreamingFlowV4Config(StreamingFlowV3Config):
    """Streaming Flow variant with token-level CLIP conditioning and a transformer velocity expert."""

    transformer_hidden_dim: int = 768
    transformer_num_layers: int = 8
    transformer_num_heads: int = 12
    transformer_ffn_dim: int = 3072
    transformer_dropout: float = 0.1
    transformer_visual_tokens_per_frame: int = 16

    def __post_init__(self):
        super().__post_init__()

        if self.transformer_hidden_dim <= 0:
            raise ValueError(f"`transformer_hidden_dim` must be positive. Got {self.transformer_hidden_dim}.")
        if self.transformer_num_layers <= 0:
            raise ValueError(f"`transformer_num_layers` must be positive. Got {self.transformer_num_layers}.")
        if self.transformer_num_heads <= 0:
            raise ValueError(f"`transformer_num_heads` must be positive. Got {self.transformer_num_heads}.")
        if self.transformer_hidden_dim % self.transformer_num_heads != 0:
            raise ValueError(
                "`transformer_hidden_dim` must be divisible by `transformer_num_heads`. "
                f"Got {self.transformer_hidden_dim=} and {self.transformer_num_heads=}."
            )
        if self.transformer_ffn_dim <= 0:
            raise ValueError(f"`transformer_ffn_dim` must be positive. Got {self.transformer_ffn_dim}.")
        if not 0.0 <= self.transformer_dropout <= 1.0:
            raise ValueError(f"`transformer_dropout` must be in [0, 1]. Got {self.transformer_dropout}.")
        visual_grid_size = math.isqrt(max(0, self.transformer_visual_tokens_per_frame))
        if (
            self.transformer_visual_tokens_per_frame <= 0
            or visual_grid_size**2 != self.transformer_visual_tokens_per_frame
        ):
            raise ValueError(
                "`transformer_visual_tokens_per_frame` must be a positive square number for spatial pooling. "
                f"Got {self.transformer_visual_tokens_per_frame}."
            )


@PreTrainedConfig.register_subclass("streaming_flow_v5")
@dataclass
class StreamingFlowV5Config(StreamingFlowV4Config):
    """Transformer SFP with frequency-scaled residual updates and direct frequency prediction."""

    sfp_freq_init: float = 1.0
    sfp_finetune_clip_image: bool = True
    vision_encoder_lr_multiplier: float = 0.1

    def __post_init__(self):
        super().__post_init__()

        if not self.sfp_freq_min <= self.sfp_freq_init <= self.sfp_freq_max:
            raise ValueError(
                "`sfp_freq_init` must lie within [`sfp_freq_min`, `sfp_freq_max`]. "
                f"Got {self.sfp_freq_init=} with range [{self.sfp_freq_min}, {self.sfp_freq_max}]."
            )
        if self.vision_encoder_lr_multiplier <= 0.0:
            raise ValueError(
                "`vision_encoder_lr_multiplier` must be positive. "
                f"Got {self.vision_encoder_lr_multiplier}."
            )
