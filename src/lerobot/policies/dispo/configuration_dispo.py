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

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig, DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("dispo")
@dataclass
class DiSPoConfig(PreTrainedConfig):
    """Configuration for a LeRobot-native DiSPo policy.

    This policy keeps DiSPo's diffusion + state-space-model shape, but implements
    the SSM block in pure PyTorch so it can run in the LeRobot environment without
    the original repo's Python-version-specific CUDA extension.
    """

    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )
    drop_n_last_frames: int = 7

    # Vision encoder.
    vision_backbone: str = "resnet18"
    resize_shape: tuple[int, int] | None = None
    crop_ratio: float = 1.0
    crop_shape: tuple[int, int] | None = None
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    use_group_norm: bool = False
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False

    # DiSPo / SSM denoiser.
    hidden_dim: int = 256
    depth: int = 12
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: int | None = None
    dropout: float = 0.0
    mlp_ratio: float = 2.0
    diffusion_step_embed_dim: int = 128

    # Optional Mamba3-inspired gated MIMO trapezoidal SSM block. The default
    # keeps the original LeRobot DiSPo pure-PyTorch selective scan path.
    ssm_block_type: str = "dispo"
    mamba3_mimo_rank: int = 4
    mamba3_omega_min: float = 0.05
    mamba3_single_stream_fallback: bool = True
    mamba3_use_rotary_angle: bool = True
    mamba3_use_complex_ssm: bool = True
    mamba3_use_output_gate: bool = True

    # Coarse-to-fine control. Values multiply the selective-scan dt at training
    # time. Leave at (1.0,) for ordinary diffusion behavior.
    train_delta_rate_choices: tuple[float, ...] = (1.0,)
    eval_delta_rate: float = 1.0

    # Noise scheduler.
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0
    num_inference_steps: int | None = None

    # Training.
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"
    do_mask_loss_for_padding: bool = False
    use_amp: bool = True
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-4
    optimizer_grad_clip_norm: float = 1.0
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        if self.n_action_steps > self.horizon - self.n_obs_steps + 1:
            raise ValueError(
                "`n_action_steps` must fit inside the generated horizon after the current observation. "
                f"Got {self.n_action_steps=}, {self.horizon=}, {self.n_obs_steps=}."
            )
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
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
        if self.hidden_dim <= 0:
            raise ValueError(f"`hidden_dim` must be positive. Got {self.hidden_dim}.")
        if self.depth <= 0:
            raise ValueError(f"`depth` must be positive. Got {self.depth}.")
        if self.d_state <= 0:
            raise ValueError(f"`d_state` must be positive. Got {self.d_state}.")
        if self.d_conv <= 0:
            raise ValueError(f"`d_conv` must be positive. Got {self.d_conv}.")
        if self.expand <= 0:
            raise ValueError(f"`expand` must be positive. Got {self.expand}.")
        if self.ssm_block_type not in {"dispo", "mamba3_gated_mimo"}:
            raise ValueError(
                "`ssm_block_type` must be 'dispo' or 'mamba3_gated_mimo'. "
                f"Got {self.ssm_block_type}."
            )
        if self.mamba3_mimo_rank <= 0:
            raise ValueError(f"`mamba3_mimo_rank` must be positive. Got {self.mamba3_mimo_rank}.")
        if not (0.0 <= self.mamba3_omega_min < 1.0):
            raise ValueError(
                f"`mamba3_omega_min` must be in [0, 1). Got {self.mamba3_omega_min}."
            )
        if self.mamba3_use_complex_ssm and not self.mamba3_use_rotary_angle:
            raise ValueError("`mamba3_use_complex_ssm=True` requires `mamba3_use_rotary_angle=True`.")
        if any(rate <= 0 for rate in self.train_delta_rate_choices):
            raise ValueError(
                f"All `train_delta_rate_choices` must be positive. Got {self.train_delta_rate_choices}."
            )
        if self.eval_delta_rate <= 0:
            raise ValueError(f"`eval_delta_rate` must be positive. Got {self.eval_delta_rate}.")
        if self.prediction_type not in {"epsilon", "sample"}:
            raise ValueError("`prediction_type` must be 'epsilon' or 'sample'.")
        if self.noise_scheduler_type not in {"DDPM", "DDIM"}:
            raise ValueError("`noise_scheduler_type` must be 'DDPM' or 'DDIM'.")

        self.drop_n_last_frames = self.horizon - self.n_action_steps - self.n_obs_steps + 1

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
        if self.action_feature is None:
            raise ValueError("DiSPoPolicy requires an `action` output feature.")

        state_features = {
            key: ft for key, ft in (self.input_features or {}).items() if ft.type is FeatureType.STATE
        }
        has_images = len(self.image_features) > 0
        if not has_images and not state_features:
            raise ValueError("DiSPoPolicy requires at least one image or state observation feature.")

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
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None


@PreTrainedConfig.register_subclass("dispo_mamba3_flow")
@dataclass
class DiSPoMamba3FlowConfig(DiSPoConfig):
    """DiSPo-Mamba3 policy trained with GR00T-style flow matching."""

    ssm_block_type: str = "mamba3_gated_mimo"
    num_train_timesteps: int = 1000
    num_inference_steps: int | None = 10

    # Flow matching time sampling, following GR00T's Beta-distributed noise schedule.
    flow_noise_beta_alpha: float = 1.5
    flow_noise_beta_beta: float = 1.0
    flow_noise_s: float = 0.999
    flow_num_timestep_buckets: int = 1000

    def __post_init__(self):
        super().__post_init__()
        if self.ssm_block_type != "mamba3_gated_mimo":
            raise ValueError("DiSPoMamba3FlowConfig requires `ssm_block_type='mamba3_gated_mimo'`.")
        if self.num_inference_steps is None or self.num_inference_steps <= 0:
            raise ValueError(
                f"`num_inference_steps` must be a positive integer for flow matching. "
                f"Got {self.num_inference_steps}."
            )
        if self.flow_noise_beta_alpha <= 0 or self.flow_noise_beta_beta <= 0:
            raise ValueError(
                "`flow_noise_beta_alpha` and `flow_noise_beta_beta` must be positive. "
                f"Got {self.flow_noise_beta_alpha}, {self.flow_noise_beta_beta}."
            )
        if self.flow_noise_s <= 0:
            raise ValueError(f"`flow_noise_s` must be positive. Got {self.flow_noise_s}.")
        if self.flow_num_timestep_buckets <= 1:
            raise ValueError(
                f"`flow_num_timestep_buckets` must be greater than 1. Got {self.flow_num_timestep_buckets}."
            )
        self.num_train_timesteps = self.flow_num_timestep_buckets


@PreTrainedConfig.register_subclass("dispo_mamba3_flow_clip")
@dataclass
class DiSPoMamba3FlowClipConfig(DiSPoMamba3FlowConfig):
    """DiSPo-Mamba3 flow matching policy with frozen CLIP image/text conditioning."""

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # CLIP image/text encoders. Images are normalized inside the CLIP encoder,
    # so VISUAL normalization stays IDENTITY for this policy.
    vision_encoder_name: str = "openai/clip-vit-base-patch16"
    text_encoder_name: str = "openai/clip-vit-base-patch16"
    clip_freeze_image_encoder: bool = True
    clip_freeze_text_encoder: bool = True
    clip_image_feature_dim: int = 512
    clip_text_projection_dim: int = 256
    clip_image_resize_shape: tuple[int, int] | None = None
    clip_image_crop_shape: tuple[int, int] | None = (224, 224)
    clip_image_crop_is_random: bool = True

    tokenizer_max_length: int = 77
    tokenizer_padding: str = "max_length"
    tokenizer_padding_side: str = "right"
    tokenizer_truncation: bool = True

    def __post_init__(self):
        super().__post_init__()
        if "clip" not in self.vision_encoder_name.lower():
            raise ValueError(
                f"`vision_encoder_name` must be a CLIP model for this policy. Got {self.vision_encoder_name}."
            )
        if "clip" not in self.text_encoder_name.lower():
            raise ValueError(
                f"`text_encoder_name` must be a CLIP model for this policy. Got {self.text_encoder_name}."
            )
        if self.clip_image_feature_dim <= 0:
            raise ValueError(
                f"`clip_image_feature_dim` must be positive. Got {self.clip_image_feature_dim}."
            )
        if self.clip_text_projection_dim <= 0:
            raise ValueError(
                f"`clip_text_projection_dim` must be positive. Got {self.clip_text_projection_dim}."
            )
        if self.tokenizer_max_length <= 0:
            raise ValueError(f"`tokenizer_max_length` must be positive. Got {self.tokenizer_max_length}.")
