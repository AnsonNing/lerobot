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

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import OBS_IMAGES

from ..rtc.configuration_rtc import RTCConfig


@PreTrainedConfig.register_subclass("smolvla_sfp")
@dataclass
class SmolVLASFPConfig(PreTrainedConfig):
    # Input / output structure.
    # SFP horizon mapping:
    #   obs    -> n_obs_steps
    #   pred   -> chunk_size
    #   action -> n_action_steps
    n_obs_steps: int = 2
    chunk_size: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (512, 512)

    # Add empty images. Used by smolvla_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False

    # Converts joint dimensions to relative values with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions_aloha: bool = False

    # Tokenizer
    tokenizer_max_length: int = 48

    # Decoding
    #num_steps: int = 10

    # Attention utils
    use_cache: bool = True

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True
    vlm_lr_multiplier: float = 0.1

    # Training presets
    optimizer_lr: float = 2e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    # Select the VLM backbone.
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    # Set to False in case of training the expert from scratch.
    # True when init from pretrained SmolVLA weights.
    load_vlm_weights: bool = False

    # Whether to use special image tokens around image features.
    add_image_special_tokens: bool = False

    attention_mode: str = "cross_attn"
    prefix_length: int = -1
    pad_language_to: str = "longest"

    # Less or equal to 0 is the default where the action expert has
    # the same number of layers of VLM.
    num_expert_layers: int = -1
    # Number of layers used in the VLM (first num_vlm_layers layers)
    num_vlm_layers: int = 16
    # Interleave SA layers each self_attn_every_n_layers
    self_attn_every_n_layers: int = 2
    # The action expert hidden size (wrt to the VLM)
    expert_width_multiplier: float = 0.75

    # Sensitivity range for the timestep used in sine-cosine positional encoding.
    min_period: float = 4e-3
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    compile_model: bool = False
    compile_mode: str = "max-autotune"

    # Stabilizing CFM / SFP training parameters.
    sfp_sigma0: float = 0.4
    sfp_k: float = 1.0
    sfp_num_train_points: int = 4
    sfp_num_consistency_points: int = 0
    sfp_consistency_loss_weight: float = 0.0
    sfp_consistency_delta_multipliers: tuple[int, ...] = (1, 2, 4, 8)

    # Adaptive frequency controls.
    sfp_use_adaptive_freq: bool = True
    sfp_freq_min: float = 0.2
    sfp_freq_max: float = 5.0
    sfp_freq_init: float = 1.0
    sfp_clamp_freq_during_training: bool = False
    sfp_clamp_freq_during_eval: bool = True
    sfp_reinit_granularity_predictor: bool = True
    sfp_freq_reg_weight: float = 0.0
    sfp_adaln_zero_init: bool = True
    sfp_adaln_residual_gate_init: float = 1.0
    sfp_adaln_residual_gate_span: float = 0.05

    # Inference initialization for ODE rollout.
    sfp_init_from_state: bool = False
    sfp_init_noise_std: float = 0.0

    def __post_init__(self):
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by smolvla for aloha real models. "
                "It is not ported yet in LeRobot."
            )

        if self.vlm_lr_multiplier <= 0.0:
            raise ValueError(f"`vlm_lr_multiplier` must be positive, got {self.vlm_lr_multiplier}.")
        if self.sfp_sigma0 < 0.0:
            raise ValueError(f"`sfp_sigma0` must be non-negative, got {self.sfp_sigma0}.")
        if self.sfp_k < 0.0:
            raise ValueError(f"`sfp_k` must be non-negative, got {self.sfp_k}.")
        if self.sfp_num_train_points <= 0:
            raise ValueError(
                f"`sfp_num_train_points` must be positive, got {self.sfp_num_train_points}."
            )
        if self.sfp_num_consistency_points < 0:
            raise ValueError(
                f"`sfp_num_consistency_points` must be non-negative, got {self.sfp_num_consistency_points}."
            )
        if self.sfp_consistency_loss_weight < 0.0:
            raise ValueError(
                f"`sfp_consistency_loss_weight` must be non-negative, got {self.sfp_consistency_loss_weight}."
            )
        if len(self.sfp_consistency_delta_multipliers) == 0:
            raise ValueError("`sfp_consistency_delta_multipliers` must not be empty.")
        if any(multiplier <= 0 for multiplier in self.sfp_consistency_delta_multipliers):
            raise ValueError(
                "`sfp_consistency_delta_multipliers` must contain positive integers, got "
                f"{self.sfp_consistency_delta_multipliers}."
            )
        if self.sfp_freq_min <= 0.0:
            raise ValueError(f"`sfp_freq_min` must be > 0, got {self.sfp_freq_min}.")
        if self.sfp_freq_max < self.sfp_freq_min:
            raise ValueError(
                f"`sfp_freq_max` must be >= `sfp_freq_min` ({self.sfp_freq_min}), got {self.sfp_freq_max}."
            )
        if not self.sfp_freq_min <= self.sfp_freq_init <= self.sfp_freq_max:
            raise ValueError(
                "`sfp_freq_init` must lie within [`sfp_freq_min`, `sfp_freq_max`]. "
                f"Got {self.sfp_freq_init=} with range [{self.sfp_freq_min}, {self.sfp_freq_max}]."
            )
        if self.sfp_freq_reg_weight < 0.0:
            raise ValueError(
                f"`sfp_freq_reg_weight` must be non-negative, got {self.sfp_freq_reg_weight}."
            )
        if self.sfp_adaln_residual_gate_init < 0.0:
            raise ValueError(
                "`sfp_adaln_residual_gate_init` must be non-negative, got "
                f"{self.sfp_adaln_residual_gate_init}."
            )
        if self.sfp_adaln_residual_gate_span < 0.0:
            raise ValueError(
                "`sfp_adaln_residual_gate_span` must be non-negative, got "
                f"{self.sfp_adaln_residual_gate_span}."
            )
        if self.sfp_init_noise_std < 0.0:
            raise ValueError(
                f"`sfp_init_noise_std` must be non-negative, got {self.sfp_init_noise_std}."
            )

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(self.n_obs_steps))

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None


@PreTrainedConfig.register_subclass("smolvla_sfp_v2")
@dataclass
class SmolVLASFPV2Config(SmolVLASFPConfig):
    pass
