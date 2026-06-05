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

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.streaming_flow.configuration_streaming_flow import StreamingFlowV5Config


@PreTrainedConfig.register_subclass("crop_controlflow")
@dataclass
class CropControlFlowConfig(StreamingFlowV5Config):
    """Streaming Flow v5 with online action-attention crop conditioning.

    The backbone remains CLIP image/text token conditioning plus the v5 transformer
    velocity expert. The crop branch is produced online from the policy's own
    action cross-attention heatmap and is used as an additional local observation
    stream in a second pass.
    """

    crop_controlflow_enabled: bool = True
    crop_controlflow_crop_size_ratio: float = 0.5
    crop_controlflow_crop_size_ratios: tuple[float, ...] = (0.35, 0.5, 0.7)
    crop_controlflow_attention_camera_index: int = 0
    crop_controlflow_attention_obs_step: int = -1
    crop_controlflow_bbox_feature_dim: int = 64
    crop_controlflow_use_bbox_token: bool = True
    crop_controlflow_detach_attention: bool = True
    crop_controlflow_attention_num_layers: int = 4
    crop_controlflow_attention_normalization: str = "contrast"
    crop_controlflow_attention_contrast_eps: float = 1e-6
    crop_controlflow_attention_smoothing_kernel: int = 3
    crop_controlflow_fallback_to_center_on_low_confidence: bool = True
    crop_controlflow_max_heatmap_entropy: float = 0.98
    crop_controlflow_min_window_mass_ratio: float = 1.05
    crop_controlflow_score_log_max: float = 4.0
    crop_controlflow_use_crop_during_training: bool = True
    crop_controlflow_use_crop_during_eval: bool = True
    crop_controlflow_global_loss_weight: float = 0.0

    def __post_init__(self):
        super().__post_init__()

        if not (0.0 < self.crop_controlflow_crop_size_ratio <= 1.0):
            raise ValueError(
                "`crop_controlflow_crop_size_ratio` must be in (0, 1]. "
                f"Got {self.crop_controlflow_crop_size_ratio}."
            )
        if not self.crop_controlflow_crop_size_ratios:
            self.crop_controlflow_crop_size_ratios = (self.crop_controlflow_crop_size_ratio,)
        self.crop_controlflow_crop_size_ratios = tuple(
            float(ratio) for ratio in self.crop_controlflow_crop_size_ratios
        )
        for ratio in self.crop_controlflow_crop_size_ratios:
            if not (0.0 < ratio <= 1.0):
                raise ValueError(
                    "`crop_controlflow_crop_size_ratios` values must be in (0, 1]. "
                    f"Got {self.crop_controlflow_crop_size_ratios}."
                )
        if self.crop_controlflow_bbox_feature_dim <= 0:
            raise ValueError(
                "`crop_controlflow_bbox_feature_dim` must be positive. "
                f"Got {self.crop_controlflow_bbox_feature_dim}."
            )
        if self.crop_controlflow_global_loss_weight < 0.0:
            raise ValueError(
                "`crop_controlflow_global_loss_weight` must be non-negative. "
                f"Got {self.crop_controlflow_global_loss_weight}."
            )
        if self.crop_controlflow_attention_num_layers <= 0:
            raise ValueError(
                "`crop_controlflow_attention_num_layers` must be positive. "
                f"Got {self.crop_controlflow_attention_num_layers}."
            )
        if self.crop_controlflow_attention_normalization not in {"none", "contrast"}:
            raise ValueError(
                "`crop_controlflow_attention_normalization` must be one of {'none', 'contrast'}. "
                f"Got {self.crop_controlflow_attention_normalization}."
            )
        if self.crop_controlflow_attention_contrast_eps < 0.0:
            raise ValueError(
                "`crop_controlflow_attention_contrast_eps` must be non-negative. "
                f"Got {self.crop_controlflow_attention_contrast_eps}."
            )
        if self.crop_controlflow_attention_smoothing_kernel <= 0:
            raise ValueError(
                "`crop_controlflow_attention_smoothing_kernel` must be positive. "
                f"Got {self.crop_controlflow_attention_smoothing_kernel}."
            )
        if self.crop_controlflow_attention_smoothing_kernel % 2 == 0:
            raise ValueError(
                "`crop_controlflow_attention_smoothing_kernel` must be odd. "
                f"Got {self.crop_controlflow_attention_smoothing_kernel}."
            )
        if not (0.0 < self.crop_controlflow_max_heatmap_entropy <= 1.0):
            raise ValueError(
                "`crop_controlflow_max_heatmap_entropy` must be in (0, 1]. "
                f"Got {self.crop_controlflow_max_heatmap_entropy}."
            )
        if self.crop_controlflow_min_window_mass_ratio < 0.0:
            raise ValueError(
                "`crop_controlflow_min_window_mass_ratio` must be non-negative. "
                f"Got {self.crop_controlflow_min_window_mass_ratio}."
            )
        if self.crop_controlflow_score_log_max <= 0.0:
            raise ValueError(
                "`crop_controlflow_score_log_max` must be positive. "
                f"Got {self.crop_controlflow_score_log_max}."
            )


@PreTrainedConfig.register_subclass("crop_controlflow_clip")
@dataclass
class CropControlFlowClipConfig(CropControlFlowConfig):
    """CropControlFlow variant with a learned CLIP image/text ROI selector.

    CLIP image/text encoders are frozen by default. The learned ROI head predicts
    a soft patch heatmap and continuous bbox from CLIP tokens, state, frequency,
    and a rough action query. Local crop tokens are sampled from the global CLIP
    token map, so the action loss can update the ROI head through differentiable
    token sampling.
    """

    sfp_finetune_clip_image: bool = False
    sfp_freeze_clip: bool = True
    crop_controlflow_clip_roi_hidden_dim: int = 256
    crop_controlflow_clip_roi_temperature: float = 1.0
    crop_controlflow_clip_roi_min_crop_ratio: float = 0.25
    crop_controlflow_clip_roi_max_crop_ratio: float = 0.75
    crop_controlflow_clip_roi_init_crop_ratio: float = 0.5
    crop_controlflow_clip_roi_detach_clip_features: bool = True
    crop_controlflow_clip_schedule_enabled: bool = True
    crop_controlflow_clip_stage1_global_steps: int = 5000
    crop_controlflow_clip_stage2_warmup_steps: int = 5000
    crop_controlflow_clip_stage2_global_loss_weight: float = 0.8
    crop_controlflow_clip_stage3_global_loss_weight: float = 0.2
    crop_controlflow_clip_stage2_train_last_n_velocity_layers: int = 2

    def __post_init__(self):
        super().__post_init__()

        if self.crop_controlflow_clip_roi_hidden_dim <= 0:
            raise ValueError(
                "`crop_controlflow_clip_roi_hidden_dim` must be positive. "
                f"Got {self.crop_controlflow_clip_roi_hidden_dim}."
            )
        if self.crop_controlflow_clip_roi_temperature <= 0.0:
            raise ValueError(
                "`crop_controlflow_clip_roi_temperature` must be positive. "
                f"Got {self.crop_controlflow_clip_roi_temperature}."
            )
        if not (0.0 < self.crop_controlflow_clip_roi_min_crop_ratio <= 1.0):
            raise ValueError(
                "`crop_controlflow_clip_roi_min_crop_ratio` must be in (0, 1]. "
                f"Got {self.crop_controlflow_clip_roi_min_crop_ratio}."
            )
        if not (0.0 < self.crop_controlflow_clip_roi_max_crop_ratio <= 1.0):
            raise ValueError(
                "`crop_controlflow_clip_roi_max_crop_ratio` must be in (0, 1]. "
                f"Got {self.crop_controlflow_clip_roi_max_crop_ratio}."
            )
        if self.crop_controlflow_clip_roi_min_crop_ratio > self.crop_controlflow_clip_roi_max_crop_ratio:
            raise ValueError(
                "`crop_controlflow_clip_roi_min_crop_ratio` must be <= "
                "`crop_controlflow_clip_roi_max_crop_ratio`. "
                f"Got {self.crop_controlflow_clip_roi_min_crop_ratio} > "
                f"{self.crop_controlflow_clip_roi_max_crop_ratio}."
            )
        if not (
            self.crop_controlflow_clip_roi_min_crop_ratio
            <= self.crop_controlflow_clip_roi_init_crop_ratio
            <= self.crop_controlflow_clip_roi_max_crop_ratio
        ):
            raise ValueError(
                "`crop_controlflow_clip_roi_init_crop_ratio` must lie within "
                "[`crop_controlflow_clip_roi_min_crop_ratio`, "
                "`crop_controlflow_clip_roi_max_crop_ratio`]. "
                f"Got {self.crop_controlflow_clip_roi_init_crop_ratio}."
            )
        if self.crop_controlflow_clip_stage1_global_steps < 0:
            raise ValueError(
                "`crop_controlflow_clip_stage1_global_steps` must be non-negative. "
                f"Got {self.crop_controlflow_clip_stage1_global_steps}."
            )
        if self.crop_controlflow_clip_stage2_warmup_steps < 0:
            raise ValueError(
                "`crop_controlflow_clip_stage2_warmup_steps` must be non-negative. "
                f"Got {self.crop_controlflow_clip_stage2_warmup_steps}."
            )
        if self.crop_controlflow_clip_stage2_global_loss_weight < 0.0:
            raise ValueError(
                "`crop_controlflow_clip_stage2_global_loss_weight` must be non-negative. "
                f"Got {self.crop_controlflow_clip_stage2_global_loss_weight}."
            )
        if self.crop_controlflow_clip_stage3_global_loss_weight < 0.0:
            raise ValueError(
                "`crop_controlflow_clip_stage3_global_loss_weight` must be non-negative. "
                f"Got {self.crop_controlflow_clip_stage3_global_loss_weight}."
            )
        if self.crop_controlflow_clip_stage2_train_last_n_velocity_layers < 0:
            raise ValueError(
                "`crop_controlflow_clip_stage2_train_last_n_velocity_layers` must be non-negative. "
                f"Got {self.crop_controlflow_clip_stage2_train_last_n_velocity_layers}."
            )
