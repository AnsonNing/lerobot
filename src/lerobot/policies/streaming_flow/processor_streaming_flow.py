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
from typing import Any

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    ObservationProcessorStep,
    PolicyAction,
    PolicyActionProcessorStep,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import ACTION, POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_streaming_flow import StreamingFlowConfig


def _stats_with_notebook_action_bounds(
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None,
    action_dim: int,
    action_normalization_mode: str,
) -> dict[str, dict[str, torch.Tensor]] | None:
    if dataset_stats is None:
        return None

    action_stats = dataset_stats.get(ACTION)
    if action_stats is None or "min" not in action_stats or "max" not in action_stats:
        return dataset_stats

    if action_normalization_mode == "per_dim":
        return dataset_stats

    stats = dict(dataset_stats)
    action_min = torch.as_tensor(action_stats["min"]).detach().float()
    action_max = torch.as_tensor(action_stats["max"]).detach().float()
    scalar_min = action_min.min().expand(action_dim).clone()
    scalar_max = action_max.max().expand(action_dim).clone()
    stats[ACTION] = dict(action_stats)
    stats[ACTION]["min"] = scalar_min
    stats[ACTION]["max"] = scalar_max
    return stats


@ProcessorStepRegistry.register(name="streaming_flow_action_clip")
@dataclass
class StreamingFlowActionClipProcessorStep(PolicyActionProcessorStep):
    action_min: list[float]
    action_max: list[float]

    def get_config(self) -> dict[str, Any]:
        return {"action_min": self.action_min, "action_max": self.action_max}

    def action(self, action: PolicyAction) -> PolicyAction:
        action_min = torch.as_tensor(self.action_min, device=action.device, dtype=action.dtype)
        action_max = torch.as_tensor(self.action_max, device=action.device, dtype=action.dtype)
        return torch.clamp(action, min=action_min, max=action_max)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="streaming_flow_visual_obs")
@dataclass
class StreamingFlowVisualObservationProcessorStep(ObservationProcessorStep):
    image_keys: tuple[str, ...]

    def get_config(self) -> dict[str, Any]:
        return {"image_keys": list(self.image_keys)}

    def _process_image(self, image) -> torch.Tensor:
        tensor = torch.as_tensor(image)

        if tensor.ndim == 3 and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1).contiguous()
        elif tensor.ndim == 4 and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(0, 3, 1, 2).contiguous()
        elif tensor.ndim == 5 and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(0, 1, 4, 2, 3).contiguous()
        elif tensor.ndim == 6 and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(0, 1, 2, 5, 3, 4).contiguous()

        if tensor.dtype == torch.uint8:
            tensor = tensor.float() / 255.0
        elif not tensor.dtype.is_floating_point:
            tensor = tensor.float()

        return tensor

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        processed_observation = dict(observation)
        for key in self.image_keys:
            if key in processed_observation:
                processed_observation[key] = self._process_image(processed_observation[key])
        return processed_observation

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_streaming_flow_pre_post_processors(
    config: StreamingFlowConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    dataset_stats = _stats_with_notebook_action_bounds(
        dataset_stats,
        action_dim=config.action_feature.shape[0],
        action_normalization_mode=config.action_normalization_mode,
    )
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        StreamingFlowVisualObservationProcessorStep(image_keys=tuple(config.image_features.keys())),
        AddBatchDimensionProcessorStep(),
    ]
    if config.sfp_use_clip_text_conditioning:
        input_steps.append(
            TokenizerProcessorStep(
                tokenizer_name=config.text_encoder_name,
                padding=config.tokenizer_padding,
                padding_side=config.tokenizer_padding_side,
                max_length=config.tokenizer_max_length,
                truncation=config.tokenizer_truncation,
            )
        )
    input_steps.extend(
        [
            DeviceProcessorStep(device=config.device),
            NormalizerProcessorStep(
                features={**config.input_features, **config.output_features},
                norm_map=config.normalization_mapping,
                stats=dataset_stats,
            ),
        ]
    )
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]
    action_stats = dataset_stats.get(ACTION) if dataset_stats is not None else None
    if action_stats is not None and "min" in action_stats and "max" in action_stats:
        output_steps.append(
            StreamingFlowActionClipProcessorStep(
                action_min=torch.as_tensor(action_stats["min"]).detach().cpu().tolist(),
                action_max=torch.as_tensor(action_stats["max"]).detach().cpu().tolist(),
            )
        )
    output_steps.append(DeviceProcessorStep(device="cpu"))

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
