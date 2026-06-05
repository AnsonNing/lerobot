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
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_diffusion import DiffusionV2Config


@ProcessorStepRegistry.register(name="diffusion_v2_visual_obs")
@dataclass
class DiffusionV2VisualObservationProcessorStep(ObservationProcessorStep):
    """Convert environment camera frames to channel-first float tensors for CLIP."""

    image_keys: tuple[str, ...]

    def get_config(self) -> dict[str, Any]:
        return {"image_keys": list(self.image_keys)}

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        processed_observation = dict(observation)
        for key in self.image_keys:
            if key not in processed_observation:
                continue
            tensor = torch.as_tensor(processed_observation[key])
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
            processed_observation[key] = tensor
        return processed_observation

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_diffusion_v2_pre_post_processors(
    config: DiffusionV2Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        DiffusionV2VisualObservationProcessorStep(image_keys=tuple(config.image_features.keys())),
        AddBatchDimensionProcessorStep(),
        TokenizerProcessorStep(
            tokenizer_name=config.text_encoder_name,
            padding=config.tokenizer_padding,
            padding_side=config.tokenizer_padding_side,
            max_length=config.tokenizer_max_length,
            truncation=config.tokenizer_truncation,
        ),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]
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
