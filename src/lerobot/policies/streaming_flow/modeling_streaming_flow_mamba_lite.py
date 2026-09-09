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

"""LeRobot Streaming Flow v2 frontend with a compact Mamba3 velocity expert."""

from collections import deque
from copy import deepcopy

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from ..pretrained import PreTrainedPolicy
from ..utils import populate_queues
from .configuration_streaming_flow import StreamingFlowMambaLiteConfig
from .modeling_streaming_flow_mamba import (
    StreamingFlowMambaModel,
    StreamingFlowMambaVelocityModel,
)
from .modeling_streaming_flow_v2 import (
    StreamingFlowModel as StreamingFlowV2Model,
    StreamingFlowPolicy as StreamingFlowV2Policy,
    TemporalImageEncoder,
)

TASK_INDEX = "task_index"


class StreamingFlowMambaLiteModel(StreamingFlowV2Model):
    """v2 temporal ResNet conditioning and cached Mamba3 ControlFlow dynamics."""

    # Reuse the policy-neutral Mamba trajectory/history algorithms without
    # inheriting the v5 CLIP model hierarchy.
    _empty_history = StreamingFlowMambaModel._empty_history
    _generated_training_history = StreamingFlowMambaModel._generated_training_history
    _training_history = StreamingFlowMambaModel._training_history
    integrate_actions = StreamingFlowMambaModel.integrate_actions
    compute_loss = StreamingFlowMambaModel.compute_loss

    def __init__(self, config: StreamingFlowMambaLiteConfig):
        nn.Module.__init__(self)
        self.config = config
        token_dim = config.transformer_hidden_dim
        global_cond_dim = 0

        num_cameras = len(config.image_features)
        if config.use_separate_rgb_encoder_per_camera:
            self.rgb_encoder = nn.ModuleList([TemporalImageEncoder(config) for _ in range(num_cameras)])
            image_feature_dim = self.rgb_encoder[0].feature_dim
        else:
            self.rgb_encoder = TemporalImageEncoder(config)
            image_feature_dim = self.rgb_encoder.feature_dim
        global_cond_dim += num_cameras * image_feature_dim
        self.image_token_proj = nn.Sequential(
            nn.LayerNorm(image_feature_dim),
            nn.Linear(image_feature_dim, token_dim),
        )

        if config.robot_state_feature is not None:
            state_dim = config.robot_state_feature.shape[0]
            global_cond_dim += config.n_obs_steps * state_dim
            self.state_token_proj = nn.Sequential(
                nn.LayerNorm(state_dim),
                nn.Linear(state_dim, token_dim),
            )
        else:
            self.state_token_proj = None

        if config.lite_use_task_embedding:
            self.task_embedding = nn.Embedding(
                config.lite_num_tasks,
                config.lite_task_embedding_dim,
            )
            self.task_token_proj = nn.Sequential(
                nn.LayerNorm(config.lite_task_embedding_dim),
                nn.Linear(config.lite_task_embedding_dim, token_dim),
            )
            global_cond_dim += config.lite_task_embedding_dim
        else:
            self.task_embedding = None
            self.task_token_proj = None

        self.context_type_embedding = nn.Parameter(torch.zeros(1, 3, token_dim))
        nn.init.normal_(self.context_type_embedding, std=0.02)
        self.text_encoder = None
        self.global_cond_dim = global_cond_dim
        self.velocity_model = StreamingFlowMambaVelocityModel(config, global_cond_dim)

    def _encode_images(self, images: Tensor) -> Tensor:
        """Encode ``(B, S, N, C, H, W)`` into one temporal feature per camera."""
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
        expected_cameras = len(self.config.image_features)
        if images.shape[2] != expected_cameras:
            raise ValueError(
                f"`{OBS_IMAGES}` must contain {expected_cameras} cameras. Got shape {tuple(images.shape)}."
            )

        if self.config.use_separate_rgb_encoder_per_camera:
            images_per_camera = einops.rearrange(images, "b s n c h w -> n b s c h w")
            camera_features = [
                encoder(camera_images)
                for encoder, camera_images in zip(
                    self.rgb_encoder,
                    images_per_camera,
                    strict=True,
                )
            ]
            return torch.stack(camera_features, dim=1)

        flat_images = einops.rearrange(images, "b s n c h w -> (b n) s c h w")
        image_features = self.rgb_encoder(flat_images)
        return einops.rearrange(
            image_features,
            "(b n) f -> b n f",
            b=images.shape[0],
            n=images.shape[2],
        )

    def _task_features(self, batch: dict[str, Tensor], batch_size: int) -> Tensor | None:
        if self.task_embedding is None:
            return None
        if TASK_INDEX not in batch:
            if self.training and self.config.require_task_index_during_training:
                raise ValueError(
                    f"Multi-task v2 training requires `{TASK_INDEX}` in the batch. "
                    "Disable `lite_use_task_embedding` for a task-agnostic model."
                )
            task_index = torch.zeros(batch_size, dtype=torch.long)
        else:
            task_index = batch[TASK_INDEX].long()
            if task_index.ndim == 0:
                task_index = task_index.unsqueeze(0)
            elif task_index.ndim > 1:
                task_index = task_index.reshape(task_index.shape[0], -1)[:, -1]
            if task_index.shape[0] != batch_size:
                raise ValueError("Task index and observation batch sizes do not match.")
            if task_index.device.type == "cpu" and (
                torch.any(task_index < 0) or torch.any(task_index >= self.config.lite_num_tasks)
            ):
                raise ValueError(f"`{TASK_INDEX}` values must lie in [0, {self.config.lite_num_tasks}).")
        task_index = task_index.to(device=self.context_type_embedding.device)
        return self.task_embedding(task_index)

    def _prepare_token_conditioning(
        self,
        batch: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if OBS_IMAGES not in batch:
            raise ValueError(f"Missing `{OBS_IMAGES}` in batch. Available keys: {list(batch)}")

        cond_features: list[Tensor] = []
        context_tokens: list[Tensor] = []
        context_masks: list[Tensor] = []
        camera_features = self._encode_images(batch[OBS_IMAGES])
        batch_size = camera_features.shape[0]
        cond_features.append(camera_features.flatten(start_dim=1))
        image_tokens = self.image_token_proj(camera_features.float()) + self.context_type_embedding[:, 0:1]
        context_tokens.append(image_tokens)
        context_masks.append(torch.ones(image_tokens.shape[:2], dtype=torch.bool, device=image_tokens.device))

        if self.state_token_proj is not None:
            if OBS_STATE not in batch:
                raise ValueError(f"Missing `{OBS_STATE}` in batch. Available keys: {list(batch)}")
            state = batch[OBS_STATE]
            if state.ndim == 2:
                state = state.unsqueeze(1)
            if state.shape[0] != batch_size or state.shape[1] != self.config.n_obs_steps:
                raise ValueError(
                    f"`{OBS_STATE}` must have batch size {batch_size} and "
                    f"{self.config.n_obs_steps} observation steps. Got {tuple(state.shape)}."
                )
            cond_features.append(state.flatten(start_dim=1))
            state_tokens = self.state_token_proj(state.float()) + self.context_type_embedding[:, 1:2]
            context_tokens.append(state_tokens)
            context_masks.append(
                torch.ones(state_tokens.shape[:2], dtype=torch.bool, device=state_tokens.device)
            )

        task_features = self._task_features(batch, batch_size)
        if task_features is not None:
            cond_features.append(task_features)
            task_tokens = (
                self.task_token_proj(task_features.float()).unsqueeze(1) + self.context_type_embedding[:, 2:3]
            )
            context_tokens.append(task_tokens)
            context_masks.append(
                torch.ones(task_tokens.shape[:2], dtype=torch.bool, device=task_tokens.device)
            )

        raw_condition = torch.cat(cond_features, dim=-1)
        normalized_condition = (
            F.normalize(raw_condition, dim=-1) if raw_condition.shape[-1] > 1 else raw_condition
        )
        return (
            raw_condition,
            normalized_condition,
            torch.cat(context_tokens, dim=1),
            torch.cat(context_masks, dim=1),
        )

    @torch.no_grad()
    def generate_actions(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
    ) -> Tensor:
        """Generate one LeRobot action chunk; optional noise supplies its initial state."""
        init_action = None
        if noise is not None:
            if noise.ndim == 2:
                init_action = noise.unsqueeze(1)
            elif noise.ndim == 3:
                init_action = noise[:, :1]
            else:
                raise ValueError(f"`noise` must have shape (B, A) or (B, H, A). Got {tuple(noise.shape)}.")
        actions, _, _ = self.integrate_actions(batch, init_action=init_action)
        return actions


class StreamingFlowMambaLitePolicy(StreamingFlowV2Policy):
    """Streaming Flow v2 policy wrapper with executed-action cross-chunk memory."""

    config_class = StreamingFlowMambaLiteConfig
    name = "streaming_flow_mamba_lite"

    def __init__(self, config: StreamingFlowMambaLiteConfig, **kwargs):
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.register_buffer("_action_min", torch.empty(0), persistent=True)
        self.register_buffer("_action_max", torch.empty(0), persistent=True)
        self.register_buffer("_ema_step", torch.zeros((), dtype=torch.long), persistent=True)
        self._init_normalization_buffers(kwargs.get("dataset_stats"))
        self.model = StreamingFlowMambaLiteModel(config)
        self.ema_model = deepcopy(self.model) if config.use_ema else None
        if self.ema_model is not None:
            self.ema_model.requires_grad_(False)
        self.reset()

    def get_optim_params(self):
        return (parameter for parameter in self.model.parameters() if parameter.requires_grad)

    def update(self) -> None:
        """EMA all lite modules, including state/task token projections."""
        if self.ema_model is None:
            return
        with torch.no_grad():
            self._ema_step += 1
            decay = self._get_ema_decay(int(self._ema_step.item()))
            one_minus_decay = 1.0 - decay
            for ema_param, param in zip(
                self.ema_model.parameters(),
                self.model.parameters(),
                strict=True,
            ):
                if param.requires_grad:
                    ema_param.lerp_(param.detach().to(dtype=ema_param.dtype), one_minus_decay)
                else:
                    ema_param.copy_(param.detach().to(dtype=ema_param.dtype))
            for ema_buffer, buffer in zip(
                self.ema_model.buffers(),
                self.model.buffers(),
                strict=True,
            ):
                ema_buffer.copy_(buffer)

    def reset(self):
        super().reset()
        self._executed_action_history: deque[Tensor] = deque(maxlen=self.config.previous_tail_len)

    def _previous_tail(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        action_dim = self.config.action_feature.shape[0]
        tail = torch.zeros(
            batch_size,
            self.config.previous_tail_len,
            action_dim,
            device=device,
            dtype=dtype,
        )
        mask = torch.zeros(
            batch_size,
            self.config.previous_tail_len,
            device=device,
            dtype=torch.bool,
        )
        if not self._executed_action_history:
            return tail, mask
        if any(action.shape[0] != batch_size for action in self._executed_action_history):
            self._executed_action_history.clear()
            return tail, mask
        history = torch.stack(list(self._executed_action_history), dim=1).to(
            device=device,
            dtype=dtype,
        )
        history = history[:, -self.config.previous_tail_len :]
        tail[:, -history.shape[1] :] = history
        mask[:, -history.shape[1] :] = True
        return tail, mask

    @torch.no_grad()
    def _predict_action_chunk_and_state(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, float | str]]:
        queued_batch = self._queued_batch()
        if TASK_INDEX in batch:
            queued_batch[TASK_INDEX] = batch[TASK_INDEX]
        init_source = "prev_action_state"
        init_action = self._prev_action_state
        if init_action is None:
            init_action, init_source = self._initial_rollout_action(queued_batch)
        if noise is not None:
            init_source = "provided_noise_first_action"
            if noise.ndim == 2:
                init_action = noise.unsqueeze(1)
            elif noise.ndim == 3:
                init_action = noise[:, :1]
            else:
                raise ValueError(f"`noise` must have shape (B, A) or (B, H, A). Got {tuple(noise.shape)}.")
        if init_action is None:
            raise RuntimeError("StreamingFlowMambaLitePolicy could not construct an initial rollout action.")
        previous_tail, history_mask = self._previous_tail(
            init_action.shape[0],
            init_action.device,
            init_action.dtype,
        )
        chunk, final_action, info = self._rollout_model().integrate_actions(
            queued_batch,
            init_action=init_action,
            previous_tail=previous_tail,
            history_mask=history_mask,
        )
        info = dict(info)
        info["init_source"] = init_source
        info["history_source"] = "executed_policy_actions"
        return chunk, final_action, info

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        batch = self._prepare_image_batch(batch)
        self._queues = populate_queues(self._queues, batch)
        self._left_pad_observation_queues()
        actions, final_action, info = self._predict_action_chunk_and_state(batch, noise=noise)
        self._prev_action_state = final_action
        for action in actions.transpose(0, 1):
            self._executed_action_history.append(action.detach())
        self._record_rollout_info(info)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        if ACTION in batch:
            batch = dict(batch)
            batch.pop(ACTION)
        batch = self._prepare_image_batch(batch)
        self._queues = populate_queues(self._queues, batch)
        self._left_pad_observation_queues()
        if len(self._queues[ACTION]) == 0:
            actions, final_action, info = self._predict_action_chunk_and_state(batch, noise=noise)
            self._prev_action_state = final_action
            self._record_rollout_info(info)
            self._queues[ACTION].extend(actions.transpose(0, 1))
        action = self._queues[ACTION].popleft()
        self._executed_action_history.append(action.detach())
        return action

# Conventional alias used by other Streaming Flow modules.
StreamingFlowPolicy = StreamingFlowMambaLitePolicy


__all__ = [
    "StreamingFlowMambaLiteConfig",
    "StreamingFlowMambaLiteModel",
    "StreamingFlowMambaLitePolicy",
    "StreamingFlowPolicy",
]
