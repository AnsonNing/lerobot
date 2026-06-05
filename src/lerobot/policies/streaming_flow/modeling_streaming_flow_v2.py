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
from collections import deque
from collections.abc import Callable
from copy import deepcopy

import einops
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from ..pretrained import PreTrainedPolicy
from ..utils import get_output_shape, populate_queues
from .configuration_streaming_flow import StreamingFlowConfig


class StreamingFlowPolicy(PreTrainedPolicy):
    config_class = StreamingFlowConfig
    name = "streaming_flow"
    NOTEBOOK_INITIAL_ACTION = (256.0, 256.0)

    def __init__(self, config: StreamingFlowConfig, **kwargs):
        super().__init__(config)
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

    def get_optim_params(self) -> dict:
        return self.model.parameters()

    def update(self) -> None:
        if self.ema_model is None:
            return

        with torch.no_grad():
            self._ema_step += 1
            decay = self._get_ema_decay(int(self._ema_step.item()))
            one_minus_decay = 1.0 - decay
            for ema_param, param in zip(
                self.ema_model.velocity_model.parameters(),
                self.model.velocity_model.parameters(),
                strict=True,
            ):
                if param.requires_grad:
                    ema_param.lerp_(param.detach().to(dtype=ema_param.dtype), one_minus_decay)
                else:
                    ema_param.copy_(param.detach().to(dtype=ema_param.dtype))

            for ema_buffer, buffer in zip(
                self.ema_model.velocity_model.buffers(),
                self.model.velocity_model.buffers(),
                strict=True,
            ):
                ema_buffer.copy_(buffer)

            if self.ema_model.rgb_encoder is not None:
                self.ema_model.rgb_encoder.load_state_dict(self.model.rgb_encoder.state_dict())

    def _get_ema_decay(self, optimization_step: int) -> float:
        step = max(0, optimization_step - self.config.ema_update_after_step - 1)
        if step <= 0:
            return 0.0

        cur_decay = (1.0 + step) / (10.0 + step)
        cur_decay = min(cur_decay, self.config.ema_decay)
        cur_decay = max(cur_decay, self.config.ema_min_decay)
        return cur_decay

    def _rollout_model(self) -> "StreamingFlowModel":
        if not self.training and self.ema_model is not None:
            return self.ema_model
        return self.model

    def reset(self):
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}
        if self.config.robot_state_feature:
            self._queues[OBS_STATE] = deque(maxlen=self.config.n_obs_steps)
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        self._prev_action_state = None
        self._rollout_chunk_index = -1
        self._last_rollout_info: dict[str, float | int | str] | None = None

    def get_rollout_info(self) -> dict[str, float | int | str] | None:
        """Return diagnostics for the action chunk currently being executed."""
        return self._last_rollout_info

    def _record_rollout_info(self, info: dict[str, float | str]) -> None:
        self._rollout_chunk_index += 1
        self._last_rollout_info = {**info, "chunk_index": self._rollout_chunk_index}

    def _init_normalization_buffers(self, dataset_stats: dict[str, dict[str, Tensor]] | None) -> None:
        if self.config.action_feature is not None:
            action_dim = self.config.action_feature.shape[0]
            self._action_min = torch.zeros(action_dim, dtype=torch.float32)
            self._action_max = torch.ones(action_dim, dtype=torch.float32)

        if dataset_stats is None:
            return

        action_stats = dataset_stats.get(ACTION)
        if action_stats is not None and "min" in action_stats and "max" in action_stats:
            action_min = torch.as_tensor(action_stats["min"]).detach().float()
            action_max = torch.as_tensor(action_stats["max"]).detach().float()
            self._action_min = action_min.min().expand_as(self._action_min).clone()
            self._action_max = action_max.max().expand_as(self._action_max).clone()

    def _has_action_stats(self) -> bool:
        if self._action_min.numel() == 0 or self._action_max.numel() == 0:
            return False
        return not (
            torch.allclose(self._action_min, torch.zeros_like(self._action_min))
            and torch.allclose(self._action_max, torch.ones_like(self._action_max))
        )

    def _normalize_action(self, action: Tensor) -> Tensor:
        if self._action_min.numel() == 0 or self._action_max.numel() == 0:
            return action

        action_min = self._action_min.to(device=action.device, dtype=action.dtype)
        action_max = self._action_max.to(device=action.device, dtype=action.dtype)
        denom = torch.where(
            action_max == action_min,
            torch.ones_like(action_max),
            action_max - action_min,
        )
        return 2 * (action - action_min) / denom - 1

    def _unnormalize_action(self, action: Tensor) -> Tensor:
        """Convert normalized action in [-1, 1] back to raw environment action space.

        This is only used for rollout diagnostics by default. Do not return this
        from select_action unless your eval loop is confirmed to bypass LeRobot's
        output unnormalization processor.
        """
        if self._action_min.numel() == 0 or self._action_max.numel() == 0:
            return action

        action_min = self._action_min.to(device=action.device, dtype=action.dtype)
        action_max = self._action_max.to(device=action.device, dtype=action.dtype)
        denom = torch.where(
            action_max == action_min,
            torch.ones_like(action_max),
            action_max - action_min,
        )
        return (action + 1) / 2 * denom + action_min

    def _clip_raw_action(self, action: Tensor) -> Tensor:
        if self._action_min.numel() == 0 or self._action_max.numel() == 0:
            return action

        action_min = self._action_min.to(device=action.device, dtype=action.dtype)
        action_max = self._action_max.to(device=action.device, dtype=action.dtype)
        return torch.minimum(torch.maximum(action, action_min), action_max)

    def _left_pad_observation_queues(self) -> None:
        """Make rollout queue initialization match the notebook: [obs0, obs0, ...].

        LeRobot's populate_queues may leave the queue shorter than n_obs_steps at
        episode start depending on the caller. Padding here removes that ambiguity.
        """
        for key, queue in self._queues.items():
            if key == ACTION or len(queue) == 0:
                continue
            while len(queue) < queue.maxlen:
                queue.appendleft(queue[0].clone())

    def _normalized_raw_rollout_action(self, raw_action: Tensor) -> Tensor:
        if self._has_action_stats():
            return self._normalize_action(raw_action)
        return raw_action

    def _initial_rollout_action(self, queued_batch: dict[str, Tensor]) -> tuple[Tensor | None, str]:
        if self.config.action_feature is None:
            return None, "missing_action_feature"
        if OBS_IMAGES not in queued_batch:
            return None, "missing_images"

        images = queued_batch[OBS_IMAGES]
        action_dim = self.config.action_feature.shape[0]
        init_action = torch.zeros(
            (images.shape[0], 1, action_dim),
            device=images.device,
            dtype=images.dtype,
        )
        mode = self.config.rollout_initial_action_mode

        if mode == "state":
            if OBS_STATE not in queued_batch:
                raise ValueError(
                    "`rollout_initial_action_mode='state'` requires `observation.state` in the rollout batch."
                )
            state = queued_batch[OBS_STATE]
            if state.shape[-1] < action_dim:
                raise ValueError(
                    "`rollout_initial_action_mode='state'` requires state to contain at least the action "
                    f"dimensions. Got state shape {tuple(state.shape)} and action_dim={action_dim}."
                )
            return state[:, -1:, :action_dim].to(device=images.device, dtype=images.dtype), "state_action"

        if mode == "constant":
            raw_initial_action = self.config.rollout_initial_action
            source = "configured_constant"
        elif mode == "auto" and action_dim == len(self.NOTEBOOK_INITIAL_ACTION):
            raw_initial_action = self.NOTEBOOK_INITIAL_ACTION
            source = "notebook_fixed_center"
        else:
            raw_initial_action = (0.0,) * action_dim
            source = "zero_action"

        fixed_action = torch.as_tensor(raw_initial_action, device=images.device, dtype=images.dtype)
        if fixed_action.numel() != action_dim:
            raise ValueError(
                "Configured rollout initial action dimension does not match policy action dimension. "
                f"Got {fixed_action.numel()} values for action_dim={action_dim}."
            )

        init_action[:, 0, :] = fixed_action
        if source == "notebook_fixed_center" and not self._has_action_stats() and fixed_action.abs().max() > 1.0:
            return torch.zeros_like(init_action), "zero_action"
        return self._normalized_raw_rollout_action(init_action), source

    def _prepare_image_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if not self.config.image_features:
            return batch

        batch = dict(batch)
        image_tensors = []
        for key in self.config.image_features:
            if key not in batch:
                continue
            image = batch[key]
            image_tensors.append(image)

        if not image_tensors:
            raise ValueError(
                "StreamingFlowPolicy expected at least one image feature in the batch, "
                f"but got keys: {list(batch)}"
            )

        image_ndims = {image.ndim for image in image_tensors}
        if len(image_ndims) != 1:
            raise ValueError(
                "All image features must have the same rank before they can be stacked. "
                f"Got ranks {sorted(image_ndims)}."
            )

        image_ndim = image_tensors[0].ndim
        if image_ndim == 4:
            # In rollout we receive one observation per call, so we keep the camera axis only and
            # let the observation queue provide the temporal dimension.
            batch[OBS_IMAGES] = torch.stack(image_tensors, dim=1)
        elif image_ndim == 5:
            # Training batches already contain the observation-step axis.
            batch[OBS_IMAGES] = torch.stack(image_tensors, dim=2)
        elif image_ndim == 6:
            batch[OBS_IMAGES] = torch.cat(image_tensors, dim=2)
        else:
            raise ValueError(
                "StreamingFlowPolicy expects image tensors with shape (B, C, H, W) for rollout or "
                f"(B, S, C, H, W) for training. Got rank {image_ndim}."
            )
        return batch

    def _queued_batch(self) -> dict[str, Tensor]:
        return {
            key: torch.stack(list(queue), dim=1)
            for key, queue in self._queues.items()
            if key != ACTION and len(queue) > 0
        }

    @torch.no_grad()
    def _predict_action_chunk_and_state(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, float]]:
        del batch
        del noise
        queued_batch = self._queued_batch()

        # The first chunk uses the configured dataset-appropriate initial action;
        # later chunks continue from the previous final action state.
        init_source = "prev_action_state"
        init_action = self._prev_action_state

        if init_action is None:
            init_action, init_source = self._initial_rollout_action(queued_batch)

        if init_action is None:
            raise RuntimeError(
                "StreamingFlowPolicy could not build an initial action for rollout. "
                f"queued_batch keys={list(queued_batch)}"
            )

        chunk, final_action_state, info = self._rollout_model().integrate_actions(
            queued_batch,
            init_action=init_action,
        )
        info = dict(info)
        info["init_source"] = init_source
        info["init_action_norm_mean_abs"] = float(init_action.abs().mean().detach().cpu())
        return chunk, final_action_state, info

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        batch = self._prepare_image_batch(batch)
        self._queues = populate_queues(self._queues, batch)
        self._left_pad_observation_queues()

        chunk, final_action_state, info = self._predict_action_chunk_and_state(batch, noise=noise)

        # Important: some rollout/eval code paths call predict_action_chunk instead
        # of select_action. Keep streaming continuity in that path as well.
        self._prev_action_state = final_action_state
        self._record_rollout_info(info)
        return chunk

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        if ACTION in batch:
            batch = dict(batch)
            batch.pop(ACTION)

        batch = self._prepare_image_batch(batch)
        self._queues = populate_queues(self._queues, batch)
        self._left_pad_observation_queues()

        if len(self._queues[ACTION]) == 0:
            actions, final_action_state, info = self._predict_action_chunk_and_state(batch, noise=noise)
            self._prev_action_state = final_action_state
            self._record_rollout_info(info)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()

        # Return normalized action and let LeRobot's postprocessor unnormalize before env.step.
        return action

    def forward(
        self,
        batch: dict[str, Tensor],
        reduction: str = "mean",
    ) -> tuple[Tensor, dict[str, float] | None]:
        batch = self._prepare_image_batch(batch)
        return self.model.compute_loss(batch, reduction=reduction)


class StreamingFlowModel(nn.Module):
    def __init__(self, config: StreamingFlowConfig):
        super().__init__()
        self.config = config

        global_cond_dim = 0
        if config.image_features:
            num_images = len(config.image_features)
            if config.use_separate_rgb_encoder_per_camera:
                self.rgb_encoder = nn.ModuleList(
                    [TemporalImageEncoder(config) for _ in range(num_images)]
                )
                global_cond_dim += num_images * self.rgb_encoder[0].feature_dim
            else:
                self.rgb_encoder = TemporalImageEncoder(config)
                global_cond_dim += num_images * self.rgb_encoder.feature_dim
        else:
            self.rgb_encoder = None

        if config.robot_state_feature is not None:
            global_cond_dim += config.n_obs_steps * config.robot_state_feature.shape[0]

        self.global_cond_dim = global_cond_dim
        self.velocity_model = AdaptiveStreamingFlowUnet(
            config=config,
            global_cond_dim=global_cond_dim,
        )

        if config.compile_model:
            self.velocity_model = torch.compile(self.velocity_model, mode=config.compile_mode)

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        cond_feats = []

        if self.config.robot_state_feature is not None:
            if OBS_STATE not in batch:
                raise ValueError(
                    f"Missing `{OBS_STATE}` in batch. Available keys: {list(batch)}"
                )

            state = batch[OBS_STATE]
            if state.ndim == 2:
                state = state.unsqueeze(1)
            if state.ndim != 3:
                raise ValueError(
                    f"`{OBS_STATE}` must have shape (B, D) or (B, S, D). Got {tuple(state.shape)}."
                )
            if state.shape[1] != self.config.n_obs_steps:
                raise ValueError(
                    f"`{OBS_STATE}` must contain {self.config.n_obs_steps} observation steps. "
                    f"Got shape {tuple(state.shape)}."
                )
            cond_feats.append(state.flatten(start_dim=1))

        if self.config.image_features:
            if OBS_IMAGES not in batch:
                raise ValueError(
                    f"Missing `{OBS_IMAGES}` in batch. Available keys: {list(batch)}"
                )

            images = batch[OBS_IMAGES]
            if images.ndim == 5:
                images = images.unsqueeze(1)
            if images.ndim != 6:
                raise ValueError(
                    f"`{OBS_IMAGES}` must have shape (B, N, C, H, W) or (B, S, N, C, H, W). "
                    f"Got {tuple(images.shape)}."
                )

            # (B, S, N, C, H, W)
            if self.config.use_separate_rgb_encoder_per_camera:
                images_per_camera = einops.rearrange(images, "b s n c h w -> n b s c h w")
                cam_feats = [
                    encoder(camera_images)
                    for encoder, camera_images in zip(self.rgb_encoder, images_per_camera, strict=True)
                ]
                cond_feats.append(torch.cat(cam_feats, dim=-1))
            else:
                flat_images = einops.rearrange(images, "b s n c h w -> (b n) s c h w")
                image_feats = self.rgb_encoder(flat_images)
                image_feats = einops.rearrange(
                    image_feats,
                    "(b n) f -> b (n f)",
                    b=images.shape[0],
                    n=images.shape[2],
                )
                cond_feats.append(image_feats)

        if not cond_feats:
            raise ValueError("StreamingFlowModel received no conditioning features.")

        raw_cond = torch.cat(cond_feats, dim=-1)
        if raw_cond.shape[-1] > 1:
            normalized_cond = F.normalize(raw_cond, dim=-1)
        else:
            normalized_cond = raw_cond
        return raw_cond, normalized_cond

    def _predict_frequency(self, raw_cond: Tensor, clamp: bool) -> tuple[Tensor, Tensor, Tensor]:
        if not self.config.sfp_use_adaptive_freq:
            freq = torch.ones(
                raw_cond.shape[0],
                device=raw_cond.device,
                dtype=torch.float32,
            )
            return freq, freq, freq

        raw_freq = self.velocity_model.granularity_predictor(raw_cond.to(dtype=torch.float32))
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

    def integrate_actions(
        self,
        batch: dict[str, Tensor],
        init_action: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, float]]:
        raw_cond, normalized_cond = self._prepare_global_conditioning(batch)
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
                (action.shape[0],),
                step_idx * dt,
                device=action.device,
                dtype=torch.float32,
            )
            velocity = self.velocity_model(
                sample=action.float(),
                timestep=timestep,
                global_cond=normalized_cond,
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
                "chunk_delta_mean_abs": float((stacked_chunk[:, 1:] - stacked_chunk[:, :-1]).abs().mean().detach().cpu())
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

        raw_cond, normalized_cond = self._prepare_global_conditioning(batch)
        raw_freq, freq, clamped_freq = self._predict_frequency(
            raw_cond, clamp=self.config.sfp_clamp_freq_during_training
        )

        start_idx = min(max(self.config.n_obs_steps - 1, 0), batch[ACTION].shape[1] - 1)
        trajectory = batch[ACTION][:, start_idx:, :]
        num_train_points = max(1, int(getattr(self.config, "sfp_num_train_points", 4)))
        time_shape = (trajectory.shape[0], num_train_points) if num_train_points > 1 else (trajectory.shape[0],)
        time = torch.rand(time_shape, device=trajectory.device, dtype=torch.float32)
        time = time * 0.999 + 0.001

        xi_t, dxi_dt = linearly_interpolate_trajectory(trajectory, time)
        noised_action, target_velocity = sample_cfm_inputs_and_targets(
            xi_t,
            dxi_dt,
            time,
            k=self.config.sfp_k,
            sigma0=self.config.sfp_sigma0,
        )

        if num_train_points > 1:
            batch_size, num_queries, action_dim = noised_action.shape
            flat_noised_action = noised_action.reshape(batch_size * num_queries, action_dim)
            flat_target_velocity = target_velocity.reshape(batch_size * num_queries, action_dim)
            flat_time = time.reshape(batch_size * num_queries)
            flat_cond = normalized_cond.repeat_interleave(num_queries, dim=0)
            flat_freq = freq.repeat_interleave(num_queries, dim=0)
        else:
            num_queries = 1
            flat_noised_action = noised_action
            flat_target_velocity = target_velocity
            flat_time = time
            flat_cond = normalized_cond
            flat_freq = freq

        pred_velocity = self.velocity_model(
            sample=flat_noised_action.unsqueeze(1),
            timestep=flat_time,
            global_cond=flat_cond,
            freq=flat_freq,
        )
        per_sample_loss = F.mse_loss(
            pred_velocity,
            flat_target_velocity.unsqueeze(1),
            reduction="none",
        ).mean(dim=(1, 2)).reshape(trajectory.shape[0], num_queries).mean(dim=1)

        if self.config.sfp_freq_reg_weight > 0.0:
            per_sample_loss = per_sample_loss + self.config.sfp_freq_reg_weight * (freq - 1.0).pow(2)

        if self.config.do_mask_loss_for_padding and "action_is_pad" in batch:
            valid_mask = (~batch["action_is_pad"][:, start_idx:]).all(dim=1)
            per_sample_loss = per_sample_loss * valid_mask.to(dtype=per_sample_loss.dtype)
            denom = valid_mask.sum().clamp_min(1)
            mean_loss = per_sample_loss.sum() / denom
        else:
            mean_loss = per_sample_loss.mean()

        output_dict = {
            "loss": float(mean_loss.detach().item()),
            "pred_freq": float(freq.mean().detach().item()),
            "raw_pred_freq": float(raw_freq.mean().detach().item()),
            "clamped_pred_freq": float(clamped_freq.mean().detach().item()),
            "sfp_num_train_points": float(num_train_points),
        }
        if reduction == "none":
            return per_sample_loss, output_dict
        return mean_loss, output_dict


class TemporalImageEncoder(nn.Module):
    def __init__(self, config: StreamingFlowConfig):
        super().__init__()
        self.config = config

        if config.resize_shape is not None:
            self.resize = torchvision.transforms.Resize(config.resize_shape)
        else:
            self.resize = None

        if config.crop_shape is not None:
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
            self.do_crop = True
        else:
            self.do_crop = False

        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            self.backbone = replace_submodules(
                root_module=self.backbone,
                predicate=lambda module: isinstance(module, nn.BatchNorm2d),
                func=lambda module: nn.GroupNorm(
                    num_groups=max(1, module.num_features // 16),
                    num_channels=module.num_features,
                ),
            )
        if config.freeze_vision_encoder:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

        image_shape = next(iter(config.image_features.values())).shape
        if config.crop_shape is not None:
            dummy_hw = config.crop_shape
        elif config.resize_shape is not None:
            dummy_hw = config.resize_shape
        else:
            dummy_hw = image_shape[1:]
        feature_map_shape = get_output_shape(self.backbone, (1, image_shape[0], *dummy_hw))[1:]

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(config.image_dropout)
        self.feature_dim = config.image_feature_dim
        self.use_imagenet_norm = (
            config.sfp_use_imagenet_visual_norm
            and str(config.normalization_mapping.get("VISUAL", "IDENTITY")).endswith("IDENTITY")
        )
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 3, 1, 1),
        )

        pooled_dim = feature_map_shape[0]
        if config.n_obs_steps >= 3:
            proj_in_dim = pooled_dim * 3
        else:
            proj_in_dim = pooled_dim * 2
        self.proj = nn.Linear(proj_in_dim, self.feature_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_vision_encoder:
            self.backbone.eval()
        return self

    def forward(self, obs: Tensor) -> Tensor:
        bsize, obs_steps, channels, height, width = obs.shape

        if self.use_imagenet_norm:
            obs = (obs.float() - self.imagenet_mean.to(dtype=obs.dtype)) / self.imagenet_std.to(
                dtype=obs.dtype
            )

        obs = obs.reshape(bsize * obs_steps, channels, height, width)

        if self.resize is not None:
            obs = self.resize(obs)
        if self.do_crop:
            if self.training:
                obs = self.maybe_random_crop(obs)
            else:
                obs = self.center_crop(obs)

        if self.config.freeze_vision_encoder:
            with torch.no_grad():
                features = self.pool(self.backbone(obs)).flatten(1)
        else:
            features = self.pool(self.backbone(obs)).flatten(1)
        features = features.reshape(bsize, obs_steps, -1)

        if obs_steps >= 3:
            velocity = features[:, -1] - features[:, -2]
            acceleration = features[:, -1] - 2 * features[:, -2] + features[:, -3]
            fused = torch.cat([features[:, -1], velocity, acceleration], dim=-1)
        elif obs_steps == 2:
            fused = torch.cat([features[:, -1], features[:, -1] - features[:, 0]], dim=-1)
        else:
            fused = torch.cat([features[:, -1], torch.zeros_like(features[:, -1])], dim=-1)

        return self.dropout(self.proj(fused))


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, scale: float = 1.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        x = x * self.scale
        half_dim = self.dim // 2
        freq = torch.exp(
            torch.arange(half_dim, device=x.device, dtype=x.dtype)
            * -(math.log(10000.0) / (half_dim - 1))
        )
        emb = x[:, None] * freq[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        num_groups = resolve_group_norm_groups(out_channels, n_groups)
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(num_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ConvDownsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class LinearDownsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x.transpose(1, 2)).transpose(1, 2)
        if x.shape[-1] >= 2:
            x = F.avg_pool1d(x, 2, stride=2)
        return x


class LinearUpsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        return self.linear(x.transpose(1, 2)).transpose(1, 2)


class ConditionalResidualBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int,
        n_groups: int,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2),
            nn.Unflatten(-1, (-1, 1)),
        )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor, delta: Tensor | None = None) -> Tensor:
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).reshape(cond.shape[0], 2, self.out_channels, 1)
        scale, bias = embed[:, 0], embed[:, 1]
        out = scale * out + bias
        out = self.blocks[1](out)
        if delta is not None:
            out = out * delta.view(-1, 1, 1)
        return out + self.residual_conv(x)


class StepScalingLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Mish(),
            nn.Linear(dim // 2, 1),
        )
        self.softplus = nn.Softplus()

    def forward(self, embedding: Tensor, freq: Tensor) -> Tensor:
        delta = self.softplus(self.fc(embedding)).squeeze(-1)
        return freq * delta


class GranularityPredictor(nn.Module):
    def __init__(self, cond_dim: int):
        super().__init__()
        hidden_dim = max(128, cond_dim // 4)
        self.net = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.8)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, cond: Tensor) -> Tensor:
        return self.net(cond).squeeze(-1)


class AdaptiveStreamingFlowUnet(nn.Module):
    def __init__(self, config: StreamingFlowConfig, global_cond_dim: int):
        super().__init__()
        self.config = config

        all_dims = [config.action_feature.shape[0], *config.down_dims]
        start_dim = config.down_dims[0]
        cond_dim = config.embedding_dim * 2 + global_cond_dim

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(config.embedding_dim, scale=config.timestep_embedding_scale),
            nn.Linear(config.embedding_dim, config.embedding_dim * 4),
            nn.Mish(),
            nn.Linear(config.embedding_dim * 4, config.embedding_dim),
        )
        self.freq_encoder = nn.Sequential(
            SinusoidalPosEmb(config.embedding_dim, scale=config.frequency_embedding_scale),
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.Mish(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )

        self.granularity_predictor = GranularityPredictor(global_cond_dim)
        self.step_scaling = StepScalingLayer(config.embedding_dim * 2)

        downsample_cls = LinearDownsample1d if config.updownsample_type == "linear" else ConvDownsample1d
        self.down_modules = nn.ModuleList()
        for idx, (dim_in, dim_out) in enumerate(zip(all_dims[:-1], all_dims[1:], strict=True)):
            is_last = idx == len(all_dims) - 2
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1d(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=config.kernel_size,
                            n_groups=config.n_groups,
                        ),
                        ConditionalResidualBlock1d(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=config.kernel_size,
                            n_groups=config.n_groups,
                        ),
                        downsample_cls(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1d(
                    all_dims[-1],
                    all_dims[-1],
                    cond_dim=cond_dim,
                    kernel_size=config.kernel_size,
                    n_groups=config.n_groups,
                ),
                ConditionalResidualBlock1d(
                    all_dims[-1],
                    all_dims[-1],
                    cond_dim=cond_dim,
                    kernel_size=config.kernel_size,
                    n_groups=config.n_groups,
                ),
            ]
        )

        self.up_modules = nn.ModuleList()
        up_pairs = list(zip(all_dims[1:-1][::-1], all_dims[2:][::-1], strict=True))
        for idx, (dim_in, dim_out) in enumerate(up_pairs):
            is_last = idx == len(up_pairs) - 1
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1d(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=config.kernel_size,
                            n_groups=config.n_groups,
                        ),
                        ConditionalResidualBlock1d(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=config.kernel_size,
                            n_groups=config.n_groups,
                        ),
                        LinearUpsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, config.kernel_size, n_groups=config.n_groups),
            nn.Conv1d(start_dim, config.action_feature.shape[0], 1),
        )
        self.register_buffer("_prev_freq", torch.zeros(1))
        self._reinitialize_granularity_predictor()

    def _reinitialize_granularity_predictor(self) -> None:
        for module in self.granularity_predictor.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        sample: Tensor,
        timestep: Tensor,
        global_cond: Tensor,
        freq: Tensor | None = None,
        smooth_freq: bool = False,
    ) -> Tensor:
        x = sample.moveaxis(-1, -2)
        batch_size, _, original_time = x.shape

        timestep = torch.as_tensor(timestep, dtype=torch.float32, device=x.device).view(-1)
        if timestep.numel() == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)
        time_embedding = self.diffusion_step_encoder(timestep)

        if freq is None:
            freq = self.granularity_predictor(global_cond)
        elif freq.ndim > 1:
            freq = freq.squeeze(-1)
        freq = freq.to(device=x.device, dtype=torch.float32)

        if not self.training and smooth_freq:
            if self._prev_freq.shape != freq.shape:
                self._prev_freq = torch.zeros_like(freq)
            freq = 0.7 * self._prev_freq + 0.3 * freq
            self._prev_freq = freq.detach()

        freq_embedding = self.freq_encoder(freq)
        cond_feat = torch.cat([0.5 * time_embedding, 0.5 * freq_embedding, global_cond], dim=-1)
        delta = self.step_scaling(torch.cat([time_embedding, freq_embedding], dim=-1), freq)

        skips = []
        for res1, res2, down in self.down_modules:
            x = res1(x, cond_feat, delta)
            x = res2(x, cond_feat, delta)
            skips.append(x)
            x = down(x)

        for mid in self.mid_modules:
            x = mid(x, cond_feat, delta)

        for res1, res2, up in self.up_modules:
            skip = skips.pop()
            if skip.shape[-1] != x.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
            x = torch.cat((x, skip), dim=1)
            x = res1(x, cond_feat, delta)
            x = res2(x, cond_feat, delta)
            x = up(x)
            if x.shape[-1] > original_time:
                x = x[..., :original_time]

        return self.final_conv(x).moveaxis(-1, -2)


def linearly_interpolate_trajectory(actions: Tensor, time: Tensor) -> tuple[Tensor, Tensor]:
    batch_size, seq_len, action_dim = actions.shape
    if seq_len <= 1:
        if time.ndim == 1:
            zeros = torch.zeros((batch_size, action_dim), dtype=actions.dtype, device=actions.device)
            return actions[:, 0, :], zeros

        xi_t = actions[:, None, 0, :].expand(batch_size, time.shape[1], action_dim)
        zeros = torch.zeros_like(xi_t)
        return xi_t, zeros

    scaled_t = time * (seq_len - 1)
    lower = scaled_t.floor().long().clamp(0, seq_len - 2)
    upper = (lower + 1).clamp(0, seq_len - 1)
    lam = (scaled_t - lower.float()).unsqueeze(-1)

    batch_index = torch.arange(batch_size, device=actions.device)
    if time.ndim == 2:
        batch_index = batch_index[:, None]
    xi_lower = actions[batch_index, lower, :]
    xi_upper = actions[batch_index, upper, :]

    xi_t = xi_lower + lam * (xi_upper - xi_lower)
    dxi_dt = (xi_upper - xi_lower) * (seq_len - 1)
    return xi_t, dxi_dt


def sample_cfm_inputs_and_targets(
    xi_t: Tensor,
    dxi_dt: Tensor,
    time: Tensor,
    k: float,
    sigma0: float,
) -> tuple[Tensor, Tensor]:
    sampled_error = sigma0 * torch.exp(-k * time).unsqueeze(-1) * torch.randn_like(xi_t)
    noised_action = xi_t + sampled_error
    target_velocity = -k * sampled_error + dxi_dt
    return noised_action, target_velocity


def resolve_group_norm_groups(num_channels: int, requested_groups: int) -> int:
    groups = min(requested_groups, num_channels)
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    if predicate(root_module):
        return func(root_module)

    replace_list = [
        name.split(".")
        for name, module in root_module.named_modules(remove_duplicate=True)
        if predicate(module)
    ]
    for *parents, key in replace_list:
        parent_module = root_module if not parents else root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(key)]
            parent_module[int(key)] = func(src_module)
        else:
            src_module = getattr(parent_module, key)
            setattr(parent_module, key, func(src_module))

    assert not any(predicate(module) for _, module in root_module.named_modules(remove_duplicate=True))
    return root_module
