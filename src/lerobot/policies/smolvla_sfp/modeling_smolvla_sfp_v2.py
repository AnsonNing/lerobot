#!/usr/bin/env python

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.utils.import_utils import require_package
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

from ..pretrained import PreTrainedPolicy
from ..rtc.modeling_rtc import RTCProcessor
from ..smolvla.modeling_smolvla import (
    ActionSelectKwargs,
    SmolVLAPolicy as _BaseSmolVLAPolicy,
    VLAFlowMatching,
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    pad_vector,
    resize_with_pad,
)
from .configuration_smolvla_sfp import SmolVLASFPV2Config


class _FiLMContext(nn.Module):
    def __init__(self):
        super().__init__()
        self._pre_gamma = None
        self._pre_beta = None
        self._post_gamma = None
        self._post_beta = None
        self._attn_gate = None
        self._mlp_gate = None

    def set_film(self, pre_gamma, pre_beta, post_gamma, post_beta, attn_gate, mlp_gate):
        self._pre_gamma = pre_gamma
        self._pre_beta = pre_beta
        self._post_gamma = post_gamma
        self._post_beta = post_beta
        self._attn_gate = attn_gate
        self._mlp_gate = mlp_gate

    def clear(self):
        self._pre_gamma = None
        self._pre_beta = None
        self._post_gamma = None
        self._post_beta = None
        self._attn_gate = None
        self._mlp_gate = None

    def get_ln_film(self, slot: str, layer_idx: int):
        if slot == "pre_ln":
            if self._pre_gamma is None or self._pre_beta is None:
                return None, None
            return self._pre_gamma[layer_idx], self._pre_beta[layer_idx]
        if slot == "post_ln":
            if self._post_gamma is None or self._post_beta is None:
                return None, None
            return self._post_gamma[layer_idx], self._post_beta[layer_idx]
        raise ValueError(f"Unknown FiLM slot: {slot}")

    def get_gate(self, slot: str, layer_idx: int):
        if slot == "attn":
            if self._attn_gate is None:
                return None
            return self._attn_gate[layer_idx]
        if slot == "mlp":
            if self._mlp_gate is None:
                return None
            return self._mlp_gate[layer_idx]
        raise ValueError(f"Unknown gate slot: {slot}")


class _FiLMLayerNorm(nn.Module):
    def __init__(self, base: nn.Module, film_ctx: _FiLMContext, slot: str, layer_idx: int):
        super().__init__()
        self.base = base
        self.film_ctx = film_ctx
        self.slot = slot
        self.layer_idx = layer_idx

    def forward(self, x: Tensor):
        y = self.base(x)
        gamma, beta = self.film_ctx.get_ln_film(self.slot, self.layer_idx)
        if gamma is None or beta is None:
            return y
        return (1.0 + gamma.to(dtype=y.dtype)) * y + beta.to(dtype=y.dtype)


class _ResidualGateModule(nn.Module):
    def __init__(self, base: nn.Module, film_ctx: _FiLMContext, slot: str, layer_idx: int):
        super().__init__()
        self.base = base
        self.film_ctx = film_ctx
        self.slot = slot
        self.layer_idx = layer_idx

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base, name)

    def _apply_gate(self, out: Tensor) -> Tensor:
        gate = self.film_ctx.get_gate(self.slot, self.layer_idx)
        if gate is None:
            return out
        return out * gate.to(dtype=out.dtype)

    def forward(self, *args, **kwargs):
        result = self.base(*args, **kwargs)

        if isinstance(result, tuple):
            if len(result) == 0:
                return result
            return (self._apply_gate(result[0]), *result[1:])

        if isinstance(result, list):
            if len(result) == 0:
                return result
            return [self._apply_gate(result[0]), *result[1:]]

        if torch.is_tensor(result):
            return self._apply_gate(result)

        return result


class _DirectFrequencyPredictor(nn.Module):
    def __init__(self, cond_dim: int, config: SmolVLASFPV2Config):
        super().__init__()
        hidden_dim = max(64, cond_dim // 4)
        self.net = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        if config.sfp_reinit_granularity_predictor:
            nn.init.xavier_uniform_(self.net[1].weight, gain=0.8)
            nn.init.zeros_(self.net[1].bias)
            nn.init.zeros_(self.net[3].weight)
            nn.init.constant_(self.net[3].bias, config.sfp_freq_init)

    def forward(self, cond: Tensor) -> Tensor:
        return self.net(cond).squeeze(-1)


class _StepScalingLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, max(1, dim // 2)),
            nn.SiLU(),
            nn.Linear(max(1, dim // 2), 1),
        )
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.constant_(self.fc[-1].bias, math.log(math.expm1(1.0)))

    def forward(self, embedding: Tensor, freq: Tensor) -> Tensor:
        learned_scale = F.softplus(self.fc(embedding)).squeeze(-1)
        return freq * learned_scale


class SmolVLASFPPolicy(_BaseSmolVLAPolicy):
    config_class = SmolVLASFPV2Config
    name = "smolvla_sfp_v2"

    def prepare_images(self, batch):
        images = []
        img_masks = []

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. "
                f"batch={batch.keys()}, image_features={self.config.image_features}"
            )

        used_obs_steps = self.config.n_obs_steps
        template_img = None
        template_mask = None
        per_camera_obs = []

        for key in present_img_keys:
            raw_img = batch[key]
            if raw_img.ndim == 5:
                obs_steps = min(self.config.n_obs_steps, raw_img.shape[1])
                used_obs_steps = min(used_obs_steps, obs_steps)
            else:
                obs_steps = 1
                used_obs_steps = 1
            per_camera_obs.append((key, raw_img, obs_steps))

        processed_per_camera_obs = []

        for key, raw_img, obs_steps in per_camera_obs:
            if raw_img.ndim == 5:
                obs_imgs = raw_img[:, -used_obs_steps:, :, :, :]
            else:
                obs_imgs = raw_img[:, None, :, :, :]

            bsize = raw_img.shape[0]
            device = raw_img.device

            if f"{key}_padding_mask" in batch:
                raw_mask = batch[f"{key}_padding_mask"].bool()
                if raw_mask.ndim == 2:
                    obs_masks = raw_mask[:, -obs_steps:]
                else:
                    obs_masks = raw_mask[:, None].expand(bsize, obs_steps)
            else:
                obs_masks = torch.ones((bsize, obs_steps), dtype=torch.bool, device=device)

            obs_imgs = obs_imgs[:, -used_obs_steps:, :, :, :]
            obs_masks = obs_masks[:, -used_obs_steps:]

            processed_per_camera_obs.append((key, obs_imgs, obs_masks))

        for obs_idx in range(used_obs_steps):
            for _key, obs_imgs, obs_masks in processed_per_camera_obs:
                img = obs_imgs[:, obs_idx, :, :, :]

                if self.config.resize_imgs_with_padding is not None:
                    img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

                img = img * 2.0 - 1.0
                mask = obs_masks[:, obs_idx]

                images.append(img)
                img_masks.append(mask)

                template_img = img
                template_mask = mask

        if template_img is None or template_mask is None:
            raise RuntimeError("Failed to build image inputs for SmolVLASFPPolicy.")

        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break

            for _ in range(used_obs_steps):
                images.append(torch.ones_like(template_img) * -1)
                img_masks.append(torch.zeros_like(template_mask))

        return images, img_masks

    def prepare_state(self, batch):
        state = batch[OBS_STATE]

        if state.ndim > 2:
            obs_steps = min(self.config.n_obs_steps, state.shape[1])
            state = state[:, -obs_steps:, :]

        state = pad_vector(state, self.config.max_state_dim)
        return state

    def __init__(self, config: SmolVLASFPV2Config, **kwargs):
        require_package("transformers", extra="smolvla")
        PreTrainedPolicy.__init__(self, config)

        config.validate_features()
        self.config = config
        self.init_rtc_processor()

        self.model = VLASFP(config, rtc_processor=self.rtc_processor)
        self.reset()

    def get_optim_params(self) -> list[dict]:
        vlm_params = []
        non_vlm_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("vlm_with_expert.vlm."):
                vlm_params.append(param)
            else:
                non_vlm_params.append(param)

        groups: list[dict] = [{"params": non_vlm_params}]
        if vlm_params:
            groups.append(
                {
                    "params": vlm_params,
                    "lr": self.config.optimizer_lr * self.config.vlm_lr_multiplier,
                }
            )
        return groups

    def reset(self):
        super().reset()
        self._prev_action_state = None

    def _get_action_chunk(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        **kwargs: ActionSelectKwargs,
    ) -> Tensor:
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        rollout_init = noise
        if rollout_init is None and self._prev_action_state is not None:
            if self._prev_action_state.shape[0] == state.shape[0]:
                rollout_init = self._prev_action_state

        actions = self.model.sample_actions(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            noise=rollout_init,
            **kwargs,
        )

        self._prev_action_state = actions[:, -1, :].detach()

        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    def forward(
        self,
        batch: dict[str, Tensor],
        noise=None,
        time=None,
        reduction: str = "mean",
    ) -> dict[str, Tensor]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")

        loss_dict = {}

        losses = self.model.forward(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            actions,
            noise,
            time,
        )
        for key, value in self.model._last_sfp_stats.items():
            loss_dict[key] = torch.as_tensor(value, device=losses.device)

        action_dim = self.model.action_dim
        losses = losses[:, :, :action_dim]
        loss_dict["losses_after_forward"] = losses.detach().mean()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad

            if losses.shape[1] == in_episode_bound.shape[1]:
                aligned_mask = in_episode_bound
            elif losses.shape[1] == 1:
                aligned_mask = in_episode_bound.any(dim=1, keepdim=True)
            else:
                aligned_mask = in_episode_bound.any(dim=1, keepdim=True).expand(-1, losses.shape[1])

            losses = losses * aligned_mask.unsqueeze(-1).to(dtype=losses.dtype)
            loss_dict["losses_after_in_ep_bound"] = losses.detach().mean()

        loss_dict["losses_after_rm_padding"] = losses.detach().mean()

        if reduction == "none":
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.detach().mean()
            return per_sample_loss, loss_dict

        loss = losses.mean()
        loss_dict["loss"] = loss.detach()
        return loss, loss_dict


class VLASFP(VLAFlowMatching):
    def __init__(self, config: SmolVLASFPV2Config, rtc_processor: RTCProcessor | None = None):
        super().__init__(config=config, rtc_processor=rtc_processor)

        self.config = config
        self.action_dim = self._infer_action_dim()
        self.model_action_dim = int(self.config.max_action_dim)

        prefix_hidden = self.vlm_with_expert.config.text_config.hidden_size
        expert_hidden = self.vlm_with_expert.expert_hidden_size

        self.action_time_mlp_in = nn.Linear(expert_hidden * 4, expert_hidden)

        self.granularity_predictor = _DirectFrequencyPredictor(prefix_hidden, config)

        self.prefix_condition_proj = nn.Sequential(
            nn.LayerNorm(prefix_hidden),
            nn.Linear(prefix_hidden, expert_hidden),
            nn.SiLU(),
            nn.Linear(expert_hidden, expert_hidden),
        )
        self.step_scaling = _StepScalingLayer(expert_hidden * 2)

        film_hidden = max(64, expert_hidden // 2)
        self.layer_film_generators = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(expert_hidden * 4, film_hidden),
                    nn.SiLU(),
                    nn.Linear(film_hidden, expert_hidden * 4 + 2),
                )
                for _ in self.vlm_with_expert.lm_expert.layers
            ]
        )
        if self.config.sfp_adaln_zero_init:
            for film_generator in self.layer_film_generators:
                nn.init.zeros_(film_generator[-1].weight)
                nn.init.zeros_(film_generator[-1].bias)

        self._film_context = _FiLMContext()
        self._install_film_modulation()
        self._last_sfp_stats = {}

    def _infer_action_dim(self) -> int:
        if ACTION in self.config.input_features:
            return int(self.config.input_features[ACTION].shape[0])
        if hasattr(self.config, "action_feature"):
            return int(self.config.action_feature.shape[0])
        return int(self.config.max_action_dim)

    def _pad_action_to_model_dim(self, actions: Tensor) -> Tensor:
        current_dim = actions.shape[-1]

        if current_dim == self.model_action_dim:
            return actions

        if current_dim > self.model_action_dim:
            return actions[:, :, : self.model_action_dim]

        pad_dim = self.model_action_dim - current_dim
        pad = torch.zeros(
            actions.shape[0],
            actions.shape[1],
            pad_dim,
            dtype=actions.dtype,
            device=actions.device,
        )
        return torch.cat([actions, pad], dim=2)

    def _get_rollout_horizon(self) -> int:
        if hasattr(self.config, "prediction_horizon"):
            return int(self.config.prediction_horizon)
        if hasattr(self.config, "horizon"):
            return int(self.config.horizon)
        return int(self.config.chunk_size)

    def _get_flow_start_idx(self, seq_len: int) -> int:
        return min(max(int(self.config.n_obs_steps) - 1, 0), seq_len - 1)

    def _get_num_train_points(self) -> int:
        return max(1, int(getattr(self.config, "sfp_num_train_points", 1)))

    def _get_num_consistency_points(self) -> int:
        return max(0, int(getattr(self.config, "sfp_num_consistency_points", 0)))

    def _get_consistency_loss_weight(self) -> float:
        return max(0.0, float(getattr(self.config, "sfp_consistency_loss_weight", 0.0)))

    def _get_consistency_delta_multipliers(self) -> tuple[int, ...]:
        multipliers = getattr(self.config, "sfp_consistency_delta_multipliers", (1, 2, 4, 8))
        if isinstance(multipliers, int):
            multipliers = (multipliers,)
        multipliers = tuple(int(multiplier) for multiplier in multipliers)
        if len(multipliers) == 0:
            return (1,)
        return tuple(max(1, multiplier) for multiplier in multipliers)

    def _install_film_modulation(self):
        for layer_idx, layer in enumerate(self.vlm_with_expert.lm_expert.layers):
            if hasattr(layer, "input_layernorm") and not isinstance(layer.input_layernorm, _FiLMLayerNorm):
                layer.input_layernorm = _FiLMLayerNorm(
                    layer.input_layernorm,
                    self._film_context,
                    slot="pre_ln",
                    layer_idx=layer_idx,
                )

            if hasattr(layer, "post_attention_layernorm") and not isinstance(
                layer.post_attention_layernorm,
                _FiLMLayerNorm,
            ):
                layer.post_attention_layernorm = _FiLMLayerNorm(
                    layer.post_attention_layernorm,
                    self._film_context,
                    slot="post_ln",
                    layer_idx=layer_idx,
                )

            if hasattr(layer, "self_attn") and not isinstance(layer.self_attn, _ResidualGateModule):
                layer.self_attn = _ResidualGateModule(
                    layer.self_attn,
                    self._film_context,
                    slot="attn",
                    layer_idx=layer_idx,
                )

            if hasattr(layer, "mlp") and not isinstance(layer.mlp, _ResidualGateModule):
                layer.mlp = _ResidualGateModule(
                    layer.mlp,
                    self._film_context,
                    slot="mlp",
                    layer_idx=layer_idx,
                )

    def _interpolate_trajectory(self, actions: Tensor, time: Tensor) -> tuple[Tensor, Tensor]:
        bsize, seq_len, _ = actions.shape

        if seq_len <= 1:
            if time.ndim == 1:
                return actions[:, :1, :], torch.zeros_like(actions[:, :1, :])
            return actions, torch.zeros_like(actions)

        if time.ndim == 1:
            time = time[:, None]

        scaled_t = time * (seq_len - 1)
        lower = scaled_t.floor().long().clamp(0, seq_len - 2)
        upper = (lower + 1).clamp(0, seq_len - 1)
        lam = (scaled_t - lower.float()).unsqueeze(-1)

        batch_idx = torch.arange(bsize, device=actions.device)[:, None].expand_as(lower)

        xi_l = actions[batch_idx, lower, :]
        xi_u = actions[batch_idx, upper, :]

        xi_t = xi_l + lam * (xi_u - xi_l)
        dxi_dt = (xi_u - xi_l) * (seq_len - 1)

        return xi_t, dxi_dt

    def _sample_cfm_inputs_targets(
        self,
        actions: Tensor,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        bsize = actions.shape[0]
        device = actions.device

        if time is None:
            num_train_points = self._get_num_train_points()
            time_shape = (bsize, num_train_points) if num_train_points > 1 else (bsize,)
            time = torch.rand(time_shape, device=device, dtype=torch.float32)
            time = time * 0.999 + 0.001
        else:
            time = time.to(device=device, dtype=torch.float32)
            if time.numel() == 1 and bsize > 1:
                time = time.expand(bsize)
            time = time.clamp(min=1e-3, max=1.0)

        xi_t, dxi_dt = self._interpolate_trajectory(actions, time)

        xi_t = xi_t[:, :, : self.action_dim]
        dxi_dt = dxi_dt[:, :, : self.action_dim]

        if noise is None:
            noise = self.sample_noise(xi_t.shape, device).to(dtype=actions.dtype)
        else:
            noise = noise.to(device=device, dtype=actions.dtype)

            if noise.ndim == 2:
                noise = noise[:, None, :]
            elif noise.ndim == 3 and noise.shape[1] != xi_t.shape[1]:
                if noise.shape[1] == 1:
                    noise = noise.expand(-1, xi_t.shape[1], -1)
                elif noise.shape[1] > xi_t.shape[1]:
                    noise = noise[:, : xi_t.shape[1], :]
                else:
                    raise ValueError(
                        "noise and SFP sampled actions have incompatible temporal dimensions: "
                        f"noise.shape={tuple(noise.shape)}, xi_t.shape={tuple(xi_t.shape)}"
                    )

            noise = noise[:, :, : self.action_dim]

        sigma = self.config.sfp_sigma0 * torch.exp(-self.config.sfp_k * time)
        if sigma.ndim == 1:
            sigma = sigma[:, None]
        perturbation = sigma[..., None].to(dtype=actions.dtype) * noise

        noised_action = xi_t + perturbation
        target_velocity = -self.config.sfp_k * perturbation + dxi_dt

        return noised_action, target_velocity, time

    def _sample_consistency_inputs_targets(
        self,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        bsize = actions.shape[0]
        device = actions.device
        dtype = actions.dtype
        num_points = self._get_num_consistency_points()

        if num_points <= 0:
            empty_shape = (bsize, 0, self.action_dim)
            empty_actions = actions.new_empty(empty_shape)
            empty_time = torch.empty((bsize, 0), device=device, dtype=torch.float32)
            return empty_actions, empty_actions, empty_time, empty_time, empty_time

        dt_base = 1.0 / max(self._get_rollout_horizon() - int(self.config.n_obs_steps), 1)
        multipliers = self._get_consistency_delta_multipliers()
        multiplier_tensor = torch.tensor(multipliers, device=device, dtype=torch.float32)
        repeats = (num_points + len(multipliers) - 1) // len(multipliers)
        multiplier_tensor = multiplier_tensor.repeat(repeats)[:num_points]
        consistency_weights = 1.0 / multiplier_tensor
        consistency_weights = consistency_weights / consistency_weights.mean().clamp_min(1e-6)
        consistency_weights = consistency_weights[None, :].expand(bsize, num_points)

        delta = (multiplier_tensor * dt_base).clamp(max=0.999)
        delta = delta[None, :].expand(bsize, num_points)

        min_time = 1e-3
        max_t0 = (1.0 - delta).clamp_min(min_time)
        time0 = min_time + torch.rand((bsize, num_points), device=device, dtype=torch.float32) * (
            max_t0 - min_time
        ).clamp_min(0.0)
        time1 = (time0 + delta).clamp(max=1.0)
        actual_delta = (time1 - time0).clamp_min(1e-4)

        xi0, _ = self._interpolate_trajectory(actions, time0)
        xi1, _ = self._interpolate_trajectory(actions, time1)
        xi0 = xi0[:, :, : self.action_dim]
        xi1 = xi1[:, :, : self.action_dim]

        noise = self.sample_noise(xi0.shape, device).to(device=device, dtype=dtype)
        sigma0 = self.config.sfp_sigma0 * torch.exp(-self.config.sfp_k * time0)
        sigma1 = self.config.sfp_sigma0 * torch.exp(-self.config.sfp_k * time1)

        action0 = xi0 + sigma0[..., None].to(dtype=dtype) * noise
        action1 = xi1 + sigma1[..., None].to(dtype=dtype) * noise
        target_velocity = (action1 - action0) / actual_delta[..., None].to(dtype=dtype)

        return action0, target_velocity, time0, actual_delta, consistency_weights

    def _repeat_past_key_values(self, past_key_values, repeats: int):
        if repeats == 1 or past_key_values is None:
            return past_key_values

        repeated = {}
        for layer_idx, layer_cache in past_key_values.items():
            repeated[layer_idx] = {
                key: value.repeat_interleave(repeats, dim=0) if torch.is_tensor(value) else value
                for key, value in layer_cache.items()
            }
        return repeated

    def _pool_prefix_features(self, prefix_features: Tensor, prefix_pad_masks: Tensor) -> Tensor:
        mask = prefix_pad_masks.unsqueeze(-1).to(dtype=prefix_features.dtype)
        denom = prefix_pad_masks.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=prefix_features.dtype)
        return (prefix_features * mask).sum(dim=1) / denom

    def _predict_frequency(
        self,
        prefix_features: Tensor,
        prefix_pad_masks: Tensor,
        clamp: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if not self.config.sfp_use_adaptive_freq:
            freq = torch.ones(
                prefix_features.shape[0],
                device=prefix_features.device,
                dtype=torch.float32,
            )
            return freq, freq, freq

        pooled_prefix = self._pool_prefix_features(prefix_features, prefix_pad_masks)

        predictor_dtype = self.granularity_predictor.net[0].weight.dtype
        raw_freq = self.granularity_predictor(pooled_prefix.to(dtype=predictor_dtype))
        raw_freq = raw_freq.to(device=prefix_features.device, dtype=torch.float32)
        clamped_freq = torch.clamp(
            raw_freq,
            min=self.config.sfp_freq_min,
            max=self.config.sfp_freq_max,
        )
        freq = clamped_freq if clamp else raw_freq

        return raw_freq, freq, clamped_freq

    def _add_obs_temporal_encoding(
        self,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        lang_masks: Tensor | None = None,
        state: Tensor | None = None,
    ) -> Tensor:
        bsize, seq_len, hidden_size = prefix_embs.shape
        device = prefix_embs.device

        position_ids = torch.cumsum(prefix_pad_masks.to(dtype=torch.long), dim=1) - 1
        position_ids = position_ids.clamp(min=0)

        valid_lengths = prefix_pad_masks.sum(dim=1).to(dtype=torch.long)
        state_len = state.shape[1] if state is not None and state.ndim > 2 else 1

        if lang_masks is not None:
            lang_len = lang_masks.sum(dim=1).to(dtype=torch.long)
        else:
            lang_len = torch.zeros_like(valid_lengths)

        obs_lengths = (valid_lengths - lang_len - state_len).clamp(min=0)
        obs_token_mask = prefix_pad_masks & (position_ids < obs_lengths[:, None])

        obs_position_ids = torch.cumsum(obs_token_mask.to(dtype=torch.long), dim=1) - 1
        obs_position_ids = obs_position_ids.clamp(min=0).to(dtype=torch.float32)

        num_frames = max(1, int(self.config.n_obs_steps))
        tokens_per_frame = (obs_lengths.to(dtype=torch.float32) / float(num_frames)).clamp(min=1.0)

        frame_ids = torch.floor(obs_position_ids / tokens_per_frame[:, None]).clamp(0, num_frames - 1)

        t_obs = frame_ids / max(num_frames - 1, 1)
        t_obs = t_obs * obs_token_mask.to(dtype=torch.float32)

        t_flat = t_obs.reshape(-1)

        t_emb = create_sinusoidal_pos_embedding(
            t_flat,
            hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        ).reshape(bsize, seq_len, hidden_size)

        valid_mask = obs_token_mask.unsqueeze(-1).to(dtype=prefix_embs.dtype)

        return prefix_embs + t_emb.to(dtype=prefix_embs.dtype) * valid_mask

    def _compute_film(
        self,
        timestep: Tensor,
        delta_t: Tensor | None,
        freq: Tensor,
        prefix_condition: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if timestep.ndim == 2:
            timestep = timestep.mean(dim=1)
        if delta_t is None:
            delta_t = torch.zeros_like(timestep)
        elif delta_t.ndim == 2:
            delta_t = delta_t.mean(dim=1)

        timestep = timestep.to(dtype=torch.float32)
        delta_t = delta_t.to(device=timestep.device, dtype=torch.float32)
        freq = freq.to(device=timestep.device, dtype=torch.float32)
        prefix_condition = prefix_condition.to(device=timestep.device)

        t_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=timestep.device,
        )

        delta_emb = create_sinusoidal_pos_embedding(
            delta_t,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=timestep.device,
        )

        freq_emb = create_sinusoidal_pos_embedding(
            freq,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=timestep.device,
        )

        prefix_condition_dtype = self.prefix_condition_proj[0].weight.dtype
        prefix_cond = self.prefix_condition_proj(prefix_condition.to(dtype=prefix_condition_dtype))

        film_in = torch.cat(
            [0.5 * t_emb, 0.5 * delta_emb, 0.5 * freq_emb, prefix_cond.to(dtype=t_emb.dtype)],
            dim=-1,
        )
        residual_scale = self.step_scaling(
            torch.cat([t_emb, freq_emb], dim=-1).to(dtype=self.step_scaling.fc[0].weight.dtype),
            freq,
        ).to(dtype=torch.float32)
        h = self.vlm_with_expert.expert_hidden_size
        per_layer_params = []
        for film_generator in self.layer_film_generators:
            film_dtype = film_generator[0].weight.dtype
            film_params = film_generator(film_in.to(dtype=film_dtype)).to(dtype=torch.float32)

            pre_gamma, pre_beta, post_gamma, post_beta, attn_gate_raw, mlp_gate_raw = torch.split(
                film_params,
                [h, h, h, h, 1, 1],
                dim=-1,
            )

            per_layer_params.append(
                (
                    0.02 * torch.tanh(pre_gamma),
                    0.02 * torch.tanh(pre_beta),
                    0.02 * torch.tanh(post_gamma),
                    0.02 * torch.tanh(post_beta),
                    residual_scale[:, None]
                    * (
                        self.config.sfp_adaln_residual_gate_init
                        + self.config.sfp_adaln_residual_gate_span * torch.tanh(attn_gate_raw)
                    ),
                    residual_scale[:, None]
                    * (
                        self.config.sfp_adaln_residual_gate_init
                        + self.config.sfp_adaln_residual_gate_span * torch.tanh(mlp_gate_raw)
                    ),
                )
            )

        stacked_params = []
        for param_group in zip(*per_layer_params, strict=True):
            stacked_params.append(torch.stack([param[:, None, :] for param in param_group], dim=0))

        return tuple(stacked_params)

    def _initial_action(self, state: Tensor, noise: Tensor | None = None) -> Tensor:
        bsize = state.shape[0]
        state_for_init = state[:, -1, :] if state.ndim > 2 else state

        if noise is not None:
            noise = noise.to(device=state.device, dtype=torch.float32)

            if noise.ndim == 2:
                return noise[:, None, : self.action_dim]

            if noise.ndim == 3 and noise.shape[1] > 0:
                return noise[:, :1, : self.action_dim]

        init_actions = torch.zeros(
            (bsize, self.action_dim),
            dtype=torch.float32,
            device=state.device,
        )

        if self.config.sfp_init_from_state:
            copy_dim = min(self.action_dim, state_for_init.shape[-1])
            init_actions[:, :copy_dim] = state_for_init[:, :copy_dim].to(dtype=torch.float32)

        if self.config.sfp_init_noise_std > 0.0:
            init_actions = init_actions + self.config.sfp_init_noise_std * torch.randn_like(init_actions)

        return init_actions[:, None, :].contiguous()

    def embed_suffix(
        self,
        noisy_actions: Tensor,
        timestep: Tensor,
        freq: Tensor,
        delta_t: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        noisy_actions = noisy_actions[:, :, : self.action_dim]
        padded_actions = self._pad_action_to_model_dim(noisy_actions)

        action_emb = self.action_in_proj(padded_actions)

        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype

        if timestep.ndim == 2:
            bsize, query_len = timestep.shape
            flat_time = timestep.reshape(-1)

            time_emb = create_sinusoidal_pos_embedding(
                flat_time,
                self.vlm_with_expert.expert_hidden_size,
                self.config.min_period,
                self.config.max_period,
                device=device,
            ).to(dtype=dtype)

            time_emb = time_emb.reshape(bsize, query_len, -1)
        else:
            time_emb = create_sinusoidal_pos_embedding(
                timestep,
                self.vlm_with_expert.expert_hidden_size,
                self.config.min_period,
                self.config.max_period,
                device=device,
            ).to(dtype=dtype)

            time_emb = time_emb[:, None, :].expand_as(action_emb)

        if delta_t is None:
            delta_t = torch.zeros_like(timestep)

        if delta_t.ndim == 2:
            flat_delta = delta_t.reshape(-1)

            delta_emb = create_sinusoidal_pos_embedding(
                flat_delta,
                self.vlm_with_expert.expert_hidden_size,
                self.config.min_period,
                self.config.max_period,
                device=device,
            ).to(dtype=dtype)

            delta_emb = delta_emb.reshape(bsize, delta_t.shape[1], -1)
        else:
            delta_emb = create_sinusoidal_pos_embedding(
                delta_t,
                self.vlm_with_expert.expert_hidden_size,
                self.config.min_period,
                self.config.max_period,
                device=device,
            ).to(dtype=dtype)

            delta_emb = delta_emb[:, None, :].expand_as(action_emb)

        freq = freq.to(device=device, dtype=torch.float32)
        freq_emb = create_sinusoidal_pos_embedding(
            freq,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        ).to(dtype=dtype)

        freq_emb = freq_emb[:, None, :].expand_as(action_emb)

        action_time_emb = torch.cat([action_emb, 0.5 * time_emb, 0.5 * delta_emb, 0.5 * freq_emb], dim=2)
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        suffix_pad_masks = torch.ones(
            bsize,
            action_time_emb.shape[1],
            dtype=torch.bool,
            device=device,
        )

        suffix_att_masks = torch.ones(
            bsize,
            action_time_emb.shape[1],
            dtype=action_time_emb.dtype,
            device=device,
        )

        return action_time_emb, suffix_pad_masks, suffix_att_masks

    def _predict_velocity_from_cache(
        self,
        noisy_actions: Tensor,
        timestep: Tensor,
        delta_t: Tensor,
        freq: Tensor,
        prefix_condition: Tensor,
        prefix_pad_masks: Tensor,
        past_key_values,
    ) -> Tensor:
        film_params = self._compute_film(timestep, delta_t, freq, prefix_condition)
        self._film_context.set_film(*film_params)

        try:
            suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
                noisy_actions,
                timestep,
                freq,
                delta_t=delta_t,
            )

            suffix_len = suffix_pad_masks.shape[1]
            batch_size = prefix_pad_masks.shape[0]
            prefix_len = prefix_pad_masks.shape[1]

            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
                batch_size,
                suffix_len,
                prefix_len,
            )

            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

            outputs_embeds, _ = self.vlm_with_expert.forward(
                attention_mask=full_att_2d_masks,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=self.config.use_cache,
                fill_kv_cache=False,
            )

            suffix_out = outputs_embeds[1][:, -1:, :].to(dtype=torch.float32)
            predicted_velocity = self.action_out_proj(suffix_out)
            return predicted_velocity[:, :, : self.action_dim]

        finally:
            self._film_context.clear()

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise=None,
        time=None,
    ) -> Tensor:
        flow_start_idx = self._get_flow_start_idx(actions.shape[1])
        flow_actions = actions[:, flow_start_idx:, :]

        noised_action, target_velocity, time = self._sample_cfm_inputs_targets(
            flow_actions,
            noise=noise,
            time=time,
        )
        bsize = actions.shape[0]
        num_queries = noised_action.shape[1]

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state=state,
        )

        prefix_embs = self._add_obs_temporal_encoding(
            prefix_embs,
            prefix_pad_masks,
            lang_masks=lang_masks,
            state=state,
        )

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_outputs, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        prefix_out = prefix_outputs[0]

        raw_freq, freq, clamped_freq = self._predict_frequency(
            prefix_out,
            prefix_pad_masks,
            clamp=self.config.sfp_clamp_freq_during_training,
        )
        prefix_condition = self._pool_prefix_features(prefix_out, prefix_pad_masks)
        self._last_sfp_stats = {
            "pred_freq": float(freq.mean().detach().cpu()),
            "raw_pred_freq": float(raw_freq.mean().detach().cpu()),
            "clamped_pred_freq": float(clamped_freq.mean().detach().cpu()),
        }

        consistency_weight = self._get_consistency_loss_weight()
        use_consistency = consistency_weight > 0.0 and self._get_num_consistency_points() > 0

        if time.ndim == 1:
            cfm_time = time[:, None]
        else:
            cfm_time = time

        cfm_delta_t = torch.zeros_like(cfm_time)
        combined_action = noised_action
        combined_target_velocity = target_velocity
        combined_time = cfm_time
        combined_delta_t = cfm_delta_t
        num_cons_queries = 0

        if use_consistency:
            (
                cons_action,
                cons_target_velocity,
                cons_time,
                cons_delta_t,
                cons_weights,
            ) = self._sample_consistency_inputs_targets(
                flow_actions,
            )
            num_cons_queries = cons_action.shape[1]

            if num_cons_queries > 0:
                combined_action = torch.cat([combined_action, cons_action], dim=1)
                combined_target_velocity = torch.cat([combined_target_velocity, cons_target_velocity], dim=1)
                combined_time = torch.cat([combined_time, cons_time], dim=1)
                combined_delta_t = torch.cat([combined_delta_t, cons_delta_t], dim=1)

        total_queries = combined_action.shape[1]

        flat_action = combined_action.reshape(bsize * total_queries, 1, self.action_dim)
        flat_time = combined_time.reshape(bsize * total_queries)
        flat_delta_t = combined_delta_t.reshape(bsize * total_queries)
        flat_freq = freq.repeat_interleave(total_queries, dim=0)
        flat_prefix_condition = prefix_condition.repeat_interleave(total_queries, dim=0)
        flat_prefix_pad_masks = prefix_pad_masks.repeat_interleave(total_queries, dim=0)
        flat_past_key_values = self._repeat_past_key_values(past_key_values, total_queries)

        predicted_velocity = self._predict_velocity_from_cache(
            flat_action,
            flat_time,
            flat_delta_t,
            flat_freq,
            flat_prefix_condition,
            flat_prefix_pad_masks,
            flat_past_key_values,
        )
        predicted_velocity = predicted_velocity.reshape(bsize, total_queries, self.action_dim)

        target_velocity = combined_target_velocity[:, :, : self.action_dim].to(dtype=torch.float32)
        cfm_predicted_velocity = predicted_velocity[:, :num_queries, :]
        cfm_target_velocity = target_velocity[:, :num_queries, :]

        losses = F.mse_loss(
            cfm_predicted_velocity,
            cfm_target_velocity,
            reduction="none",
        )

        if self.config.sfp_freq_reg_weight > 0.0:
            freq_reg = self.config.sfp_freq_reg_weight * (freq - 1.0).pow(2)
            losses = losses + freq_reg[:, None, None].to(dtype=losses.dtype)

        if num_cons_queries <= 0:
            return losses

        cons_predicted_velocity = predicted_velocity[:, num_queries:, :]
        cons_target_velocity = target_velocity[:, num_queries:, :]
        cons_losses = F.mse_loss(
            cons_predicted_velocity,
            cons_target_velocity,
            reduction="none",
        )
        cons_losses = cons_losses * cons_weights[:, :, None].to(dtype=cons_losses.dtype)

        return losses + consistency_weight * cons_losses.mean(dim=1, keepdim=True).to(dtype=losses.dtype)

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        **kwargs: ActionSelectKwargs,
    ) -> Tensor:
        bsize = state.shape[0]
        device = state.device

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state=state,
        )

        prefix_embs = self._add_obs_temporal_encoding(
            prefix_embs,
            prefix_pad_masks,
            lang_masks=lang_masks,
            state=state,
        )

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_outputs, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        prefix_out = prefix_outputs[0]

        raw_freq, freq, clamped_freq = self._predict_frequency(
            prefix_out,
            prefix_pad_masks,
            clamp=self.config.sfp_clamp_freq_during_eval,
        )
        prefix_condition = self._pool_prefix_features(prefix_out, prefix_pad_masks)
        self._last_sfp_stats = {
            "pred_freq": float(freq.mean().detach().cpu()),
            "raw_pred_freq": float(raw_freq.mean().detach().cpu()),
            "clamped_pred_freq": float(clamped_freq.mean().detach().cpu()),
        }

        x_t = self._initial_action(state=state, noise=noise).detach()

        rollout_horizon = self._get_rollout_horizon()
        rollout_steps = int(self.config.n_action_steps)

        if rollout_horizon <= 0:
            raise ValueError("rollout horizon must be > 0")
        if rollout_steps <= 0:
            raise ValueError("n_action_steps must be > 0")

        dt = 1.0 / max(rollout_horizon - int(self.config.n_obs_steps), 1)

        generated_actions = []

        for token_idx in range(rollout_steps):
            time = token_idx * dt

            time_tensor = torch.full(
                (bsize,),
                time,
                dtype=torch.float32,
                device=device,
            )
            consistency_enabled = self._get_consistency_loss_weight() > 0.0 and self._get_num_consistency_points() > 0
            inference_delta_t = dt if consistency_enabled else 0.0
            delta_t_tensor = torch.full(
                (bsize,),
                inference_delta_t,
                dtype=torch.float32,
                device=device,
            )

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=input_x_t,
                    timestep=current_timestep,
                    delta_t=delta_t_tensor,
                    freq=freq,
                    prefix_condition=prefix_condition,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            v_t = v_t[:, :, : self.action_dim]
            x_t = x_t + dt * v_t

            generated_actions.append(x_t)

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        actions = torch.cat(generated_actions, dim=1)
        return actions[:, :, : self.action_dim]

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        delta_t,
        freq,
        prefix_condition,
    ):
        x_t = x_t[:, :, : self.action_dim]

        return self._predict_velocity_from_cache(
            x_t,
            timestep,
            delta_t,
            freq,
            prefix_condition,
            prefix_pad_masks,
            past_key_values,
        )
