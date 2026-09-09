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

"""Gymnasium adapter for released MimicGen robomimic environments."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .utils import parse_camera_names

ACTION_DIM = 7
STATE_KEYS = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)
SUPPORTED_TASKS = {
    "square_d1": "Square_D1",
    "threading_d1": "Threading_D1",
}
ROBOSUITE_CAMERA_NAMES = {
    "agentview_image": "agentview",
    "robot0_eye_in_hand_image": "robot0_eye_in_hand",
}
TASK_DESCRIPTIONS = {
    "Square_D1": "insert the square nut onto the square peg",
    "Threading_D1": "thread the needle through the tripod ring",
}


def canonical_mimicgen_task(task: str) -> str:
    """Normalize the two supported task spellings and reject silent mismatches."""
    key = task.strip().lower()
    if key not in SUPPORTED_TASKS:
        choices = ", ".join(SUPPORTED_TASKS.values())
        raise ValueError(f"Unsupported MimicGen task '{task}'. Supported tasks: {choices}.")
    return SUPPORTED_TASKS[key]


def _image_to_hwc_uint8(image: np.ndarray) -> np.ndarray:
    """Convert robomimic's processed CHW float image to Gym HWC uint8."""
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3-D camera image, got shape {image.shape}.")
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.moveaxis(image, 0, -1)
    elif image.shape[-1] not in (1, 3):
        raise ValueError(f"Could not identify the image channel axis in shape {image.shape}.")

    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all():
            raise ValueError("Camera observation contains non-finite values.")
        if image.size and float(image.max()) <= 1.0 + 1e-6:
            image = image * 255.0
        image = np.rint(np.clip(image, 0.0, 255.0)).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


class MimicGenGymEnv(gym.Env):
    """One lazy off-screen MimicGen environment with reproducible seeded resets."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        *,
        task: str,
        dataset_path: str | Path,
        episode_length: int = 400,
        camera_name: str | Sequence[str] = "agentview_image,robot0_eye_in_hand_image",
        obs_type: str = "pixels_agent_pos",
        render_mode: str = "rgb_array",
        observation_height: int = 84,
        observation_width: int = 84,
        eager_init: bool = False,
    ):
        super().__init__()
        self.task = canonical_mimicgen_task(task)
        self.task_description = TASK_DESCRIPTIONS[self.task]
        self.dataset_path = Path(dataset_path).expanduser().resolve()
        if not self.dataset_path.is_file():
            raise FileNotFoundError(f"MimicGen dataset not found: {self.dataset_path}")
        if obs_type not in {"pixels", "pixels_agent_pos"}:
            raise ValueError(f"Unsupported obs_type: {obs_type}")
        if episode_length <= 0:
            raise ValueError(f"episode_length must be positive, got {episode_length}.")

        self.camera_names = parse_camera_names(camera_name)
        unknown_cameras = [camera for camera in self.camera_names if camera not in ROBOSUITE_CAMERA_NAMES]
        if unknown_cameras:
            raise ValueError(
                f"Unsupported MimicGen observation cameras {unknown_cameras}. "
                f"Supported cameras: {list(ROBOSUITE_CAMERA_NAMES)}."
            )
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_height = observation_height
        self.observation_width = observation_width
        self._max_episode_steps = episode_length
        self._elapsed_steps = 0
        self._env: Any | None = None
        self._seed_state_map: dict[int, np.ndarray] = {}
        self._render_cache: np.ndarray | None = None

        image_spaces = {
            camera: spaces.Box(
                low=0,
                high=255,
                shape=(observation_height, observation_width, 3),
                dtype=np.uint8,
            )
            for camera in self.camera_names
        }
        observation_spaces: dict[str, spaces.Space] = {"pixels": spaces.Dict(image_spaces)}
        if obs_type == "pixels_agent_pos":
            observation_spaces["agent_pos"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(9,),
                dtype=np.float32,
            )
        self.observation_space = spaces.Dict(observation_spaces)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)
        if eager_init:
            # LeRobot constructs eval environments before placing the policy on
            # CUDA. Initialize EGL here to preserve that safe ordering.
            self._ensure_env()

    def _ensure_env(self) -> None:
        if self._env is not None:
            return
        try:
            import mimicgen_envs  # noqa: F401
        except ImportError:
            try:
                import mimicgen  # noqa: F401
            except ImportError as exc:
                raise ModuleNotFoundError(
                    "MimicGen evaluation requires mimicgen_envs (or mimicgen), robomimic, and robosuite."
                ) from exc

        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.file_utils as FileUtils
        import robomimic.utils.obs_utils as ObsUtils

        env_meta = FileUtils.get_env_metadata_from_dataset(str(self.dataset_path))
        dataset_task = canonical_mimicgen_task(env_meta["env_name"])
        if dataset_task != self.task:
            raise ValueError(
                f"Task '{self.task}' does not match HDF5 environment '{env_meta['env_name']}' "
                f"in {self.dataset_path}."
            )

        env_meta["env_kwargs"]["use_object_obs"] = False
        env_meta["env_kwargs"]["camera_names"] = [
            ROBOSUITE_CAMERA_NAMES[camera] for camera in self.camera_names
        ]
        env_meta["env_kwargs"]["camera_heights"] = self.observation_height
        env_meta["env_kwargs"]["camera_widths"] = self.observation_width
        ObsUtils.initialize_obs_modality_mapping_from_dict(
            {
                "rgb": list(self.camera_names),
                "low_dim": list(STATE_KEYS),
            }
        )
        self._env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=True,
            use_image_obs=True,
        )
        # Robosuite hard resets rebuild MuJoCo and leak memory over long evals.
        self._env.env.hard_reset = False

    def _format_observation(self, raw_obs: Mapping[str, np.ndarray]) -> dict[str, Any]:
        images = {camera: _image_to_hwc_uint8(raw_obs[camera]) for camera in self.camera_names}
        self._render_cache = images[self.camera_names[0]]
        observation: dict[str, Any] = {"pixels": images}
        if self.obs_type == "pixels_agent_pos":
            missing = [key for key in STATE_KEYS if key not in raw_obs]
            if missing:
                raise KeyError(f"MimicGen observation is missing state fields: {missing}")
            observation["agent_pos"] = np.concatenate(
                [np.asarray(raw_obs[key], dtype=np.float32).reshape(-1) for key in STATE_KEYS]
            )
        return observation

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        super().reset(seed=seed)
        self._ensure_env()
        assert self._env is not None
        if seed is not None and seed in self._seed_state_map:
            raw_obs = self._env.reset_to({"states": self._seed_state_map[seed]})
        else:
            if seed is not None:
                # Released MimicGen / robosuite task placement uses NumPy's
                # global RNG, matching the official robomimic runner.
                np.random.seed(seed)
            raw_obs = self._env.reset()
            if seed is not None:
                self._seed_state_map[seed] = np.asarray(self._env.get_state()["states"]).copy()
        self._elapsed_steps = 0
        return self._format_observation(raw_obs), {"is_success": False, "task": self.task}

    def step(self, action: np.ndarray):
        self._ensure_env()
        assert self._env is not None
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"Expected action shape ({ACTION_DIM},), got {action.shape}.")

        raw_obs, reward, done, info = self._env.step(action)
        self._elapsed_steps += 1
        success_result = self._env.is_success()
        if isinstance(success_result, Mapping):
            is_success = (
                bool(success_result["task"])
                if "task" in success_result
                else any(bool(value) for value in success_result.values())
            )
        else:
            is_success = bool(success_result)
        terminated = bool(done or is_success)
        truncated = self._elapsed_steps >= self._max_episode_steps and not terminated
        info = dict(info)
        info.update({"is_success": is_success, "task": self.task, "done": bool(done)})
        return self._format_observation(raw_obs), float(reward), terminated, truncated, info

    def render(self) -> np.ndarray:
        if self._render_cache is None:
            raise RuntimeError("Call reset() before render().")
        return self._render_cache.copy()

    def close(self) -> None:
        if self._env is not None:
            if hasattr(self._env, "close"):
                self._env.close()
            elif hasattr(self._env, "env") and hasattr(self._env.env, "close"):
                self._env.env.close()
            self._env = None


def create_mimicgen_envs(
    *,
    task: str,
    dataset_path: str | Path,
    n_envs: int,
    camera_name: str | Sequence[str] = "agentview_image,robot0_eye_in_hand_image",
    episode_length: int = 400,
    gym_kwargs: dict[str, Any] | None = None,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
) -> dict[str, dict[int, Any]]:
    """Build a LeRobot-compatible vector environment for one MimicGen task."""
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable vector-environment constructor.")
    if n_envs <= 0:
        raise ValueError(f"n_envs must be positive, got {n_envs}.")
    canonical_task = canonical_mimicgen_task(task)
    kwargs = dict(gym_kwargs or {})
    env_fns = [
        partial(
            MimicGenGymEnv,
            task=canonical_task,
            dataset_path=dataset_path,
            episode_length=episode_length,
            camera_name=camera_name,
            eager_init=True,
            **kwargs,
        )
        for _ in range(n_envs)
    ]
    return {canonical_task: {0: env_cls(env_fns)}}
