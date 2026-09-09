#!/usr/bin/env python

"""Convert released Square_D1 / Threading_D1 HDF5 demos to LeRobotDataset v3."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.mimicgen import canonical_mimicgen_task
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

CAMERA_KEYS = (
    "agentview_image",
    "robot0_eye_in_hand_image",
)
STATE_KEYS = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)
STATE_NAMES = (
    "eef_x",
    "eef_y",
    "eef_z",
    "eef_quat_x",
    "eef_quat_y",
    "eef_quat_z",
    "eef_quat_w",
    "gripper_left_qpos",
    "gripper_right_qpos",
)
ACTION_NAMES = (
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_rx",
    "delta_ry",
    "delta_rz",
    "gripper",
)
_DEMO_NAME_RE = re.compile(r"^demo_(\d+)$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5-path", required=True, type=Path)
    parser.add_argument("--repo-id", required=True, type=str)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--task",
        default=None,
        help="Optional Square_D1 or Threading_D1 override; must match the HDF5 metadata.",
    )
    parser.add_argument("--fps", default=20, type=int)
    parser.add_argument(
        "--use-videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Encode cameras as MP4 (default); use --no-use-videos for image files.",
    )
    parser.add_argument("--image-writer-threads", default=8, type=int)
    parser.add_argument("--max-episodes", default=None, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _demo_sort_key(name: str) -> int:
    match = _DEMO_NAME_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"Unexpected episode key '{name}'; expected demo_<integer>.")
    return int(match.group(1))


def _decode_env_metadata(data_group: Any) -> dict[str, Any]:
    if "env_args" not in data_group.attrs:
        raise KeyError("HDF5 data group has no 'env_args' environment metadata.")
    raw = data_group.attrs["env_args"]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _episode_length(demo: Any) -> int:
    required = ["actions", *(f"obs/{key}" for key in (*CAMERA_KEYS, *STATE_KEYS))]
    missing = [key for key in required if key not in demo]
    if missing:
        raise KeyError(f"{demo.name} is missing required datasets: {missing}")
    lengths = {key: int(demo[key].shape[0]) for key in required}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{demo.name} contains inconsistent trajectory lengths: {lengths}")
    if demo["actions"].shape[1:] != (7,):
        raise ValueError(f"{demo.name}/actions must have shape (T, 7), got {demo['actions'].shape}.")
    for camera in CAMERA_KEYS:
        shape = demo[f"obs/{camera}"].shape
        if len(shape) != 4 or shape[-1] != 3:
            raise ValueError(f"{demo.name}/obs/{camera} must be HWC RGB, got {shape}.")
    state_dim = sum(int(demo[f"obs/{key}"].shape[1]) for key in STATE_KEYS)
    if state_dim != len(STATE_NAMES):
        raise ValueError(f"{demo.name} expected a 9-D robot state, got {state_dim}.")
    return next(iter(lengths.values()))


def _dataset_features(first_demo: Any, use_videos: bool) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {
        ACTION: {
            "dtype": "float32",
            "shape": (len(ACTION_NAMES),),
            "names": list(ACTION_NAMES),
        },
        OBS_STATE: {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": list(STATE_NAMES),
        },
    }
    for camera in CAMERA_KEYS:
        shape = tuple(first_demo[f"obs/{camera}"].shape[1:])
        features[f"{OBS_IMAGES}.{camera}"] = {
            "dtype": "video" if use_videos else "image",
            "shape": shape,
            "names": ["height", "width", "channels"],
        }
    return features


def convert_mimicgen_hdf5(
    *,
    hdf5_path: Path,
    repo_id: str,
    root: Path,
    task: str | None = None,
    fps: int = 20,
    use_videos: bool = True,
    image_writer_threads: int = 8,
    max_episodes: int | None = None,
    overwrite: bool = False,
) -> None:
    """Convert one released MimicGen HDF5 without modifying the source file."""
    try:
        import h5py
    except ImportError as exc:
        raise ModuleNotFoundError("MimicGen HDF5 conversion requires h5py (`pip install h5py`).") from exc

    hdf5_path = hdf5_path.expanduser().resolve()
    root = root.expanduser().resolve()
    if not hdf5_path.is_file():
        raise FileNotFoundError(f"MimicGen HDF5 file not found: {hdf5_path}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}.")
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError(f"max_episodes must be positive, got {max_episodes}.")
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"{root} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(root)

    with h5py.File(hdf5_path, "r") as source:
        if "data" not in source:
            raise KeyError(f"{hdf5_path} has no 'data' group.")
        data = source["data"]
        metadata = _decode_env_metadata(data)
        metadata_task = canonical_mimicgen_task(metadata["env_name"])
        selected_task = metadata_task if task is None else canonical_mimicgen_task(task)
        if selected_task != metadata_task:
            raise ValueError(
                f"Requested task '{selected_task}' does not match HDF5 environment '{metadata_task}'."
            )

        demo_keys = sorted(data.keys(), key=_demo_sort_key)
        if not demo_keys:
            raise ValueError(f"{hdf5_path} contains no demonstrations.")
        if max_episodes is not None:
            demo_keys = demo_keys[:max_episodes]
        first_demo = data[demo_keys[0]]
        _episode_length(first_demo)

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=root,
            fps=fps,
            features=_dataset_features(first_demo, use_videos),
            robot_type="panda_mimicgen",
            use_videos=use_videos,
            image_writer_threads=image_writer_threads,
            batch_encoding_size=1,
        )

        for episode_index, demo_key in enumerate(demo_keys):
            demo = data[demo_key]
            length = _episode_length(demo)
            for frame_index in range(length):
                state = np.concatenate(
                    [
                        np.asarray(demo[f"obs/{key}"][frame_index], dtype=np.float32).reshape(-1)
                        for key in STATE_KEYS
                    ]
                )
                frame: dict[str, Any] = {
                    ACTION: np.asarray(demo["actions"][frame_index], dtype=np.float32),
                    OBS_STATE: state,
                    "task": selected_task,
                }
                for camera in CAMERA_KEYS:
                    frame[f"{OBS_IMAGES}.{camera}"] = np.asarray(
                        demo[f"obs/{camera}"][frame_index], dtype=np.uint8
                    )
                dataset.add_frame(frame)
            dataset.save_episode()
            print(f"saved episode {episode_index + 1}/{len(demo_keys)} ({length} frames)")

        dataset.finalize()
    print(f"Done. LeRobot dataset written to: {root}")
    print(f"Train with: --dataset.repo_id={repo_id} --dataset.root={root}")


def main() -> None:
    args = _parse_args()
    convert_mimicgen_hdf5(
        hdf5_path=args.hdf5_path,
        repo_id=args.repo_id,
        root=args.root,
        task=args.task,
        fps=args.fps,
        use_videos=args.use_videos,
        image_writer_threads=args.image_writer_threads,
        max_episodes=args.max_episodes,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
