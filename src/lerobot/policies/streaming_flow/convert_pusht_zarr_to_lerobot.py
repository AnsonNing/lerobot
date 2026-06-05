#!/usr/bin/env python

"""Convert the notebook PushT image zarr dataset to a local LeRobot dataset."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import zarr

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr-path", required=True, type=Path)
    parser.add_argument("--repo-id", default="ningan/pusht_multi_freq_image", type=str)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--fps", default=10, type=int)
    parser.add_argument("--task", default="PushT-v0", type=str)
    parser.add_argument("--use-videos", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-episodes", default=None, type=int)
    return parser.parse_args()


def _image_feature_shape(images: np.ndarray) -> tuple[int, int, int]:
    if images.ndim != 4:
        raise ValueError(f"Expected image array with shape (N,C,H,W) or (N,H,W,C), got {images.shape}")
    if images.shape[1] in (1, 3):
        return tuple(images.shape[1:])  # (C,H,W)
    if images.shape[-1] in (1, 3):
        return tuple(images.shape[1:])  # (H,W,C)
    raise ValueError(f"Could not infer image channel dimension from shape {images.shape}")


def main() -> None:
    args = _parse_args()

    if args.root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.root} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(args.root)

    root = zarr.open(args.zarr_path, mode="r")
    data = root["data"]
    actions = data["action"]
    images = data["image"]
    states = data["state"] if "state" in data else None
    episode_ends = np.asarray(root["meta"]["episode_ends"])

    features = {
        ACTION: {
            "dtype": "float32",
            "shape": tuple(actions.shape[1:]),
            "names": {"motors": ["x", "y"]},
        },
        OBS_IMAGE: {
            "dtype": "video" if args.use_videos else "image",
            "shape": _image_feature_shape(images),
            "names": ["channel", "height", "width"]
            if images.shape[1] in (1, 3)
            else ["height", "width", "channel"],
        },
    }
    if states is not None:
        features[OBS_STATE] = {
            "dtype": "float32",
            "shape": tuple(states.shape[1:]),
            "names": None,
        }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.root,
        fps=args.fps,
        features=features,
        robot_type="pusht",
        use_videos=args.use_videos,
        image_writer_threads=4,
        vcodec="h264",
    )

    start = 0
    n_episodes = len(episode_ends) if args.max_episodes is None else min(args.max_episodes, len(episode_ends))
    for ep_idx, end in enumerate(episode_ends[:n_episodes]):
        end = int(end)
        for frame_idx in range(start, end):
            frame = {
                ACTION: np.asarray(actions[frame_idx], dtype=np.float32),
                OBS_IMAGE: np.asarray(images[frame_idx]),
                "task": args.task,
            }
            if states is not None:
                frame[OBS_STATE] = np.asarray(states[frame_idx], dtype=np.float32)
            dataset.add_frame(frame)
        dataset.save_episode()
        print(f"saved episode {ep_idx + 1}/{n_episodes} ({end - start} frames)")
        start = end

    dataset.finalize()
    print(f"Done. LeRobot dataset written to: {args.root}")
    print(f"Train with: --dataset.repo_id={args.repo_id} --dataset.root={args.root}")


if __name__ == "__main__":
    main()
