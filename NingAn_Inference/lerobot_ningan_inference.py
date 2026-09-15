#!/usr/bin/env python3
"""Deploy a NingAn Streaming Flow V2 checkpoint to an SO-101 follower."""

from __future__ import annotations

import argparse
from pathlib import Path

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.configs import PreTrainedConfig
from lerobot.robots import make_robot_from_config
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.rollout import RolloutConfig
from lerobot.rollout.configs import BaseStrategyConfig
from lerobot.rollout.inference import SyncInferenceConfig
from lerobot.scripts.lerobot_rollout import rollout


class NingAnSO101FollowerConfig(SOFollowerRobotConfig):
    """Disambiguate SO-101 from the shared SO-100/SO-101 config class."""

    @property
    def type(self) -> str:
        return "so101_follower"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--robot-port", default="/dev/ttyACM0")
    parser.add_argument("--robot-id", default="my_awesome_follower_right_arm")
    parser.add_argument("--front-camera", type=int, default=4)
    parser.add_argument("--side-camera", type=int, default=6)
    parser.add_argument("--max-relative-target", type=float, default=5.0)
    parser.add_argument("--task", default="Push the button")
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Connect and print joint positions without loading the policy or sending actions.",
    )
    parser.add_argument(
        "--leave-final-pose",
        action="store_true",
        help="Do not smoothly return to the startup pose during shutdown.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    required_files = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    )
    missing = [name for name in required_files if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete checkpoint {checkpoint}; missing: {', '.join(missing)}")

    policy = PreTrainedConfig.from_pretrained(checkpoint)
    if policy.type != "streaming_flow_v2":
        raise ValueError(f"Expected a streaming_flow_v2 checkpoint, got {policy.type!r}")
    policy.device = args.device
    policy.pretrained_path = checkpoint

    robot = NingAnSO101FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        use_degrees=True,
        max_relative_target=args.max_relative_target,
        cameras={
            "front": OpenCVCameraConfig(
                index_or_path=args.front_camera,
                width=640,
                height=480,
                fps=30,
                fourcc="MJPG",
            ),
            "side": OpenCVCameraConfig(
                index_or_path=args.side_camera,
                width=640,
                height=480,
                fps=30,
                fourcc="MJPG",
            ),
        },
    )

    if args.inspect_only:
        follower = make_robot_from_config(robot)
        try:
            follower.connect()
            observation = follower.get_observation()
            positions = {key: value for key, value in observation.items() if key.endswith(".pos")}
            print("SO-101 joint positions (no action was sent):")
            for key, value in positions.items():
                print(f"  {key}: {value:.6f}")
        finally:
            if follower.is_connected:
                follower.disconnect()
        return

    config = RolloutConfig(
        robot=robot,
        policy=policy,
        strategy=BaseStrategyConfig(),
        inference=SyncInferenceConfig(),
        device=args.device,
        duration=args.duration,
        fps=args.fps,
        task=args.task,
        return_to_initial_position=not args.leave_final_pose,
    )

    # Bypass draccus parsing: all values above are typed dataclass instances.
    rollout.__wrapped__(config)


if __name__ == "__main__":
    main()
