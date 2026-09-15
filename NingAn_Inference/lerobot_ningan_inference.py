#!/usr/bin/env python3
"""Deploy a NingAn Streaming Flow V2 checkpoint to an SO-101 follower."""

from __future__ import annotations

import argparse
from datetime import datetime
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
    parser.add_argument("--robot-id")
    parser.add_argument(
        "--calibration-file",
        type=Path,
        help="Existing SO-101 motor calibration JSON to use explicitly.",
    )
    parser.add_argument("--front-camera", type=int, default=4)
    parser.add_argument("--side-camera", type=int, default=6)
    parser.add_argument("--max-relative-target", type=float, default=5.0)
    parser.add_argument("--task", default="Push the button")
    parser.add_argument(
        "--output_each_step_action",
        "--output-each-step-action",
        action="store_true",
        help="Write every action actually sent to the six motors into a CSV file.",
    )
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

    calibration_dir = None
    robot_id = args.robot_id or "my_awesome_follower_right_arm"
    if args.calibration_file is not None:
        calibration_file = args.calibration_file.expanduser().resolve()
        if not calibration_file.is_file():
            raise FileNotFoundError(f"Calibration file does not exist: {calibration_file}")
        if calibration_file.suffix.lower() != ".json":
            raise ValueError(f"Calibration file must be JSON: {calibration_file}")
        if args.robot_id is not None and args.robot_id != calibration_file.stem:
            raise ValueError(
                "When --calibration-file is used, --robot-id must be omitted or match "
                f"the filename stem ({calibration_file.stem!r})."
            )
        calibration_dir = calibration_file.parent
        robot_id = calibration_file.stem

    robot = NingAnSO101FollowerConfig(
        port=args.robot_port,
        id=robot_id,
        calibration_dir=calibration_dir,
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

    output_action_csv = None
    if args.output_each_step_action:
        log_dir = Path(__file__).resolve().parent / "action_logs"
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        output_action_csv = log_dir / f"{checkpoint.name}_{timestamp}.csv"
        collision_index = 2
        while output_action_csv.exists():
            output_action_csv = log_dir / f"{checkpoint.name}_{timestamp}_{collision_index}.csv"
            collision_index += 1
        print(f"Actions will be recorded in: {output_action_csv}")

    config = RolloutConfig(
        robot=robot,
        policy=policy,
        strategy=BaseStrategyConfig(),
        inference=SyncInferenceConfig(),
        device=args.device,
        duration=args.duration,
        fps=args.fps,
        task=args.task,
        output_action_csv=output_action_csv,
        return_to_initial_position=not args.leave_final_pose,
    )

    # Bypass draccus parsing: all values above are typed dataclass instances.
    rollout.__wrapped__(config)


if __name__ == "__main__":
    main()
