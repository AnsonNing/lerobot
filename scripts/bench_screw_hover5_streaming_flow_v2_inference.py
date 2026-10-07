#!/usr/bin/env python3
"""Measure a saved screw Streaming Flow V2 policy pipeline without robot I/O."""

import argparse
import contextlib
import io
import json
import statistics
import time
from pathlib import Path

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.streaming_flow.modeling_streaming_flow_v2 import StreamingFlowPolicy

ROOT = Path("/mnt/data/ningan/screw_datasets/cut_pull_screw_hover5_gripper10_assemble_episode")
REPO_ID = "ningan/cut_pull_screw_hover5_gripper10_assemble_episode"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--repo-id", default=REPO_ID)
    args = parser.parse_args()

    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.device = args.device
    with contextlib.redirect_stdout(io.StringIO()):
        policy = StreamingFlowPolicy.from_pretrained(args.checkpoint, config=cfg).to(args.device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    dataset = LeRobotDataset(args.repo_id, root=args.root, episodes=[0], return_uint8=True)
    sample = dataset[0]
    raw = {key: sample[key] for key in cfg.input_features}
    for key in cfg.image_features:
        if raw[key].dtype == torch.uint8:
            raw[key] = raw[key].float() / 255.0

    latencies = []
    with torch.inference_mode():
        for trial in range(args.trials + 5):
            policy.reset()
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            batch = preprocessor(raw)
            action = policy.select_action(batch)
            physical_action = postprocessor(action)
            if physical_action.shape != (1, 6) or not torch.isfinite(physical_action).all():
                raise RuntimeError("Expected a finite six-joint screw action")
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            if trial >= 5:
                latencies.append((time.perf_counter() - start) * 1000)

    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "device": args.device,
                "trials": args.trials,
                "pipeline_ms_median": statistics.median(latencies),
                "pipeline_ms_p95": sorted(latencies)[int(0.95 * (len(latencies) - 1))],
                "action_shape": list(physical_action.shape),
                "camera_keys": list(cfg.image_features),
                "hardware_loop_included": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
