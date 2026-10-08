#!/usr/bin/env python3
"""Evaluate a Diffusion or Streaming Flow checkpoint on all 301 training episodes."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import get_policy_class, make_pre_post_processors


DEFAULT_ROOT = Path(
    "/home/lerobot/NingAn/lerobot/NingAn_Inference/Pull_Screw/cut_datasets/"
    "histogram_balance_cut_pull_screw_all_hover_gripper_25_assemble_episode"
)
DEFAULT_REPO_ID = "HZL/histogram_balance_cut_pull_screw_all_hover_gripper_25_assemble_episode"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=100000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    root = args.root.expanduser().resolve()
    config = PreTrainedConfig.from_pretrained(checkpoint)
    if config.type not in {"diffusion", "streaming_flow_v2"}:
        raise ValueError(f"Expected diffusion or streaming_flow_v2, got {config.type!r}")
    config.device = args.device

    metadata = LeRobotDatasetMetadata(args.repo_id, root=root)
    if metadata.total_episodes != 301:
        raise ValueError(f"Expected exactly 301 episodes, got {metadata.total_episodes}")
    episodes = list(range(metadata.total_episodes))
    dataset = LeRobotDataset(
        args.repo_id,
        root=root,
        episodes=episodes,
        delta_timestamps=resolve_delta_timestamps(config, metadata),
        return_uint8=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    policy_class = get_policy_class(config.type)
    with contextlib.redirect_stdout(io.StringIO()):
        policy = policy_class.from_pretrained(checkpoint, config=config).to(args.device).eval()
    preprocessor, _ = make_pre_post_processors(
        config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    streaming_model = None
    if config.type == "streaming_flow_v2":
        streaming_model = policy.ema_model if policy.ema_model is not None else policy.model

    weighted_loss = 0.0
    samples = 0
    batches = 0
    for batch_index, batch in enumerate(loader):
        if args.max_batches and batch_index >= args.max_batches:
            break
        batch_size = int(batch["action"].shape[0])
        for camera_key in metadata.camera_keys:
            if batch[camera_key].dtype == torch.uint8:
                batch[camera_key] = batch[camera_key].float() / 255.0
        batch = preprocessor(batch)
        torch.manual_seed(args.seed + batch_index)
        if args.device.startswith("cuda"):
            torch.cuda.manual_seed_all(args.seed + batch_index)
        use_amp = args.device.startswith("cuda") and bool(config.use_amp)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=use_amp
        ):
            if config.type == "streaming_flow_v2":
                prepared = policy._prepare_image_batch(batch)
                loss, _ = streaming_model.compute_loss(prepared, reduction="mean")
            else:
                loss, _ = policy.forward(batch)
        weighted_loss += float(loss.detach().cpu()) * batch_size
        samples += batch_size
        batches += 1

    if samples == 0:
        raise RuntimeError("No samples were evaluated")
    result = {
        "checkpoint": str(checkpoint),
        "policy_type": config.type,
        "metric": "deterministic teacher-forced checkpoint loss",
        "warning": "All 301 evaluated episodes were also used for training; this is not held-out validation.",
        "episodes": len(episodes),
        "samples": samples,
        "batches": batches,
        "seed": args.seed,
        "loss": weighted_loss / samples,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
