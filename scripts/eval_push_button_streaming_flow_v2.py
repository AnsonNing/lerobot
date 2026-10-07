#!/usr/bin/env python3
"""Teacher-forced held-out flow loss for the merged push-button dataset.

This is an offline diagnostic; only a real rollout can measure button-press success.
"""

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
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.streaming_flow.modeling_streaming_flow_v2 import StreamingFlowPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path, help="Directory containing model.safetensors")
    parser.add_argument(
        "--data-root", type=Path, default=Path("/mnt/data/ningan/button_datasets/push_button_merged_260914")
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=0, help="0 evaluates every held-out frame")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    split = json.loads((args.data_root / "meta/button_split_260914.json").read_text())
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    if cfg.type != "streaming_flow_v2":
        raise ValueError(f"Expected streaming_flow_v2, got {cfg.type}")
    cfg.device = args.device
    meta = LeRobotDatasetMetadata("HZL/push_button_merged_260914", root=args.data_root)
    delta = resolve_delta_timestamps(cfg, meta)
    with contextlib.redirect_stdout(io.StringIO()):
        policy = StreamingFlowPolicy.from_pretrained(args.checkpoint, config=cfg).to(args.device).eval()
    preprocessor, _ = make_pre_post_processors(
        cfg,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    output = {"checkpoint": str(args.checkpoint), "metric": "teacher_forced_sfp_loss_ema", "groups": {}}
    for group, source_episode_range in (("old", range(184)), ("new", range(184, 224))):
        episodes = sorted(set(split["validation"]) & set(source_episode_range))
        dataset = LeRobotDataset(
            "HZL/push_button_merged_260914",
            root=args.data_root,
            episodes=episodes,
            delta_timestamps=delta,
            return_uint8=True,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        loss_sum = 0.0
        frames = 0
        for batch_index, batch in enumerate(loader):
            if args.max_batches and batch_index >= args.max_batches:
                break
            for camera_key in meta.camera_keys:
                if batch[camera_key].dtype == torch.uint8:
                    batch[camera_key] = batch[camera_key].float() / 255.0
            batch = preprocessor(batch)
            torch.manual_seed(100000 + batch_index)
            with torch.inference_mode(), torch.autocast("cuda", enabled=args.device.startswith("cuda")):
                prepared = policy._prepare_image_batch(batch)
                model = policy.ema_model if policy.ema_model is not None else policy.model
                losses, _ = model.compute_loss(prepared, reduction="none")
            loss_sum += float(losses.sum().cpu())
            frames += len(losses)
        output["groups"][group] = {
            "episodes": len(episodes),
            "frames": frames,
            "loss": loss_sum / frames if frames else None,
        }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
