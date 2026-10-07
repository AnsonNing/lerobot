#!/usr/bin/env python3
"""Held-out flow loss and action-motion diagnostics for the screw task."""

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.streaming_flow.modeling_streaming_flow_v2 import StreamingFlowPolicy

ROOT = Path("/mnt/data/ningan/screw_datasets/cut_pull_screw_hover5_gripper10_assemble_episode")
REPO_ID = "ningan/cut_pull_screw_hover5_gripper10_assemble_episode"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--flow-only", action="store_true", help="Use for checkpoint selection; skip action sampling"
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--split-file", default="screw_split_260930.json")
    args = parser.parse_args()

    split = json.loads((args.root / "meta" / args.split_file).read_text())
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    if cfg.type != "streaming_flow_v2":
        raise ValueError(f"Expected streaming_flow_v2, got {cfg.type}")
    cfg.device = args.device
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.root,
        episodes=split["validation"],
        delta_timestamps=resolve_delta_timestamps(cfg, meta),
        return_uint8=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    # Mark the stable open phase before the final close. This distinguishes
    # harmless small command drift from a potentially premature close.
    action_table = pq.read_table(args.root / "data/chunk-000/file-000.parquet", columns=["action"])
    gripper = np.asarray(action_table["action"].to_pylist(), dtype=np.float32)[:, -1]
    episode_table = pq.read_table(args.root / "meta/episodes/chunk-000/file-000.parquet")
    early_close_cutoffs = {}
    for episode in split["validation"]:
        start = int(episode_table["dataset_from_index"][episode].as_py())
        end = int(episode_table["dataset_to_index"][episode].as_py())
        commands = gripper[start:end]
        tail = max(5, len(commands) // 10)
        open_value = float(np.median(commands[:tail]))
        closed_value = float(np.median(commands[-tail:]))
        onset_candidates = np.flatnonzero(
            (np.arange(len(commands)) >= len(commands) // 2)
            & (commands < open_value - 0.1 * (open_value - closed_value))
        )
        if len(onset_candidates) == 0:
            raise ValueError(f"No final gripper close detected for episode {episode}")
        early_close_cutoffs[episode] = max(0, int(onset_candidates[0]) - 5)
    with contextlib.redirect_stdout(io.StringIO()):
        policy = StreamingFlowPolicy.from_pretrained(args.checkpoint, config=cfg).to(args.device).eval()
    preprocessor, _ = make_pre_post_processors(
        cfg,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    model = policy.ema_model if policy.ema_model is not None else policy.model

    metrics = {
        "flow_loss_sum": 0.0,
        "first_action_mae_sum": 0.0,
        "first_action_gripper_close_mae_sum": 0.0,
        "first_action_gripper_close_count": 0,
        "gripper_close_abs_error_sum": 0.0,
        "gripper_close_signed_error_sum": 0.0,
        "gripper_close_predicted_decrease_count": 0,
        "gripper_predicted_decrease_count": 0,
        "gripper_false_decrease_count": 0,
        "gripper_close_true_delta_sum": 0.0,
        "gripper_close_pred_delta_sum": 0.0,
        "gripper_close_strong_decrease_count": 0,
        "gripper_false_strong_decrease_count": 0,
        "gripper_early_open_frames": 0,
        "gripper_early_strong_decrease_count": 0,
        "moving_frames": 0,
        "predicted_still_on_moving_frames": 0,
        "target_flat_chunks": 0,
        "predicted_flat_chunks": 0,
        "frames": 0,
    }
    by_task = {}
    for batch_index, batch in enumerate(loader):
        if args.max_batches and batch_index >= args.max_batches:
            break
        for camera_key in meta.camera_keys:
            if batch[camera_key].dtype == torch.uint8:
                batch[camera_key] = batch[camera_key].float() / 255.0
        task_indices = batch["task_index"].clone()
        early_open = torch.tensor(
            [
                int(frame) < early_close_cutoffs[int(episode)]
                for episode, frame in zip(
                    batch["episode_index"].tolist(), batch["frame_index"].tolist(), strict=True
                )
            ],
            device=args.device,
        )
        batch = preprocessor(batch)
        with torch.inference_mode(), torch.autocast("cuda", enabled=args.device.startswith("cuda")):
            prepared = policy._prepare_image_batch(batch)
            torch.manual_seed(100000 + batch_index)
            losses, _ = model.compute_loss(prepared, reduction="none")
            for task_index in task_indices.unique().tolist():
                selected = task_indices == task_index
                task = by_task.setdefault(str(task_index), {"frames": 0, "flow_loss_sum": 0.0})
                task["frames"] += int(selected.sum())
                task["flow_loss_sum"] += float(losses.detach().cpu()[selected].sum())
            if args.flow_only:
                metrics["flow_loss_sum"] += float(losses.sum().cpu())
                metrics["frames"] += len(losses)
                continue
            previous = batch["action"][:, 0]
            target = batch["action"][:, 1 : 1 + cfg.n_action_steps]
            predicted, _, _ = model.integrate_actions(prepared, init_action=previous.unsqueeze(1))

            first_error = (predicted[:, 0] - target[:, 0]).abs().mean(dim=-1)
            true_move = (target[:, 0] - previous).abs().amax(dim=-1)
            pred_move = (predicted[:, 0] - previous).abs().amax(dim=-1)
            moving = true_move > 0.02
            closing = (target[:, 0, -1] - previous[:, -1]) < -0.02
            gripper_error = predicted[:, 0, -1] - target[:, 0, -1]
            predicted_gripper_change = predicted[:, 0, -1] - previous[:, -1]
            predicted_closing = predicted_gripper_change < -0.005
            predicted_strong_closing = predicted_gripper_change < -0.02
            target_range = (target.amax(dim=1) - target.amin(dim=1)).amax(dim=-1)
            pred_range = (predicted.amax(dim=1) - predicted.amin(dim=1)).amax(dim=-1)

        for task_index in task_indices.unique().tolist():
            selected = (task_indices == task_index).to(args.device)
            task = by_task[str(task_index)]
            for key in (
                "first_action_mae_sum",
                "close_abs_error_sum",
                "closing_frames",
                "early_open_frames",
                "early_close_frames",
            ):
                task.setdefault(key, 0.0)
            task["first_action_mae_sum"] += float(first_error[selected].sum().cpu())
            task["close_abs_error_sum"] += float(gripper_error[selected & closing].abs().sum().cpu())
            task["closing_frames"] += int((selected & closing).sum().cpu())
            task["early_open_frames"] += int((selected & early_open).sum().cpu())
            task["early_close_frames"] += int((selected & early_open & predicted_strong_closing).sum().cpu())

        metrics["flow_loss_sum"] += float(losses.sum().cpu())
        metrics["first_action_mae_sum"] += float(first_error.sum().cpu())
        metrics["first_action_gripper_close_mae_sum"] += float(first_error[closing].sum().cpu())
        metrics["first_action_gripper_close_count"] += int(closing.sum().cpu())
        metrics["gripper_close_abs_error_sum"] += float(gripper_error[closing].abs().sum().cpu())
        metrics["gripper_close_signed_error_sum"] += float(gripper_error[closing].sum().cpu())
        metrics["gripper_close_predicted_decrease_count"] += int((closing & predicted_closing).sum().cpu())
        metrics["gripper_predicted_decrease_count"] += int(predicted_closing.sum().cpu())
        metrics["gripper_false_decrease_count"] += int((~closing & predicted_closing).sum().cpu())
        metrics["gripper_close_true_delta_sum"] += float(
            (target[:, 0, -1] - previous[:, -1])[closing].sum().cpu()
        )
        metrics["gripper_close_pred_delta_sum"] += float(predicted_gripper_change[closing].sum().cpu())
        metrics["gripper_close_strong_decrease_count"] += int(
            (closing & predicted_strong_closing).sum().cpu()
        )
        metrics["gripper_false_strong_decrease_count"] += int(
            (~closing & predicted_strong_closing).sum().cpu()
        )
        metrics["gripper_early_open_frames"] += int(early_open.sum().cpu())
        metrics["gripper_early_strong_decrease_count"] += int(
            (early_open & predicted_strong_closing).sum().cpu()
        )
        metrics["moving_frames"] += int(moving.sum().cpu())
        metrics["predicted_still_on_moving_frames"] += int((moving & (pred_move < 0.005)).sum().cpu())
        metrics["target_flat_chunks"] += int((target_range < 0.02).sum().cpu())
        metrics["predicted_flat_chunks"] += int((pred_range < 0.02).sum().cpu())
        metrics["frames"] += len(losses)

    frames = metrics["frames"]
    by_task_result = {}
    for task_index, task in by_task.items():
        result = {"frames": task["frames"], "flow_loss": task["flow_loss_sum"] / task["frames"]}
        if not args.flow_only:
            result.update(
                {
                    "first_action_mae_per_normalized_joint": task["first_action_mae_sum"] / task["frames"],
                    "gripper_closing_frames": int(task["closing_frames"]),
                    "gripper_close_abs_error_normalized": task["close_abs_error_sum"] / task["closing_frames"]
                    if task["closing_frames"]
                    else None,
                    "gripper_early_strong_decrease_fraction": task["early_close_frames"]
                    / task["early_open_frames"]
                    if task["early_open_frames"]
                    else None,
                }
            )
        by_task_result[task_index] = result
    if args.flow_only:
        print(
            json.dumps(
                {
                    "checkpoint": str(args.checkpoint),
                    "metric": "teacher_forced_ema_flow_loss",
                    "episodes": len(split["validation"]),
                    "frames": frames,
                    "flow_loss": metrics["flow_loss_sum"] / frames,
                    "by_task_index": by_task_result,
                },
                indent=2,
            )
        )
        return
    result = {
        "checkpoint": str(args.checkpoint),
        "metric": "teacher_forced_ema_flow_and_first_action",
        "episodes": len(split["validation"]),
        "frames": frames,
        "flow_loss": metrics["flow_loss_sum"] / frames,
        "by_task_index": by_task_result,
        "first_action_mae_per_normalized_joint": metrics["first_action_mae_sum"] / frames,
        "gripper_closing_frames": metrics["first_action_gripper_close_count"],
        "first_action_mae_on_gripper_closing": (
            metrics["first_action_gripper_close_mae_sum"] / metrics["first_action_gripper_close_count"]
            if metrics["first_action_gripper_close_count"]
            else None
        ),
        "gripper_close_abs_error_normalized": (
            metrics["gripper_close_abs_error_sum"] / metrics["first_action_gripper_close_count"]
            if metrics["first_action_gripper_close_count"]
            else None
        ),
        "gripper_close_signed_error_normalized": (
            metrics["gripper_close_signed_error_sum"] / metrics["first_action_gripper_close_count"]
            if metrics["first_action_gripper_close_count"]
            else None
        ),
        "gripper_close_predicted_decrease_fraction": (
            metrics["gripper_close_predicted_decrease_count"] / metrics["first_action_gripper_close_count"]
            if metrics["first_action_gripper_close_count"]
            else None
        ),
        "gripper_close_true_mean_delta_normalized": metrics["gripper_close_true_delta_sum"]
        / metrics["first_action_gripper_close_count"]
        if metrics["first_action_gripper_close_count"]
        else None,
        "gripper_close_pred_mean_delta_normalized": metrics["gripper_close_pred_delta_sum"]
        / metrics["first_action_gripper_close_count"]
        if metrics["first_action_gripper_close_count"]
        else None,
        "gripper_close_strong_decrease_recall": metrics["gripper_close_strong_decrease_count"]
        / metrics["first_action_gripper_close_count"]
        if metrics["first_action_gripper_close_count"]
        else None,
        "gripper_false_strong_decrease_fraction_of_nonclosing": metrics["gripper_false_strong_decrease_count"]
        / (frames - metrics["first_action_gripper_close_count"])
        if frames > metrics["first_action_gripper_close_count"]
        else None,
        "gripper_early_open_frames": metrics["gripper_early_open_frames"],
        "gripper_early_strong_decrease_fraction": metrics["gripper_early_strong_decrease_count"]
        / metrics["gripper_early_open_frames"]
        if metrics["gripper_early_open_frames"]
        else None,
        "gripper_predicted_decrease_count": metrics["gripper_predicted_decrease_count"],
        "gripper_false_decrease_fraction_of_nonclosing": (
            metrics["gripper_false_decrease_count"] / (frames - metrics["first_action_gripper_close_count"])
            if frames > metrics["first_action_gripper_close_count"]
            else None
        ),
        "gripper_predicted_decrease_precision": (
            metrics["gripper_close_predicted_decrease_count"] / metrics["gripper_predicted_decrease_count"]
            if metrics["gripper_predicted_decrease_count"]
            else None
        ),
        "moving_frames": metrics["moving_frames"],
        "predicted_still_given_moving_fraction": (
            metrics["predicted_still_on_moving_frames"] / metrics["moving_frames"]
            if metrics["moving_frames"]
            else None
        ),
        "target_flat_chunk_fraction": metrics["target_flat_chunks"] / frames,
        "predicted_flat_chunk_fraction": metrics["predicted_flat_chunks"] / frames,
        "thresholds": {
            "moving_action_max_abs_delta": 0.02,
            "predicted_still_max_abs_delta": 0.005,
            "flat_chunk_max_joint_range": 0.02,
            "strong_gripper_decrease_delta": -0.02,
            "early_open_phase": "Before 10% of the final gripper command closing excursion, minus five frames",
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
