#!/usr/bin/env python3
"""Package selected Streaming V2 policies and verified training provenance."""

import hashlib
import json
import math
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from correct_screw_dataset_index_stats import repair

ROOT = Path("/mnt/data/ningan/screw_datasets/outputs/streaming_flow_v2_screw_gripper25_20261005")
DATA = Path("/mnt/data/ningan/screw_datasets/cut_pull_screw_all_hover_gripper25_merged_20261005")
DELIVERY = Path("/home/ningan/screw_gripper25_streaming_flow_v2_checkpoints_20261005")
VARIANTS = ["8_1", "16_1"]
STEPS = [5000, 10000, 15000, 20000]


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    assert not DELIVERY.exists(), DELIVERY
    dataset = json.loads((DATA / "PREPARE_REPORT.json").read_text())
    selection = {}
    for v in VARIANTS:
        assert (ROOT / f"JOB_DONE_{v}").read_text().strip() == "success"
        train = [json.loads(line) for line in (ROOT / v / "train_metrics.jsonl").read_text().splitlines()]
        assert train[-1]["step"] == 20000
        assert all(math.isfinite(x["loss"]) and math.isfinite(x["grad_norm"]) for x in train)
        results = []
        for step in STEPS:
            val = json.loads((ROOT / f"val_{v}_{step:06}.json").read_text())
            assert (
                val["frames"] == dataset["validation_frames"]
                and val["episodes"] == dataset["validation_episodes"]
            )
            assert math.isfinite(val["flow_loss"]) and math.isfinite(
                val["first_action_mae_per_normalized_joint"]
            )
            results.append((step, val))
        step, val = min(results, key=lambda pair: pair[1]["flow_loss"])
        source = ROOT / v / "checkpoints" / f"{step:06}" / "pretrained_model"
        cfg = json.loads((source / "config.json").read_text())
        assert cfg["type"] == "streaming_flow_v2" and cfg["n_action_steps"] == int(v.split("_")[0])
        assert cfg["chunk_size"] == cfg["n_action_steps"] + 1 and cfg["execution_horizon"] == 1
        assert cfg["use_separate_rgb_encoder_per_camera"] and not cfg["freeze_vision_encoder"]
        assert cfg["use_previous_action_alignment"] and cfg["rollout_initial_action_mode"] == "state"
        selection[v] = {
            "step": step,
            "source": str(source),
            "validation": val,
            "last_train_minibatch_loss": train[-1]["loss"],
        }
    repair(DATA)
    DELIVERY.mkdir()
    checksums = {}
    for v, s in selection.items():
        src = Path(s["source"])
        dst = DELIVERY / v
        shutil.copytree(src, dst)
        files = {}
        for f in sorted(src.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(src)
            h = digest(f)
            assert digest(dst / rel) == h, (v, rel)
            files[str(rel)] = {"bytes": f.stat().st_size, "sha256": h}
        checksums[v] = {"selected_step": s["step"], "source": str(src), "files": files}
        (DELIVERY / f"validation_{v}.json").write_text(json.dumps(s["validation"], indent=2) + "\n")
        best = ROOT / v / "checkpoints/best_offline"
        assert not best.exists() and not best.is_symlink()
        best.symlink_to(f"{s['step']:06}")
        benchmark = subprocess.check_output(
            [
                sys.executable,
                "/home/ningan/lerobot/scripts/bench_screw_hover5_streaming_flow_v2_inference.py",
                str(dst),
                "--root",
                str(DATA),
                "--repo-id",
                "ningan/cut_pull_screw_all_hover_gripper25_merged_20261005",
                "--device",
                "cuda",
                "--trials",
                "30",
            ],
            text=True,
        )
        parsed = json.loads(benchmark)
        assert parsed["action_shape"] == [1, 6] and not parsed["hardware_loop_included"]
        (ROOT / f"bench_{v}_selected.json").write_text(benchmark)
        (DELIVERY / f"inference_benchmark_{v}.json").write_text(benchmark)
    (DELIVERY / "CHECKSUMS.json").write_text(json.dumps(checksums, indent=2) + "\n")
    (ROOT / "SELECTION.json").write_text(json.dumps(selection, indent=2) + "\n")
    for src, name in [
        ("PREPARE_REPORT.json", "DATASET_REPORT.json"),
        ("MERGE_REPORT.json", "DATASET_MERGE_REPORT.json"),
        ("PRIOR_DATASET_COMPARISON.json", "PRIOR_DATASET_COMPARISON.json"),
        ("meta/gripper25_split_20261005.json", "DATASET_SPLIT.json"),
    ]:
        shutil.copy2(DATA / src, DELIVERY / name)
    shutil.copy2(ROOT / "source_snapshot/git_head.txt", DELIVERY / "lerobot_base_commit.txt")
    shutil.copy2(
        ROOT / "source_snapshot/streaming_flow_local_changes.patch", DELIVERY / "streaming_flow_compat.patch"
    )
    shutil.copytree(ROOT / "source_snapshot/streaming_flow", DELIVERY / "runtime_source/streaming_flow")
    shutil.copy2(ROOT / "RUN_CONTRACT.json", DELIVERY / "RUN_CONTRACT.json")
    shutil.copy2(ROOT / "SMOKE_CHECKS.json", DELIVERY / "SMOKE_CHECKS.json")
    shutil.copy2(DATA / "INDEX_STATISTICS_REPAIR.json", DELIVERY / "INDEX_STATISTICS_REPAIR.json")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, v in zip(axes, VARIANTS, strict=True):
        train = [json.loads(line) for line in (ROOT / v / "train_metrics.jsonl").read_text().splitlines()]
        ax.plot(
            [x["step"] for x in train],
            [x["loss"] for x in train],
            alpha=0.5,
            label="Sampled train mini-batch",
        )
        vals = [json.loads((ROOT / f"val_{v}_{step:06}.json").read_text())["flow_loss"] for step in STEPS]
        ax.plot(STEPS, vals, "o-", label="Full held-out EMA")
        ax.set(xlabel="Optimizer step", ylabel="Flow loss", title=v, yscale="log")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.savefig(DELIVERY / "loss_curves.png", dpi=160)
    plt.close(fig)
    bench = {v: json.loads((ROOT / f"bench_{v}_selected.json").read_text()) for v in VARIANTS}
    table = "\n".join(
        f"| {v} | {s['step']:,} | {s['validation']['flow_loss']:.6f} | {s['validation']['first_action_mae_per_normalized_joint']:.6f} | {s['validation']['gripper_close_abs_error_normalized']:.6f} |"
        for v, s in selection.items()
    )
    report = f"""# Screw gripper25 Streaming Flow V2 — 2026-10-05

Both policies completed 20,000 optimizer updates, with finite logged loss and gradients.
The delivered model for each horizon was selected by the lowest full held-out EMA flow
loss among steps 5,000, 10,000, 15,000 and 20,000. This is offline checkpoint selection.

| Predicted/executed | Selected step | Held-out flow loss | First-action MAE per normalized joint | Gripper-close MAE normalized |
| --- | ---: | ---: | ---: | ---: |
{table}

The ZIP contained {dataset["episodes"]} independent single-episode LeRobot v3 datasets,
with {dataset["frames"]:,} frames at {dataset["fps"]} FPS. They were merged in numeric episode
order without changing any action/state values or resampling the supplied sequences.
Front and side H264 videos were concatenated by packet stream copy without re-encoding.
The source metadata incorrectly declared AV1; the merged metadata declares H264.
The merged dataset reloads with LeRobot, and sampled episode boundary images decode.
Two task/source labels each contain 100 episodes. The two policies do not condition on task text.

The exact source-stratified split matches the preceding histogram-balanced experiment:
{dataset["train_episodes"]} training episodes / {dataset["train_frames"]:,} frames and
{dataset["validation_episodes"]} held-out episodes / {dataset["validation_frames"]:,} frames,
seed 100000. State/action normalization statistics are fitted on training episodes only.
All 200 episodes have identical action/state values to the previous histogram-balanced
dataset, but decoded image content differs in both cameras. Full decoded video frame
hash comparison and first-frame RGB pixel diagnostics are in PRIOR_DATASET_COMPARISON.json.
This experiment changes the image input while retaining action demonstrations and split.

The source preprocessing CSVs report a gripper closing phase averaging about 25% of each
sequence. A different diagnostic, normalized per-frame gripper decrease below -0.02,
counts {100 * dataset["normalized_gripper_closing_transition_fraction"]:.2f}% of transitions;
these measure phase duration and sufficiently large changes, respectively. No extra
sample weighting, smoothing, interpolation or temporal editing was added.

Both variants use two fully trainable independent ImageNet ResNet18 camera encoders,
2 observation frames, batch 16, seed 100000, AdamW lr 1e-4, cosine scheduling with
500 warmup steps, EMA, 4 flow training points, adaptive frequency and step scaling.
RGB is resized from 480x640 to 240x320 and randomly cropped to 228x304 for training;
inference uses the center crop and saved preprocessing. State/actions use per-dimension
MIN_MAX normalization. W&B is disabled; local metrics are retained. `chunk_size=9/17`
contains the previous action plus 8/16 new commands; both execute one before replanning.
Initial previous action uses the observed joint state.

The directories 8_1 and 16_1 contain native LeRobot model.safetensors, config.json,
train_config.json and complete pre/postprocessors. Load each complete directory with
the matching streaming_flow_v2 source. The runtime source snapshot, base Git commit,
compatibility patch and SHA256 manifest are included. Preserve camera viewpoints,
follower motor/gripper calibration, and the state/action order: shoulder_pan.pos,
shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos, gripper.pos.
Inputs are observation.images.front, observation.images.side and observation.state.
Reset the policy per episode and maintain previous-action alignment during replanning.

The delivered copies were reloaded for inference and produced finite 6D actions.
On this RTX5090 host, preprocessor -> policy -> postprocessor median latency was
{bench["8_1"]["pipeline_ms_median"]:.1f} ms for 8/1 and {bench["16_1"]["pipeline_ms_median"]:.1f} ms for 16/1.
P95 was {bench["8_1"]["pipeline_ms_p95"]:.1f} / {bench["16_1"]["pipeline_ms_p95"]:.1f} ms.
Camera capture and motor communication are excluded. A synchronous 30Hz control loop
has a 33.3ms total budget; measure the entire loop on the deployment host.

Validation is teacher-forced offline evaluation, including demonstration previous actions.
No physical robot task success rate was measured. Flow loss scales differ between prediction
horizons, so their absolute values are not a direct cross-horizon ranking.

Dataset: {DATA}
Training outputs: {ROOT}
Launcher: /home/ningan/lerobot/scripts/train_screw_gripper25_streaming_flow_v2.sh
Source archive: {dataset["source_archive"]}
Source archive SHA256: {dataset["source_archive_sha256"]}
Selected and final checkpoints and full optimizer state are retained in the training outputs.
Each copied model file was verified against its source SHA256. ZIP CRC and reload checks passed.
"""
    (DELIVERY / "TRAINING_RESULT.md").write_text(report)
    (ROOT / "TRAINING_RESULT.md").write_text(report)
    archive = DELIVERY.with_suffix(".zip")
    assert not archive.exists(), archive
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for p in sorted(DELIVERY.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(DELIVERY.parent))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    archive.with_suffix(".zip.sha256").write_text(f"{digest(archive)}  {archive.name}\n")
    print(
        json.dumps(
            {
                "selected_steps": {v: s["step"] for v, s in selection.items()},
                "directory": str(DELIVERY),
                "zip": str(archive),
                "zip_bytes": archive.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
