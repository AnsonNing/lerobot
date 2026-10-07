#!/usr/bin/env python3
"""Select held-out checkpoints, verify delivery copies and create a download ZIP."""

import hashlib
import json
import math
import shutil
import zipfile
from pathlib import Path

ROOT = Path("/home/ningan/training_outputs/screw_histbalance_20261004")
DATA = Path(
    "/mnt/data/ningan/screw_datasets/histogram_balance_cut_pull_screw_all_hover_gripper_25_assemble_episode"
)
DELIVERY = Path("/home/ningan/screw_histbalance_streaming_flow_v2_checkpoints_20261004")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    if DELIVERY.exists():
        raise FileExistsError(DELIVERY)
    selections = {}
    for variant in ("8_1", "16_1"):
        metrics = []
        for step in (5000, 10000, 15000, 20000):
            path = ROOT / f"val_{variant}_{step:06}.json"
            result = json.loads(path.read_text())
            assert result["frames"] == 2131 and result["episodes"] == 20, path
            assert math.isfinite(result["flow_loss"]) and math.isfinite(
                result["first_action_mae_per_normalized_joint"]
            )
            assert set(result["by_task_index"]) == {"0", "1"}
            metrics.append((step, result))
        step, result = min(metrics, key=lambda item: item[1]["flow_loss"])
        source = ROOT / variant / "checkpoints" / f"{step:06}" / "pretrained_model"
        cfg = json.loads((source / "config.json").read_text())
        assert cfg["type"] == "streaming_flow_v2"
        assert cfg["n_action_steps"] == int(variant.split("_")[0])
        assert cfg["execution_horizon"] == 1 and cfg["use_separate_rgb_encoder_per_camera"]
        selections[variant] = {"step": step, "source": str(source), "validation": result}

    DELIVERY.mkdir()
    manifest = {}
    for variant, selected in selections.items():
        source = Path(selected["source"])
        target = DELIVERY / variant
        shutil.copytree(source, target)
        checksums = {}
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            digest = sha256(path)
            assert sha256(target / relative) == digest, relative
            checksums[str(relative)] = {"bytes": path.stat().st_size, "sha256": digest}
        manifest[variant] = {"selected_step": selected["step"], "source": str(source), "files": checksums}
        (DELIVERY / f"validation_{variant}.json").write_text(
            json.dumps(selected["validation"], indent=2) + "\n"
        )
        best = ROOT / variant / "checkpoints/best_offline"
        if best.exists() or best.is_symlink():
            raise FileExistsError(best)
        best.symlink_to(f"{selected['step']:06}")
    (DELIVERY / "CHECKSUMS.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (ROOT / "SELECTION.json").write_text(json.dumps(selections, indent=2) + "\n")
    shutil.copy2(DATA / "PREPARE_REPORT.json", DELIVERY / "DATASET_REPORT.json")
    shutil.copy2(DATA / "meta/histbalance_split_20261004.json", DELIVERY / "DATASET_SPLIT.json")
    shutil.copy2(
        ROOT / "source_snapshot/streaming_flow_local_changes.patch", DELIVERY / "streaming_flow_compat.patch"
    )
    shutil.copy2(ROOT / "source_snapshot/git_head.txt", DELIVERY / "lerobot_base_commit.txt")
    shutil.copytree(ROOT / "source_snapshot/streaming_flow", DELIVERY / "runtime_source/streaming_flow")
    for variant in selections:
        shutil.copy2(
            ROOT / f"bench_{variant}_selected.json", DELIVERY / f"inference_benchmark_{variant}.json"
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, variant in zip(axes, selections, strict=True):
        rows = [
            json.loads(line) for line in (ROOT / variant / "train_metrics.jsonl").read_text().splitlines()
        ]
        ax.plot([x["step"] for x in rows], [x["loss"] for x in rows], label="Train mini-batch", alpha=0.5)
        validation = [
            (step, json.loads((ROOT / f"val_{variant}_{step:06}.json").read_text()))
            for step in (5000, 10000, 15000, 20000)
        ]
        ax.plot(
            [s for s, _ in validation], [v["flow_loss"] for _, v in validation], "o-", label="Held-out EMA"
        )
        ax.set(xlabel="Optimizer step", ylabel="Flow loss", title=variant, yscale="log")
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.savefig(DELIVERY / "loss_curves.png", dpi=160)
    plt.close(fig)

    rows = []
    for variant, selected in selections.items():
        val = selected["validation"]
        rows.append(
            f"| {variant} | {selected['step']:,} | {val['flow_loss']:.6f} | {val['first_action_mae_per_normalized_joint']:.6f} | {val['gripper_close_abs_error_normalized']:.6f} |"
        )
    benchmark = {
        variant: json.loads((ROOT / f"bench_{variant}_selected.json").read_text()) for variant in selections
    }
    report = (
        """# Histogram-balanced screw Streaming Flow V2 training

Both variants finished 20,000 optimizer steps. Checkpoints were selected by
the lowest full held-out EMA flow loss within each prediction horizon.

| Variant (predicted/executed) | Selected step | Held-out flow loss | First-action MAE per normalized joint | Gripper-close MAE (normalized) |
| --- | ---: | ---: | ---: | ---: |
"""
        + "\n".join(rows)
        + """

The original archive contains 200 episodes / 21,845 frames at 30 FPS, two
640×480 RGB cameras (`front`, `side`), and six joint-position state/actions.
The two source/task labels each have 100 episodes. The seeded source-stratified
split uses 180 episodes / 19,714 frames for training and 20 episodes /
2,131 frames for validation. State/action normalization stats were computed
on training episodes only. H.264 codec metadata was corrected in the extracted
copy. No frames were dropped, retimed or interpolated in this training run.
The supplied histogram balancing was used without additional sample weighting.
Closing transitions account for about 7.4% using normalized gripper delta < -0.02.

Both runs use seed 100000, batch 16, AdamW 1e-4, cosine schedule, 500 warmup
steps, EMA, two observation frames, independent ImageNet ResNet18 camera
encoders with full fine-tuning, per-dimension MIN_MAX normalization, and four
flow training points with adaptive frequency and step scaling. RGB is resized
to 240×320 and randomly cropped to 228×304 in training, with center cropping
at inference. W&B was disabled. Trajectories contain the previous action plus
8 or 16 new actions (`chunk_size=9` or `17`); both execute one action before
replanning. Initial previous action is approximated from observed state.

Each `8_1/` or `16_1/` directory contains `model.safetensors`, `config.json`
and the saved pre/postprocessors. Use the complete directory for LeRobot
`from_pretrained`; keep the matching custom `streaming_flow_v2` implementation.
`streaming_flow_compat.patch` and `lerobot_base_commit.txt` record local source
changes and the base commit, and a full source snapshot remains in the training
output directory. Apply a source patch only to a compatible checkout.
The policy conditions on images and state; these V2 settings do not use task
text as a selectable instruction.

Inference inputs are RGB `observation.images.front` and `.side` plus six
positions ordered `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`,
`wrist_roll`, `gripper`. Preserve recording-time camera viewpoints and follower
calibration. Use the saved preprocessing and action postprocessing, and reset
the policy at every episode boundary.

These are teacher-forced offline diagnostics: the previous action comes from
the demonstration during validation. No real robot rollout or task success
rate was measured. Flow loss magnitudes should not be compared directly across
the two prediction horizons. Per-source action/gripper diagnostics are in
`validation_8_1.json` and `validation_16_1.json`.

For 8→1, the 10,000-step checkpoint had slightly lower first-action MAE
and lower gripper-close MAE than the selected 20,000-step checkpoint; it
remains in the training output for follow-up comparisons. Selection by flow
loss does not establish that every action metric or robot success improved.

"""
        + f"""The current host's saved preprocessor → policy → postprocessor pipeline
takes approximately {benchmark["8_1"]["pipeline_ms_median"]:.1f} ms median for 8→1 and
{benchmark["16_1"]["pipeline_ms_median"]:.1f} ms for 16→1. Camera capture and motor communication are excluded.
Thus 16→1 already exceeds the 33.3 ms budget for synchronous 30 Hz replanning;
8→1 leaves about {33.333 - benchmark["8_1"]["pipeline_ms_median"]:.1f} ms for hardware I/O. Measure the complete loop on the robot host.
"""
        + """

Training outputs and logs: `/home/ningan/training_outputs/screw_histbalance_20261004/`.
Training launcher: `/home/ningan/lerobot/scripts/train_screw_histbalance_streaming_flow_v2.sh`.
Dataset: `/mnt/data/ningan/screw_datasets/histogram_balance_cut_pull_screw_all_hover_gripper_25_assemble_episode`.
All delivery files copied from the selected checkpoints were verified by SHA256;
`CHECKSUMS.json` records the hashes. The delivered copies preserve the training
output checkpoints.
"""
    )
    (DELIVERY / "TRAINING_RESULT.md").write_text(report)
    (ROOT / "TRAINING_RESULT.md").write_text(report)
    archive = DELIVERY.with_suffix(".zip")
    if archive.exists():
        raise FileExistsError(archive)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for path in sorted(DELIVERY.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(DELIVERY.parent))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    archive.with_suffix(".zip.sha256").write_text(f"{sha256(archive)}  {archive.name}\n")
    print(
        json.dumps(
            {
                "selected_steps": {key: val["step"] for key, val in selections.items()},
                "directory": str(DELIVERY),
                "zip": str(archive),
                "zip_bytes": archive.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
