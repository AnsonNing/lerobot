#!/usr/bin/env python3
"""Validate and prepare the supplied histogram-balanced screw LeRobot archive."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(
    "/mnt/data/ningan/screw_datasets/histogram_balance_cut_pull_screw_all_hover_gripper_25_assemble_episode"
)
ARCHIVE = Path("/home/ningan/histogram_balance_cut_pull_screw_all_hover_gripper_25_assemble_episode.zip")


def main():
    global ROOT, ARCHIVE
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--split-file", default="histbalance_split_20261004.json")
    args = parser.parse_args()
    ROOT, ARCHIVE = args.root, args.archive
    info_path = ROOT / "meta/info.json"
    info = json.loads(info_path.read_text())
    stats_path = ROOT / "meta/stats.json"
    stats = json.loads(stats_path.read_text())
    data = pq.read_table(ROOT / "data/chunk-000/file-000.parquet")
    episodes = pq.read_table(ROOT / "meta/episodes/chunk-000/file-000.parquet")
    tasks = pq.read_table(ROOT / "meta/tasks.parquet").to_pydict()
    action = np.asarray(data["action"].to_pylist(), dtype=np.float32)
    state = np.asarray(data["observation.state"].to_pylist(), dtype=np.float32)
    starts = episodes["dataset_from_index"].to_numpy()
    ends = episodes["dataset_to_index"].to_numpy()
    lengths = episodes["length"].to_numpy()
    assert len(episodes) == info["total_episodes"]
    assert len(data) == info["total_frames"]
    assert action.shape == state.shape == (len(data), 6)
    assert np.isfinite(action).all() and np.isfinite(state).all()
    assert np.array_equal(starts, np.r_[0, np.cumsum(lengths)[:-1]])
    assert np.array_equal(ends, np.cumsum(lengths))
    assert np.array_equal(data["index"].to_numpy(), np.arange(len(data)))
    source_episodes = {}
    fingerprints = set()
    transitions = []
    for episode, (start, end) in enumerate(zip(starts, ends, strict=True)):
        assert np.all(data["episode_index"].to_numpy()[start:end] == episode)
        assert np.array_equal(data["frame_index"].to_numpy()[start:end], np.arange(end - start))
        assert np.allclose(
            data["timestamp"].to_numpy()[start:end], np.arange(end - start) / info["fps"], atol=1e-4
        )
        source = np.unique(data["task_index"].to_numpy()[start:end])
        assert len(source) == 1
        source_episodes.setdefault(int(source[0]), []).append(episode)
        fingerprint = hashlib.sha256(action[start:end].tobytes() + state[start:end].tobytes()).hexdigest()
        assert fingerprint not in fingerprints, f"Duplicate episode {episode}"
        fingerprints.add(fingerprint)
        transitions.append(np.diff(action[start:end], axis=0))
        for camera in ("front", "side"):
            key = f"videos/observation.images.{camera}"
            assert abs(episodes[f"{key}/from_timestamp"][episode].as_py() - start / info["fps"]) < 1e-4
            assert abs(episodes[f"{key}/to_timestamp"][episode].as_py() - end / info["fps"]) < 1e-4

    video = {}
    metadata_fixes = []
    for camera in ("front", "side"):
        key = f"observation.images.{camera}"
        output = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,avg_frame_rate,nb_frames",
                "-of",
                "json",
                str(ROOT / f"videos/{key}/chunk-000/file-000.mp4"),
            ],
            text=True,
        )
        video[camera] = json.loads(output)["streams"][0]
        assert int(video[camera]["nb_frames"]) == len(data)
        assert (video[camera]["width"], video[camera]["height"]) == (640, 480)
        assert video[camera]["avg_frame_rate"] == "30/1"
        if info["features"][key]["info"]["video.codec"] != video[camera]["codec_name"]:
            metadata_fixes.append(f"{key}: video.codec corrected to {video[camera]['codec_name']}")
        info["features"][key]["info"]["video.codec"] = video[camera]["codec_name"]

    rng = np.random.default_rng(100000)
    validation = []
    for candidates in source_episodes.values():
        # Hold out one episode per consecutive source decile.
        for start in range(0, len(candidates), 10):
            validation.append(int(rng.choice(candidates[start : start + 10])))
    train = sorted(set(range(len(episodes))) - set(validation))
    split = {
        "seed": 100000,
        "train": train,
        "validation": sorted(validation),
        "selection": "One seeded episode per consecutive group of ten within each task/source",
    }
    (ROOT / "meta" / args.split_file).write_text(json.dumps(split, indent=2) + "\n")
    selected = np.concatenate([np.arange(starts[i], ends[i]) for i in train])
    for key, values in (("action", action), ("observation.state", state)):
        sample = values[selected].astype(np.float64)
        stats[key] = {
            "min": sample.min(axis=0).tolist(),
            "max": sample.max(axis=0).tolist(),
            "mean": sample.mean(axis=0).tolist(),
            "std": sample.std(axis=0).tolist(),
            "count": [len(sample)],
            **{
                f"q{int(q * 100):02}": np.quantile(sample, q, axis=0).tolist()
                for q in (0.01, 0.1, 0.5, 0.9, 0.99)
            },
        }
    info_path.write_text(json.dumps(info, indent=4) + "\n")
    stats_path.write_text(json.dumps(stats, indent=4) + "\n")
    delta = np.concatenate(transitions)
    normalized_gripper_delta = 2 * delta[:, -1] / (action[:, -1].max() - action[:, -1].min())
    report = {
        "source_archive": str(ARCHIVE),
        "source_archive_sha256": hashlib.file_digest(ARCHIVE.open("rb"), "sha256").hexdigest(),
        "root": str(ROOT),
        "episodes": len(episodes),
        "frames": len(data),
        "fps": info["fps"],
        "tasks": tasks,
        "video": video,
        "metadata_fixes": metadata_fixes,
        "train_episodes": len(train),
        "train_frames": len(selected),
        "validation_episodes": len(validation),
        "validation_frames": len(data) - len(selected),
        "episode_length_min_median_max": [int(lengths.min()), float(np.median(lengths)), int(lengths.max())],
        "normalized_gripper_closing_transition_fraction": float(np.mean(normalized_gripper_delta < -0.02)),
        "stationary_all_joint_transition_fraction_tolerance_0_01": float(
            np.mean(np.max(np.abs(delta), axis=1) < 0.01)
        ),
        "normalization": "State/action stats use train episodes only; internal ImageNet normalization for RGB",
        "extra_sample_weighting": None,
        "temporal_edit": "None; supplied action/state/video sequence retained",
    }
    (ROOT / "PREPARE_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
