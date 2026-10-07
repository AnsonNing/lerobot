#!/usr/bin/env python3
"""Audit the extracted screw dataset and prepare a reproducible local training copy."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path("/mnt/data/ningan/screw_datasets/cut_pull_screw_hover5_gripper10_assemble_episode")
ARCHIVE = Path("/home/ningan/cut_pull_screw_hover5_gripper10_assemble_episode-20260930T130545Z-1-001.zip")
SEED = 100000


def video_info(path: Path) -> dict:
    result = subprocess.run(
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
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)["streams"][0]


def main() -> None:
    meta_dir = ROOT / "meta"
    info_path = meta_dir / "info.json"
    stats_path = meta_dir / "stats.json"
    info = json.loads(info_path.read_text())
    stats = json.loads(stats_path.read_text())
    data = pq.read_table(ROOT / "data/chunk-000/file-000.parquet")
    episodes = pq.read_table(meta_dir / "episodes/chunk-000/file-000.parquet")
    action = np.asarray(data["action"].to_pylist(), dtype=np.float64)
    state = np.asarray(data["observation.state"].to_pylist(), dtype=np.float64)
    lengths = episodes["length"].to_numpy()
    starts = episodes["dataset_from_index"].to_numpy()
    ends = episodes["dataset_to_index"].to_numpy()
    assert len(lengths) == 100 and len(action) == len(state) == 19133
    assert info["total_episodes"] == len(lengths) and info["total_frames"] == len(action)
    assert np.all(np.isfinite(action)) and np.all(np.isfinite(state))
    assert np.array_equal(starts, np.r_[0, np.cumsum(lengths)[:-1]])
    assert np.array_equal(ends, np.cumsum(lengths))
    for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        assert np.all(data["episode_index"].to_numpy()[start:end] == index)
        assert np.array_equal(data["frame_index"].to_numpy()[start:end], np.arange(end - start))

    videos = {}
    for camera in ("front", "side"):
        key = f"observation.images.{camera}"
        path = ROOT / f"videos/{key}/chunk-000/file-000.mp4"
        videos[camera] = video_info(path)
        assert videos[camera]["codec_name"] == "h264"
        assert int(videos[camera]["nb_frames"]) == len(action)
        assert (int(videos[camera]["width"]), int(videos[camera]["height"])) == (640, 480)
        assert videos[camera]["avg_frame_rate"] == "30/1"
        info["features"][key]["info"]["video.codec"] = videos[camera]["codec_name"]

    # Sample one held-out episode from each index decile. Keep episodes with
    # unusual early gripper adjustments (52, 77) in training.
    rng = np.random.default_rng(SEED)
    validation = []
    for first in range(0, len(lengths), 10):
        candidates = [index for index in range(first, first + 10) if index not in (52, 77)]
        validation.append(int(rng.choice(candidates)))
    train = sorted(set(range(len(lengths))) - set(validation))
    split = {
        "seed": SEED,
        "train": train,
        "validation": sorted(validation),
        "selection": "One seeded episode per consecutive group of ten; episodes 52 and 77 retained in training",
    }
    (meta_dir / "screw_split_260930.json").write_text(json.dumps(split, indent=2) + "\n")

    selected = np.concatenate([np.arange(starts[i], ends[i]) for i in train])
    for key, values in (("action", action), ("observation.state", state)):
        sample = values[selected]
        stats[key] = {
            "min": sample.min(axis=0).tolist(),
            "max": sample.max(axis=0).tolist(),
            "mean": sample.mean(axis=0).tolist(),
            "std": sample.std(axis=0).tolist(),
            "count": [int(len(sample))],
            **{
                f"q{int(quantile * 100):02d}": np.quantile(sample, quantile, axis=0).tolist()
                for quantile in (0.01, 0.1, 0.5, 0.9, 0.99)
            },
        }
    # The policy uses identity image normalization, so image statistics remain
    # as supplied by the source dataset. Save only the corrected local copy.
    info_path.write_text(json.dumps(info, indent=4) + "\n")
    stats_path.write_text(json.dumps(stats, indent=4) + "\n")

    digest = hashlib.file_digest(ARCHIVE.open("rb"), "sha256").hexdigest()
    report = {
        "source_zip": str(ARCHIVE),
        "source_zip_sha256": digest,
        "root": str(ROOT),
        "episodes": len(lengths),
        "frames": len(action),
        "fps": info["fps"],
        "video": videos,
        "train_episodes": len(train),
        "train_frames": int(len(selected)),
        "validation_episodes": len(validation),
        "validation_frames": int(len(action) - len(selected)),
        "episode_length_min_median_max": [int(lengths.min()), float(np.median(lengths)), int(lengths.max())],
        "metadata_fixes": ["Both video.codec fields corrected from av1 to h264"],
        "normalization": "State and action statistics recomputed on the 90 training episodes",
    }
    (ROOT / "PREPARE_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
