#!/usr/bin/env python3
"""Merge independent single-episode datasets without re-encoding or retiming."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from correct_screw_dataset_index_stats import repair

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.io_utils import load_stats, write_stats
from lerobot.datasets.video_utils import concatenate_video_files

SOURCE = Path(
    "/mnt/data/ningan/screw_datasets/cut_pull_screw_all_hover_gripper25_source_20261005/cut_pull_screw_all_hover_gripper_25_each_episode"
)
TARGET = Path("/mnt/data/ningan/screw_datasets/cut_pull_screw_all_hover_gripper25_merged_20261005")
PRIOR = Path(
    "/mnt/data/ningan/screw_datasets/histogram_balance_cut_pull_screw_all_hover_gripper_25_assemble_episode"
)


def main():
    global SOURCE, TARGET, PRIOR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    parser.add_argument("--output-root", type=Path, default=TARGET)
    parser.add_argument(
        "--comparison-root",
        type=Path,
        default=None,
        help="Optional previous merged dataset for action/state comparison",
    )
    parser.add_argument("--expected-episodes", type=int, default=200)
    args = parser.parse_args()
    SOURCE, TARGET, PRIOR = args.source_root, args.output_root, args.comparison_root
    assert not TARGET.exists(), TARGET
    roots = sorted(SOURCE.glob("episode_[0-9][0-9][0-9]"))
    assert len(roots) == args.expected_episodes and roots, (len(roots), args.expected_episodes)
    infos = [json.loads((p / "meta/info.json").read_text()) for p in roots]
    original_metadata_codec = infos[0]["features"]["observation.images.front"]["info"]["video.codec"]
    tasks = []
    tables, episode_tables, stats, mapping, fingerprints = [], [], [], [], set()
    offset = 0
    previous = previous_episodes = comparison = None
    if PRIOR is not None:
        previous = pq.read_table(PRIOR / "data/chunk-000/file-000.parquet")
        previous_episodes = pq.read_table(PRIOR / "meta/episodes/chunk-000/file-000.parquet")
        assert len(previous_episodes) == len(roots), "Comparison episode counts must match"
        comparison = {"exact_action_state_matches": 0, "episodes_compared": len(roots)}
    video_files = {c: [] for c in ["front", "side"]}
    codec_summary = {}
    for i, (root, info) in enumerate(zip(roots, infos, strict=True)):
        assert info["total_episodes"] == 1 and info["fps"] == 30
        assert info["features"] == infos[0]["features"]
        table = pq.read_table(root / "data/chunk-000/file-000.parquet")
        ep = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()[0]
        n = len(table)
        assert n == info["total_frames"] == ep["length"]
        for key in ["episode_index", "task_index"]:
            assert np.all(table[key].to_numpy() == 0)
        for key in ["frame_index", "index"]:
            assert np.array_equal(table[key].to_numpy(), np.arange(n))
        assert np.allclose(table["timestamp"].to_numpy(), np.arange(n) / 30, atol=1e-4)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        assert action.shape == state.shape == (n, 6)
        assert np.isfinite(action).all() and np.isfinite(state).all()
        fingerprint = hashlib.sha256(action.tobytes() + state.tobytes()).hexdigest()
        assert fingerprint not in fingerprints, root
        fingerprints.add(fingerprint)
        if previous is not None:
            start = previous_episodes["dataset_from_index"][i].as_py()
            end = previous_episodes["dataset_to_index"][i].as_py()
            pa_ = np.asarray(previous["action"].slice(start, end - start).to_pylist(), dtype=np.float32)
            ps_ = np.asarray(
                previous["observation.state"].slice(start, end - start).to_pylist(), dtype=np.float32
            )
            comparison["exact_action_state_matches"] += int(
                np.array_equal(action, pa_) and np.array_equal(state, ps_)
            )
        source_task = pq.read_table(root / "meta/tasks.parquet")["task"][0].as_py()
        if source_task not in tasks:
            tasks.append(source_task)
        task_id = tasks.index(source_task)
        for key, values in [
            ("index", np.arange(offset, offset + n)),
            ("episode_index", np.full(n, i)),
            ("task_index", np.full(n, task_id)),
        ]:
            pos = table.schema.get_field_index(key)
            table = table.set_column(pos, table.schema.field(pos), pa.array(values, type=pa.int64()))
        tables.append(table)
        ep.update({"episode_index": i, "dataset_from_index": offset, "dataset_to_index": offset + n})
        for key in [
            "data/chunk_index",
            "data/file_index",
            "meta/episodes/chunk_index",
            "meta/episodes/file_index",
        ]:
            ep[key] = 0
        for camera in video_files:
            key = "observation.images." + camera
            path = root / f"videos/{key}/chunk-000/file-000.mp4"
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                assert stream.frames == n and stream.average_rate == 30
                assert (stream.width, stream.height) == (640, 480)
                codec = stream.codec_context.name
                assert codec == "h264", (path, codec)
                codec_summary[camera] = codec
            video_files[camera].append(path)
            prefix = "videos/" + key + "/"
            ep[prefix + "chunk_index"] = ep[prefix + "file_index"] = 0
            ep[prefix + "from_timestamp"] = offset / 30
            ep[prefix + "to_timestamp"] = (offset + n) / 30
        episode_tables.append(ep)
        stats.append(load_stats(root))
        mapping.append(
            {
                "merged_episode": i,
                "source_episode_dir": str(root),
                "task_index": task_id,
                "frames": n,
                "action_state_sha256": fingerprint,
            }
        )
        offset += n
    TARGET.mkdir(parents=True)
    (TARGET / "data/chunk-000").mkdir(parents=True)
    pq.write_table(pa.concat_tables(tables), TARGET / "data/chunk-000/file-000.parquet")
    (TARGET / "meta/episodes/chunk-000").mkdir(parents=True)
    source_schema = pq.read_table(roots[0] / "meta/episodes/chunk-000/file-000.parquet").schema
    pq.write_table(
        pa.Table.from_pylist(episode_tables, schema=source_schema),
        TARGET / "meta/episodes/chunk-000/file-000.parquet",
    )
    pd.DataFrame({"task_index": range(len(tasks))}, index=pd.Index(tasks, name="task")).to_parquet(
        TARGET / "meta/tasks.parquet"
    )
    info = infos[0]
    info.update(
        {
            "total_episodes": len(roots),
            "total_frames": offset,
            "total_tasks": len(tasks),
            "splits": {"train": f"0:{len(roots)}"},
            "video_files_size_in_mb": 1024,
        }
    )
    for camera in video_files:
        info["features"]["observation.images." + camera]["info"]["video.codec"] = codec_summary[camera]
    (TARGET / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    write_stats(aggregate_stats(stats), TARGET)
    repair(TARGET)
    for camera, paths in video_files.items():
        out = TARGET / f"videos/observation.images.{camera}/chunk-000/file-000.mp4"
        concatenate_video_files(paths, out, compatibility_check=True)
        with av.open(str(out)) as container:
            stream = container.streams.video[0]
            assert stream.frames == offset, (camera, stream.frames, offset)
        print("MERGED_VIDEO", camera, offset, "frames", flush=True)
    closing_summary = None
    report_paths = [SOURCE / f"{p.name}_report.csv" for p in roots]
    if all(p.is_file() for p in report_paths):
        reports = []
        for path in report_paths:
            with path.open() as handle:
                reports.append(next(csv.DictReader(handle)))
        values = [float(r["closing_percent_actual"]) for r in reports]
        closing_summary = [min(values), float(np.mean(values)), max(values)]
    report = {
        "source_root": str(SOURCE),
        "merged_root": str(TARGET),
        "episodes": len(roots),
        "frames": offset,
        "source_episode_mapping": mapping,
        "task_labels": tasks,
        "all_low_dim_values_preserved": True,
        "video_merge": "H264 packet stream copy, no decoding/re-encoding or temporal edits",
        "original_metadata_codec": original_metadata_codec,
        "actual_metadata_codec": "h264",
        "prior_histbalance_comparison": comparison,
        "closing_percent_actual_min_mean_max": closing_summary,
    }
    (TARGET / "MERGE_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "source_episode_mapping"}, indent=2))


if __name__ == "__main__":
    main()
