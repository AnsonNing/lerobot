#!/usr/bin/env python3
"""Refresh statistics for indices remapped during a single-episode merge."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def describe(a):
    a = np.asarray(a, dtype=np.float64).reshape(-1, 1)
    return {
        "min": a.min(axis=0).tolist(),
        "max": a.max(axis=0).tolist(),
        "mean": a.mean(axis=0).tolist(),
        "std": a.std(axis=0).tolist(),
        "count": [len(a)],
        **{f"q{int(q * 100):02}": np.quantile(a, q, axis=0).tolist() for q in [0.01, 0.1, 0.5, 0.9, 0.99]},
    }


def repair(root):
    root = Path(root)
    keys = ["index", "episode_index", "task_index"]
    data = pq.read_table(root / "data/chunk-000/file-000.parquet", columns=keys)
    ep_path = root / "meta/episodes/chunk-000/file-000.parquet"
    episodes = pq.read_table(ep_path)
    stats_path = root / "meta/stats.json"
    stats = json.loads(stats_path.read_text())
    policy_before = {
        k: stats[k]
        for k in ["action", "observation.state", "observation.images.front", "observation.images.side"]
    }
    columns = {k: data[k].to_numpy() for k in keys}
    previous = {k: stats[k] for k in keys}
    stats.update({k: describe(v) for k, v in columns.items()})
    rows = episodes.to_pylist()
    for row in rows:
        start, end = row["dataset_from_index"], row["dataset_to_index"]
        for k in keys:
            values = describe(columns[k][start:end])
            for statistic, value in values.items():
                name = f"stats/{k}/{statistic}"
                if name in row:
                    row[name] = value
    assert stats["index"]["max"] == [len(data) - 1]
    assert stats["episode_index"]["max"] == [len(episodes) - 1]
    assert all(stats[k] == v for k, v in policy_before.items())
    pq.write_table(pa.Table.from_pylist(rows, schema=episodes.schema), ep_path)
    stats_path.write_text(json.dumps(stats, indent=4) + "\n")
    result = {
        "corrected_keys": keys,
        "previous_global_stats": previous,
        "new_global_stats": {k: stats[k] for k in keys},
        "policy_normalization_stats_unchanged": True,
        "data_and_videos_unchanged": True,
        "frames": len(data),
        "episodes": len(episodes),
    }
    (root / "INDEX_STATISTICS_REPAIR.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
