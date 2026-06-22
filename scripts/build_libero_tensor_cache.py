#!/usr/bin/env python

"""Download LIBERO data and build LeRobot image tensor cache shards.

Default arguments match src/lerobot/policies/dispo/libero_spatial.txt.
The cache layout is the one consumed by lerobot.datasets.tensor_cache.ImageTensorCache:

    tensor_cache/
      manifest.json
      observation.images.image/shard-000000.pt
      observation.images.image2/shard-000000.pt
      ...
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections.abc import Iterable
from pathlib import Path

os.environ.setdefault("HF_HOME", "/home/ningan/hf_cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/home/ningan/hf_cache/hub")
os.environ.setdefault("HF_HUB_CACHE", "/home/ningan/hf_cache/hub")
os.environ.setdefault("HF_DATASETS_CACHE", "/home/ningan/hf_cache/datasets")
os.environ.setdefault("HF_LEROBOT_HOME", "/home/ningan/lerobot_data")
os.environ.setdefault("TMPDIR", "/home/ningan/tmp")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import torch.nn.functional as F
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.tensor_cache import MANIFEST_NAME

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - convenience fallback for minimal envs

    def tqdm(iterable: Iterable, **_: object) -> Iterable:
        return iterable


def parse_episodes(value: str) -> list[int] | None:
    value = value.strip()
    if value.lower() in {"all", "none", ""}:
        return None

    episodes: list[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            parts = chunk.split(":")
            if len(parts) not in {2, 3}:
                raise ValueError(f"Invalid episode range: {chunk!r}")
            start = int(parts[0])
            stop = int(parts[1])
            step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
            episodes.extend(range(start, stop, step))
        else:
            episodes.append(int(chunk))
    return episodes


def tensor_to_uint8_chw(
    tensor: torch.Tensor,
    resize_shape: tuple[int, int] | None = None,
) -> torch.Tensor:
    tensor = tensor.detach().cpu()
    if tensor.ndim != 3:
        raise ValueError(f"Expected image tensor shape (C,H,W), got {tuple(tensor.shape)}")
    if resize_shape is not None and tuple(tensor.shape[-2:]) != resize_shape:
        if tensor.dtype == torch.uint8:
            tensor = tensor.float() / 255.0
        elif not torch.is_floating_point(tensor):
            tensor = tensor.float()
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=resize_shape,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    if tensor.dtype == torch.uint8:
        return tensor.contiguous()
    if not torch.is_floating_point(tensor):
        return tensor.to(torch.uint8).contiguous()
    return (tensor.clamp(0, 1) * 255).round().to(torch.uint8).contiguous()


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp_path.replace(path)


def parquet_episode_range(path: Path | str) -> tuple[int, int]:
    table = pq.read_table(path, columns=["episode_index"])
    episodes = table.column("episode_index").to_numpy()
    return int(episodes.min()), int(episodes.max())


def _file_sort_key(path: str) -> tuple[int, int, str]:
    parts = Path(path).parts
    chunk = 0
    file_idx = 0
    for part in parts:
        if part.startswith("chunk-"):
            chunk = int(part.removeprefix("chunk-"))
        elif part.startswith("file-") and part.endswith(".parquet"):
            file_idx = int(part.removeprefix("file-").removesuffix(".parquet"))
    return chunk, file_idx, path


def predownload_episode_data_files(
    repo_id: str,
    revision: str,
    root: Path,
    episodes: list[int] | None,
) -> None:
    """Download data parquet files by their actual embedded episode_index ranges.

    HuggingFaceVLA/libero v3.0 has metadata entries whose data/file_index fields do not
    match the parquet files currently hosted on the Hub. LeRobotDataset's normal
    selective download follows those metadata entries, then filters by parquet
    episode_index and can end up with no rows. This resolver treats the parquet
    episode_index column as the source of truth.
    """
    if episodes is None:
        return
    if not episodes:
        raise ValueError("Episode list is empty.")

    min_ep = min(episodes)
    max_ep = max(episodes)
    api = HfApi()
    data_files = sorted(
        (
            f
            for f in api.list_repo_files(repo_id, repo_type="dataset", revision=revision)
            if f.startswith("data/") and f.endswith(".parquet")
        ),
        key=_file_sort_key,
    )
    if not data_files:
        raise FileNotFoundError(f"No data parquet files found in {repo_id}@{revision}")

    range_cache_path = root / ".cache" / "data_file_episode_ranges.json"
    range_cache = load_json(range_cache_path) or {}

    def get_range(file_idx: int) -> tuple[int, int]:
        rel_path = data_files[file_idx]
        cached = range_cache.get(rel_path)
        if cached is not None:
            return int(cached[0]), int(cached[1])

        local_path = root / rel_path
        if local_path.exists():
            ep_range = parquet_episode_range(local_path)
        else:
            cached_path = hf_hub_download(
                repo_id,
                repo_type="dataset",
                revision=revision,
                filename=rel_path,
                cache_dir=os.environ.get("HUGGINGFACE_HUB_CACHE"),
            )
            ep_range = parquet_episode_range(cached_path)
        range_cache[rel_path] = list(ep_range)
        save_json(range_cache_path, range_cache)
        return ep_range

    # Parquet files are ordered by episode_index on the Hub. Binary search the
    # first/last files intersecting the requested episode interval.
    lo, hi = 0, len(data_files) - 1
    first = len(data_files)
    while lo <= hi:
        mid = (lo + hi) // 2
        _, file_max = get_range(mid)
        if file_max >= min_ep:
            first = mid
            hi = mid - 1
        else:
            lo = mid + 1

    lo, hi = 0, len(data_files) - 1
    last = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        file_min, _ = get_range(mid)
        if file_min <= max_ep:
            last = mid
            lo = mid + 1
        else:
            hi = mid - 1

    if first > last:
        raise ValueError(f"No remote data files contain episode range {min_ep}:{max_ep + 1}.")

    selected = data_files[first : last + 1]
    print(
        f"Resolved episode range {min_ep}:{max_ep + 1} to {len(selected)} data file(s): "
        f"{selected[0]} .. {selected[-1]}"
    )
    snapshot_download(
        repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=root,
        allow_patterns=["meta/", *selected],
    )


def existing_cache_complete(
    cache_dir: Path,
    camera_keys: list[str],
    frame_indices: torch.Tensor,
    shard_size: int,
) -> bool:
    manifest = load_json(cache_dir / MANIFEST_NAME)
    if manifest is None:
        return False
    if set(manifest.get("camera_keys", [])) < set(camera_keys):
        return False
    if int(manifest.get("shard_size", shard_size)) != shard_size:
        raise ValueError(
            f"Existing tensor cache uses shard_size={manifest.get('shard_size')}; "
            f"rerun with --shard-size={manifest.get('shard_size')} or --overwrite."
        )

    frame_indices = frame_indices.to(torch.long)
    unique_shards = torch.unique(torch.div(frame_indices, shard_size, rounding_mode="floor")).tolist()
    for camera_key in camera_keys:
        for shard_idx in unique_shards:
            shard_indices = frame_indices[frame_indices // shard_size == shard_idx]
            max_offset = int((shard_indices % shard_size).max().item())
            shard_path = cache_dir / camera_key / f"shard-{int(shard_idx):06d}.pt"
            if not shard_path.exists():
                return False
            shard = torch.load(shard_path, map_location="cpu")
            if not isinstance(shard, torch.Tensor):
                return False
            if shard.dtype != torch.uint8 or shard.ndim != 4 or shard.shape[0] <= max_offset:
                return False
    return True


class TensorCacheShardWriter:
    def __init__(
        self,
        cache_dir: Path,
        camera_keys: list[str],
        shard_size: int,
        overwrite: bool,
    ) -> None:
        self.cache_dir = cache_dir
        self.camera_keys = camera_keys
        self.shard_size = shard_size
        self.overwrite = overwrite
        self.current_shard_idx: int | None = None
        self.buffers: dict[str, torch.Tensor] = {}
        self.max_offsets: dict[str, int] = {}
        self.initial_lengths: dict[str, int] = {}

    def add(self, frame_index: int, frames: dict[str, torch.Tensor]) -> None:
        shard_idx = frame_index // self.shard_size
        offset = frame_index % self.shard_size
        if self.current_shard_idx is None:
            self._start_shard(shard_idx, frames)
        elif shard_idx != self.current_shard_idx:
            self.flush()
            self._start_shard(shard_idx, frames)

        for camera_key, frame in frames.items():
            self.buffers[camera_key][offset] = frame
            self.max_offsets[camera_key] = max(self.max_offsets[camera_key], offset)

    def flush(self) -> None:
        if self.current_shard_idx is None:
            return
        for camera_key, buffer in self.buffers.items():
            max_len = max(self.initial_lengths[camera_key], self.max_offsets[camera_key] + 1)
            out_dir = self.cache_dir / camera_key
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"shard-{self.current_shard_idx:06d}.pt"
            tmp_path = out_path.with_suffix(".pt.tmp")
            torch.save(buffer[:max_len].contiguous(), tmp_path)
            tmp_path.replace(out_path)
        self.current_shard_idx = None
        self.buffers = {}
        self.max_offsets = {}
        self.initial_lengths = {}

    def _start_shard(self, shard_idx: int, sample_frames: dict[str, torch.Tensor]) -> None:
        self.current_shard_idx = shard_idx
        for camera_key in self.camera_keys:
            frame = sample_frames[camera_key]
            existing = self._load_existing(camera_key, shard_idx)
            if existing is None:
                shape = (self.shard_size, *frame.shape)
                existing = torch.zeros(shape, dtype=torch.uint8)
                initial_len = 0
            else:
                initial_len = existing.shape[0]
                if existing.shape[0] < self.shard_size:
                    padded = torch.zeros((self.shard_size, *existing.shape[1:]), dtype=torch.uint8)
                    padded[: existing.shape[0]] = existing
                    existing = padded
            if tuple(existing.shape[1:]) != tuple(frame.shape):
                raise ValueError(
                    f"Existing shard shape for {camera_key} is {tuple(existing.shape[1:])}, "
                    f"but decoded frame shape is {tuple(frame.shape)}."
                )
            self.buffers[camera_key] = existing
            self.initial_lengths[camera_key] = initial_len
            self.max_offsets[camera_key] = -1

    def _load_existing(self, camera_key: str, shard_idx: int) -> torch.Tensor | None:
        if self.overwrite:
            return None
        path = self.cache_dir / camera_key / f"shard-{shard_idx:06d}.pt"
        if not path.exists():
            return None
        shard = torch.load(path, map_location="cpu")
        if not isinstance(shard, torch.Tensor) or shard.dtype != torch.uint8 or shard.ndim != 4:
            raise ValueError(f"Invalid existing tensor cache shard: {path}")
        return shard.contiguous()


def build_manifest(
    cache_dir: Path,
    repo_id: str,
    root: Path,
    camera_keys: list[str],
    frame_indices: torch.Tensor,
    shard_size: int,
    episodes: list[int] | None,
    resize_shape: tuple[int, int] | None,
) -> None:
    indices = frame_indices.to(torch.long)
    manifest = {
        "format": "torch_shards",
        "dtype": "uint8",
        "camera_keys": camera_keys,
        "shard_size": shard_size,
        "index_start": int(indices.min().item()),
        "num_frames": int(indices.max().item() - indices.min().item() + 1),
        "repo_id": repo_id,
        "dataset_root": str(root),
        "episodes": episodes,
        "resize_shape": list(resize_shape) if resize_shape is not None else None,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    tmp_path = cache_dir / f"{MANIFEST_NAME}.tmp"
    with tmp_path.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp_path.replace(cache_dir / MANIFEST_NAME)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="HuggingFaceVLA/libero")
    parser.add_argument("--revision", default="v3.0")
    parser.add_argument("--root", type=Path, default=Path("/home/ningan/lerobot_data/HuggingFaceVLA/libero"))
    parser.add_argument(
        "--tensor-cache-dir",
        type=Path,
        default=Path("/home/ningan/lerobot_data/HuggingFaceVLA/libero/tensor_cache"),
    )
    parser.add_argument(
        "--episodes",
        default="1261:1693",
        help="Comma list or Python-style ranges. Default 1261:1693 matches LIBERO spatial.",
    )
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument(
        "--resize-shape",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help="Optionally resize cached CHW image tensors before writing, e.g. --resize-shape 96 96.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete the existing tensor cache first.")
    parser.add_argument("--download-only", action="store_true", help="Only download/load the dataset.")
    parser.add_argument("--force-cache-sync", action="store_true", help="Ask LeRobotDataset to refresh files.")
    parser.add_argument(
        "--skip-data-file-resolution",
        action="store_true",
        help="Use LeRobotDataset's default selective download path without resolving parquet episode ranges.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes = parse_episodes(args.episodes)

    for path in [
        Path(os.environ["HF_HOME"]),
        Path(os.environ["HUGGINGFACE_HUB_CACHE"]),
        Path(os.environ["HF_DATASETS_CACHE"]),
        Path(os.environ["HF_LEROBOT_HOME"]),
        Path(os.environ["TMPDIR"]),
        args.root,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    if args.overwrite and args.tensor_cache_dir.exists():
        shutil.rmtree(args.tensor_cache_dir)

    if not args.skip_data_file_resolution:
        predownload_episode_data_files(args.repo_id, args.revision, args.root, episodes)

    disabled_cache_dir = args.tensor_cache_dir.parent / ".tensor_cache_disabled_for_build"
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.root,
        episodes=episodes,
        revision=args.revision,
        image_transforms=None,
        force_cache_sync=args.force_cache_sync,
        download_videos=True,
        video_backend=args.video_backend,
        return_uint8=True,
        tensor_cache_dir=disabled_cache_dir,
    )

    frame_indices = torch.as_tensor(dataset.hf_dataset.data.column("index").combine_chunks().to_numpy())
    if frame_indices.numel() == 0:
        raise ValueError("Selected episodes contain no frames.")
    if frame_indices.numel() > 1 and not torch.all(frame_indices[1:] >= frame_indices[:-1]):
        raise ValueError("Dataset frame indices are not sorted; this shard writer expects sequential reads.")
    camera_keys = list(dataset.meta.camera_keys)
    if not camera_keys:
        raise ValueError("Dataset has no camera/image keys to cache.")

    print(f"Dataset root: {dataset.root}")
    print(f"Selected episodes: {dataset.num_episodes}")
    print(f"Selected frames: {len(dataset)}")
    print(f"Camera keys: {camera_keys}")
    print(f"Tensor cache: {args.tensor_cache_dir}")
    if args.resize_shape is not None:
        print(f"Cached image resize shape: {tuple(args.resize_shape)}")

    if args.download_only:
        print("Download-only mode complete.")
        return

    args.tensor_cache_dir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite and existing_cache_complete(
        args.tensor_cache_dir, camera_keys, frame_indices, args.shard_size
    ):
        print("Tensor cache already covers the selected episodes.")
        return

    writer = TensorCacheShardWriter(
        cache_dir=args.tensor_cache_dir,
        camera_keys=camera_keys,
        shard_size=args.shard_size,
        overwrite=args.overwrite,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    for batch in tqdm(loader, total=len(loader), desc="Building tensor cache"):
        batch_indices = batch["index"].to(torch.long).tolist()
        for row, frame_index in enumerate(batch_indices):
            frames = {
                camera_key: tensor_to_uint8_chw(
                    batch[camera_key][row],
                    resize_shape=tuple(args.resize_shape) if args.resize_shape is not None else None,
                )
                for camera_key in camera_keys
            }
            writer.add(int(frame_index), frames)
    writer.flush()

    build_manifest(
        cache_dir=args.tensor_cache_dir,
        repo_id=args.repo_id,
        root=dataset.root,
        camera_keys=camera_keys,
        frame_indices=frame_indices,
        shard_size=args.shard_size,
        episodes=episodes,
        resize_shape=tuple(args.resize_shape) if args.resize_shape is not None else None,
    )

    if not existing_cache_complete(args.tensor_cache_dir, camera_keys, frame_indices, args.shard_size):
        raise RuntimeError("Tensor cache verification failed after writing.")
    print("Tensor cache build complete.")


if __name__ == "__main__":
    main()
