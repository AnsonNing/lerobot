#!/usr/bin/env python

"""Small on-disk uint8 tensor cache for visual LeRobot features."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

DEFAULT_TENSOR_CACHE_DIRNAME = "tensor_cache"
MANIFEST_NAME = "manifest.json"


class ImageTensorCache:
    def __init__(self, root: Path | str, max_open_shards: int = 2):
        self.root = Path(root)
        manifest_path = self.root / MANIFEST_NAME
        with manifest_path.open() as f:
            self.manifest = json.load(f)
        self.camera_keys = list(self.manifest["camera_keys"])
        self.format = self.manifest.get("format", "torch_shards")
        self.shard_size = int(self.manifest.get("shard_size", 512))
        self.index_start = int(self.manifest.get("index_start", 0))
        self.num_frames = int(self.manifest.get("num_frames", 0))
        self.dtype = self.manifest.get("dtype", "uint8")
        if self.dtype != "uint8":
            raise ValueError(f"Unsupported tensor cache dtype: {self.dtype}")
        self._max_open_shards = max_open_shards
        self._shards: OrderedDict[tuple[str, int], torch.Tensor] = OrderedDict()
        self._memmaps: dict[str, np.memmap] = {}

    @classmethod
    def discover(cls, dataset_root: Path | str, cache_dir: Path | str | None = None) -> ImageTensorCache | None:
        root = Path(cache_dir) if cache_dir is not None else Path(dataset_root) / DEFAULT_TENSOR_CACHE_DIRNAME
        if (root / MANIFEST_NAME).exists():
            return cls(root)
        return None

    def get(self, camera_key: str, frame_index: int) -> torch.Tensor:
        if camera_key not in self.camera_keys:
            raise KeyError(f"Camera key {camera_key!r} is not in tensor cache")
        if self.format == "memmap":
            offset = frame_index - self.index_start
            if offset < 0 or offset >= self.num_frames:
                raise IndexError(f"Frame {frame_index} is outside tensor cache range")
            return torch.from_numpy(np.asarray(self._load_memmap(camera_key)[offset]))

        shard_index = frame_index // self.shard_size
        offset = frame_index % self.shard_size
        shard = self._load_shard(camera_key, shard_index)
        if offset >= shard.shape[0]:
            raise IndexError(f"Frame {frame_index} is outside shard {shard_index} for {camera_key}")
        return shard[offset]

    def get_many(self, camera_key: str, frame_indices: list[int]) -> torch.Tensor:
        if self.format == "memmap":
            offsets = np.asarray(frame_indices, dtype=np.int64) - self.index_start
            if (offsets < 0).any() or (offsets >= self.num_frames).any():
                raise IndexError(f"Requested frames outside tensor cache range for {camera_key}")
            return torch.from_numpy(np.asarray(self._load_memmap(camera_key)[offsets]))
        return torch.stack([self.get(camera_key, int(idx)) for idx in frame_indices])

    def _load_memmap(self, camera_key: str) -> np.memmap:
        if camera_key not in self._memmaps:
            self._memmaps[camera_key] = np.load(self.root / f"{camera_key}.npy", mmap_mode="r")
        return self._memmaps[camera_key]

    def _load_shard(self, camera_key: str, shard_index: int) -> torch.Tensor:
        key = (camera_key, shard_index)
        if key in self._shards:
            self._shards.move_to_end(key)
            return self._shards[key]

        path = self.root / camera_key / f"shard-{shard_index:06d}.pt"
        shard = torch.load(path, map_location="cpu")
        if not isinstance(shard, torch.Tensor):
            raise TypeError(f"Expected tensor shard at {path}, got {type(shard)}")
        self._shards[key] = shard
        self._shards.move_to_end(key)
        while len(self._shards) > self._max_open_shards:
            self._shards.popitem(last=False)
        return shard
