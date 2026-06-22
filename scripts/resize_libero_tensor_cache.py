#!/usr/bin/env python

"""Resize an existing LeRobot uint8 image tensor cache into a smaller cache."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

MANIFEST_NAME = "manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/home/ningan/lerobot_data/HuggingFaceVLA/libero/tensor_cache"),
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("/home/ningan/lerobot_data/HuggingFaceVLA/libero/tensor_cache_96"),
    )
    parser.add_argument("--resize-shape", type=int, nargs=2, default=(96, 96), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resize_uint8_nchw(
    shard: torch.Tensor,
    resize_shape: tuple[int, int],
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    if shard.dtype != torch.uint8 or shard.ndim != 4:
        raise ValueError(f"Expected uint8 NCHW shard, got dtype={shard.dtype}, shape={tuple(shard.shape)}")
    if tuple(shard.shape[-2:]) == resize_shape:
        return shard.contiguous()

    resized_chunks = []
    for chunk in shard.split(chunk_size, dim=0):
        chunk = chunk.to(device=device, dtype=torch.float32, non_blocking=True) / 255.0
        chunk = F.interpolate(chunk, size=resize_shape, mode="bilinear", align_corners=False)
        chunk = (chunk.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu()
        resized_chunks.append(chunk)
    return torch.cat(resized_chunks, dim=0).contiguous()


def main() -> None:
    args = parse_args()
    source_manifest_path = args.source / MANIFEST_NAME
    if not source_manifest_path.exists():
        raise FileNotFoundError(f"Missing source manifest: {source_manifest_path}")
    if args.dest.exists() and any(args.dest.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Destination is not empty: {args.dest}. Pass --overwrite to replace shards.")

    with source_manifest_path.open() as f:
        manifest = json.load(f)
    if manifest.get("dtype") != "uint8":
        raise ValueError(f"Only uint8 tensor caches are supported, got {manifest.get('dtype')!r}")

    resize_shape = tuple(args.resize_shape)
    device = torch.device(args.device)
    args.dest.mkdir(parents=True, exist_ok=True)

    for camera_key in manifest["camera_keys"]:
        source_dir = args.source / camera_key
        dest_dir = args.dest / camera_key
        dest_dir.mkdir(parents=True, exist_ok=True)
        shard_paths = sorted(source_dir.glob("shard-*.pt"))
        if not shard_paths:
            raise FileNotFoundError(f"No shards found for {camera_key}: {source_dir}")

        print(f"{camera_key}: resizing {len(shard_paths)} shard(s) to {resize_shape}")
        for path in shard_paths:
            dest_path = dest_dir / path.name
            if dest_path.exists() and not args.overwrite:
                continue
            shard = torch.load(path, map_location="cpu")
            resized = resize_uint8_nchw(shard, resize_shape, device=device, chunk_size=args.chunk_size)
            torch.save(resized, dest_path)

    manifest = dict(manifest)
    manifest["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["source_tensor_cache"] = str(args.source)
    manifest["resize_shape"] = list(resize_shape)
    tmp_manifest_path = args.dest / f"{MANIFEST_NAME}.tmp"
    with tmp_manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp_manifest_path.replace(args.dest / MANIFEST_NAME)
    print(f"Resized tensor cache complete: {args.dest}")


if __name__ == "__main__":
    main()
