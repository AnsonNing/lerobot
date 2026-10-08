#!/usr/bin/env python3
"""Compare 15k and 20k training-loss windows for the Pull Screw baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


RUN_NAMES = {
    "diffusion": "diffusion_8_1",
    "streaming_flow": "streaming_flow_8_1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", nargs="?", choices=("all", *RUN_NAMES), default="all")
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "outputs" / "screw_histbalance_baselines",
    )
    parser.add_argument("--steps", type=int, nargs="+", default=(15000, 20000))
    parser.add_argument("--window-steps", type=int, default=1000)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def load_records(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "step" not in record or "loss" not in record:
                raise ValueError(f"{path}:{line_number} lacks step/loss")
            records[int(record["step"])] = record
    if not records:
        raise ValueError(f"No training metrics found in {path}")
    return records


def summarize(run_dir: Path, steps: list[int], window_steps: int) -> dict:
    metrics_path = run_dir / "train_metrics.jsonl"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Training metrics not found: {metrics_path}")
    records = load_records(metrics_path)
    summaries = []
    for step in steps:
        checkpoint = run_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        if step not in records:
            raise ValueError(f"No logged loss at step {step} in {metrics_path}")
        window = [
            record
            for record_step, record in sorted(records.items())
            if step - window_steps < record_step <= step
        ]
        if not window:
            raise ValueError(f"No loss records in trailing window for step {step}")
        losses = [float(record["loss"]) for record in window]
        checkpoint_report = run_dir / f"all301_checkpoint_loss_{step:06d}.json"
        checkpoint_loss = None
        if checkpoint_report.is_file():
            checkpoint_loss = float(json.loads(checkpoint_report.read_text(encoding="utf-8"))["loss"])
        summaries.append(
            {
                "step": step,
                "checkpoint": str(checkpoint),
                "logged_loss": float(records[step]["loss"]),
                "trailing_window_steps": window_steps,
                "trailing_records": len(window),
                "trailing_mean_loss": sum(losses) / len(losses),
                "trailing_min_loss": min(losses),
                "trailing_max_loss": max(losses),
                "all301_checkpoint_loss": checkpoint_loss,
            }
        )
    has_checkpoint_losses = all(item["all301_checkpoint_loss"] is not None for item in summaries)
    selection_metric = "all301_checkpoint_loss" if has_checkpoint_losses else "trailing_mean_loss"
    selected = min(summaries, key=lambda item: item[selection_metric])
    return {
        "run_dir": str(run_dir),
        "metric": "training loss (not held-out validation)",
        "checkpoints": summaries,
        "selection_metric": selection_metric,
        "suggested_step": selected["step"],
    }


def main() -> None:
    args = parse_args()
    if args.window_steps <= 0 or any(step <= 0 for step in args.steps):
        raise ValueError("steps and window-steps must be positive")
    variants = list(RUN_NAMES) if args.variant == "all" else [args.variant]
    output = {
        "warning": (
            "All 301 episodes were used for optimization. Compare steps within each policy only; "
            "these losses are not held-out validation or real-robot success rates."
        ),
        "models": {
            variant: summarize(args.output_base / RUN_NAMES[variant], args.steps, args.window_steps)
            for variant in variants
        },
    }

    print("variant         step   logged_loss   trailing_mean   all301_checkpoint")
    for variant, result in output["models"].items():
        for checkpoint in result["checkpoints"]:
            all301_loss = checkpoint["all301_checkpoint_loss"]
            all301_display = all301_loss if all301_loss is not None else "not-run"
            print(
                f"{variant:15} {checkpoint['step']:5d}   "
                f"{checkpoint['logged_loss']:.8f}   {checkpoint['trailing_mean_loss']:.8f}   "
                f"{all301_display}"
            )
        print(f"  suggested by {result['selection_metric']}: {result['suggested_step']}")

    destination = args.json_output or args.output_base / "loss_comparison.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote: {destination}")


if __name__ == "__main__":
    main()
