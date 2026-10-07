# Handoff verification — 2026-10-07

Validated in the environment recorded in `environment-observed-20261007.json`.

- `pytest -q tests/policies/streaming_flow tests/utils/test_sample_weighting.py tests/scripts/test_streaming_flow_v2_handoff.py tests/envs/test_mimicgen.py`: **81 passed**.
- Ruff 0.14.1 check and format checks passed on all 16 Python files included in this handoff.
- Shell syntax checks passed on all seven included shell scripts; Python compilation and local documentation links passed.
- Launcher dry runs for 8/1 and 16/1 parsed with the actual `TrainPipelineConfig`, including normal/smoke settings and paths containing spaces outside the checkout. An overlapping train/validation split was rejected.
- All 68 historical snapshots matched the source SHA256 recorded in `experiments/PROVENANCE.json`.
- Merge/preparation integration used copies of two actual source episodes, 272 frames, in a temporary directory. Merge worked without the previous dataset and with the optional comparison dataset; both action/state fingerprints matched the comparison. Source action arrays remained identical. Front/side images decoded at both episode boundaries. Preparation produced separate train/validation episodes and train-only stats for 128 frames. Temporary datasets were removed after the check.
- Both delivered gripper25 models reloaded with the published V2 source and saved processors, producing finite actions of shape `(1,6)` from dataset observations. Each used five warmups and two measured calls on CUDA. These concurrent short checks establish reload/interface compatibility; their timings are not deployment benchmarks.

No new long training, clean-machine installation, simulator task evaluation or physical robot rollout was performed for this handoff. Historical losses and latency measurements remain under each experiment directory.
