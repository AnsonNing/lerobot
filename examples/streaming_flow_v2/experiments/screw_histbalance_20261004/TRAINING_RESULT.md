# Histogram-balanced screw Streaming Flow V2 training

Both variants finished 20,000 optimizer steps. Checkpoints were selected by
the lowest full held-out EMA flow loss within each prediction horizon.

| Variant (predicted/executed) | Selected step | Held-out flow loss | First-action MAE per normalized joint | Gripper-close MAE (normalized) |
| --- | ---: | ---: | ---: | ---: |
| 8_1 | 20,000 | 0.076607 | 0.039971 | 0.153474 |
| 16_1 | 15,000 | 0.164411 | 0.021363 | 0.111572 |

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

The current host's saved preprocessor → policy → postprocessor pipeline
takes approximately 19.4 ms median for 8→1 and
35.9 ms for 16→1. Camera capture and motor communication are excluded.
Thus 16→1 already exceeds the 33.3 ms budget for synchronous 30 Hz replanning;
8→1 leaves about 13.9 ms for hardware I/O. Measure the complete loop on the robot host.


Training outputs and logs: `/home/ningan/training_outputs/screw_histbalance_20261004/`.
Training launcher: `/home/ningan/lerobot/scripts/train_screw_histbalance_streaming_flow_v2.sh`.
Dataset: `/mnt/data/ningan/screw_datasets/histogram_balance_cut_pull_screw_all_hover_gripper_25_assemble_episode`.
All delivery files copied from the selected checkpoints were verified by SHA256;
`CHECKSUMS.json` records the hashes. The delivered copies preserve the training
output checkpoints.
