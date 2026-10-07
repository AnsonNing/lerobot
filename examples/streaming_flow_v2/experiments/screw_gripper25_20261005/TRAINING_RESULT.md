# Screw gripper25 Streaming Flow V2 — 2026-10-05

Both policies completed 20,000 optimizer updates, with finite logged loss and gradients.
The delivered model for each horizon was selected by the lowest full held-out EMA flow
loss among steps 5,000, 10,000, 15,000 and 20,000. This is offline checkpoint selection.

| Predicted/executed | Selected step | Held-out flow loss | First-action MAE per normalized joint | Gripper-close MAE normalized |
| --- | ---: | ---: | ---: | ---: |
| 8_1 | 20,000 | 0.075691 | 0.039477 | 0.165951 |
| 16_1 | 15,000 | 0.165102 | 0.021225 | 0.121445 |

The ZIP contained 200 independent single-episode LeRobot v3 datasets,
with 21,845 frames at 30 FPS. They were merged in numeric episode
order without changing any action/state values or resampling the supplied sequences.
Front and side H264 videos were concatenated by packet stream copy without re-encoding.
The source metadata incorrectly declared AV1; the merged metadata declares H264.
The merged dataset reloads with LeRobot, and sampled episode boundary images decode.
Two task/source labels each contain 100 episodes. The two policies do not condition on task text.

The exact source-stratified split matches the preceding histogram-balanced experiment:
180 training episodes / 19,714 frames and
20 held-out episodes / 2,131 frames,
seed 100000. State/action normalization statistics are fitted on training episodes only.
All 200 episodes have identical action/state values to the previous histogram-balanced
dataset, but decoded image content differs in both cameras. Full decoded video frame
hash comparison and first-frame RGB pixel diagnostics are in PRIOR_DATASET_COMPARISON.json.
This experiment changes the image input while retaining action demonstrations and split.

The source preprocessing CSVs report a gripper closing phase averaging about 25% of each
sequence. A different diagnostic, normalized per-frame gripper decrease below -0.02,
counts 7.40% of transitions;
these measure phase duration and sufficiently large changes, respectively. No extra
sample weighting, smoothing, interpolation or temporal editing was added.

Both variants use two fully trainable independent ImageNet ResNet18 camera encoders,
2 observation frames, batch 16, seed 100000, AdamW lr 1e-4, cosine scheduling with
500 warmup steps, EMA, 4 flow training points, adaptive frequency and step scaling.
RGB is resized from 480x640 to 240x320 and randomly cropped to 228x304 for training;
inference uses the center crop and saved preprocessing. State/actions use per-dimension
MIN_MAX normalization. W&B is disabled; local metrics are retained. `chunk_size=9/17`
contains the previous action plus 8/16 new commands; both execute one before replanning.
Initial previous action uses the observed joint state.

The directories 8_1 and 16_1 contain native LeRobot model.safetensors, config.json,
train_config.json and complete pre/postprocessors. Load each complete directory with
the matching streaming_flow_v2 source. The runtime source snapshot, base Git commit,
compatibility patch and SHA256 manifest are included. Preserve camera viewpoints,
follower motor/gripper calibration, and the state/action order: shoulder_pan.pos,
shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos, gripper.pos.
Inputs are observation.images.front, observation.images.side and observation.state.
Reset the policy per episode and maintain previous-action alignment during replanning.

The delivered copies were reloaded for inference and produced finite 6D actions.
On this RTX5090 host, preprocessor -> policy -> postprocessor median latency was
19.4 ms for 8/1 and 35.4 ms for 16/1.
P95 was 20.3 / 35.5 ms.
Camera capture and motor communication are excluded. A synchronous 30Hz control loop
has a 33.3ms total budget; measure the entire loop on the deployment host.

Validation is teacher-forced offline evaluation, including demonstration previous actions.
No physical robot task success rate was measured. Flow loss scales differ between prediction
horizons, so their absolute values are not a direct cross-horizon ranking.

Dataset: /mnt/data/ningan/screw_datasets/cut_pull_screw_all_hover_gripper25_merged_20261005
Training outputs: /mnt/data/ningan/screw_datasets/outputs/streaming_flow_v2_screw_gripper25_20261005
Launcher: /home/ningan/lerobot/scripts/train_screw_gripper25_streaming_flow_v2.sh
Source archive: /home/ningan/cut_pull_screw_all_hover_gripper_25_assemble_episode.zip
Source archive SHA256: 455e75f5cc6c943aacfa96f4c2f237de1b6dc2cf69d92686684c87d4251d4ec5
Selected and final checkpoints and full optimizer state are retained in the training outputs.
Each copied model file was verified against its source SHA256. ZIP CRC and reload checks passed.
