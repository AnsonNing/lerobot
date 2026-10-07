# Push-button Streaming Flow V2

The merged LeRobot v3 dataset is `/mnt/data/ningan/button_datasets/push_button_merged_260914` (`HZL/push_button_merged_260914`): 224 episodes, 36,393 frames, 30 FPS, 6 named joint-position actions/states, and `front`/`side` RGB cameras at 480×640. Both source batches used the same motor/gripper calibration according to the collector. The recorded gripper command ranges still differ substantially (old 1.6355–2.2586, new 0.6494–1.1544), as do camera viewpoints and start poses. This is a distribution shift, not evidence by itself of a calibration mismatch. No success labels are present.

`meta/button_split_260914.json` contains a source-stratified, seeded split: 201 training episodes (165 old, 36 new), 22 held-out episodes (18 old, 4 new), and one excluded episode (165). Episode 165 is only 42 frames with almost no arm movement; sampled start/middle/end frames show no press. The source archives and merged raw episodes remain intact. The dataset metadata statistics include all 224 episodes, including held-out episodes; offline validation based on these normalization statistics has minor feature-statistics leakage. Neither model should be judged by a one-step training loss or offline loss alone; ultimately score real button-press success and safety on held-out starts.

Both recipes use two independent ImageNet-pretrained ResNet18 encoders, one per camera, with all visual weights trainable. They use joint-wise action min/max normalization (`per_dim`), two observation frames, aligned previous-action trajectory sampling, EMA, no W&B, the same train split and seed, and no simulator evaluation. The dataset already has LeRobot format; image processing happens in the model. Images are resized from 480×640 to 240×320 and randomly cropped to 228×304 during training; inference takes the same-size center crop. This preserves the 4:3 aspect ratio. In original-image coordinates, a random crop can remove up to 24 pixels vertically or 32 horizontally from one side; a center crop removes 12 and 16 from each side. Avoid a stronger 90% crop initially because keyboard keys and gripper enter near image borders. The model's ImageNet normalization is applied to RGB inputs in `[0,1]`; do not pre-normalize or manually resize videos on disk.

| Recipe | `chunk_size` | Future predictions | Executed before replan | Nominal open-loop duration at 30 FPS |
| --- | ---: | ---: | ---: | ---: |
| `16_8` | 17 | 16 | 8 | 267 ms |
| `8_1` | 9 | 8 | 1 | 33 ms |

In aligned mode `chunk_size` includes `a_(t-1)` as path state zero. Thus `chunk_size=17` yields 16 new actions, and `chunk_size=9` yields 8. `execution_horizon` controls how many returned actions are sent before replanning; the model continues from the last returned action, not the unexecuted chunk tail. A clipped or modified hardware action can break that internal continuity, so compare actual `send_action` output against policy commands in an instrumented first rollout if enabling `max_relative_target` or other robot action processors.

Run the short validation first, then start either long run deliberately:

```bash
cd /home/ningan/lerobot
SMOKE=1 CUDA_VISIBLE_DEVICES=0 bash scripts/train_push_button_streaming_flow_v2.sh 16_8
SMOKE=1 CUDA_VISIBLE_DEVICES=0 bash scripts/train_push_button_streaming_flow_v2.sh 8_1

CUDA_VISIBLE_DEVICES=0 bash scripts/train_push_button_streaming_flow_v2.sh 16_8
CUDA_VISIBLE_DEVICES=1 bash scripts/train_push_button_streaming_flow_v2.sh 8_1
```

The default full run is 50,000 optimizer steps, batch size 16, with checkpoints every 10,000 steps under `/mnt/data/ningan/button_datasets/outputs`. Override `STEPS`, `BATCH_SIZE`, `SAVE_FREQ`, `OUTPUT_DIR`, or `NUM_WORKERS` through environment variables. The two model runs are independent from scratch; do not resume one variant into the other. Checkpoint directories contain `pretrained_model`, including policy weights, config, and saved pre/post-processors. Use that whole directory for inference. Each one-step checkpoint used about 1.6 GB on disk; budget roughly 16 GB for ten checkpoints across two complete runs, plus logs and temporary write headroom.

The 2026-09-14 runs were stopped after the 40,000-step checkpoints because both source-stratified held-out losses were worse at 30,000 and 40,000 than at 20,000. The symlink `checkpoints/best_offline` points to `020000` for each variant; `checkpoints/last` points to `040000`. See the merged dataset's `TRAINING_RESULT.md` for measured losses and actual run state. `best_offline` is an offline candidate, not proof of hardware success.

After checkpoints are saved, compute held-out flow loss separately for each source batch:

```bash
PYTHONPATH=/home/ningan/lerobot/src /mnt/data/ningan/lerobot_mimicgen/venv/bin/python \
  scripts/eval_push_button_streaming_flow_v2.py \
  /mnt/data/ningan/button_datasets/outputs/streaming_flow_v2_button_16_8_crop95/checkpoints/best_offline/pretrained_model
```

Repeat with the `8_1` path. This scores the EMA policy with deterministic center cropping and the same held-out episodes. The flow losses are trajectory-length dependent, so compare a variant across checkpoints and sources; do not use their absolute values as a direct success comparison between `16_8` and `8_1`.

For real-hardware deployment, use this checkout's `lerobot-rollout` CLI, `--strategy.type=base --inference.type=sync --interpolation_multiplier=1`, and point `--policy.path` at the chosen `pretrained_model` directory. The robot config must expose camera names exactly `front` and `side`, each 640×480 at 30 FPS, preserve the six action names/order in dataset `meta/info.json`, and use the same motor calibration and `use_degrees` setting as collection. Supply the actual robot model (`so100_follower` or `so101_follower`), serial port, calibration ID, and camera indices/paths from the deployment machine. A template command is:

```bash
CUDA_VISIBLE_DEVICES=0 lerobot-rollout \
  --strategy.type=base \
  --inference.type=sync \
  --policy.path=/mnt/data/ningan/button_datasets/outputs/streaming_flow_v2_button_16_8_crop95/checkpoints/best_offline/pretrained_model \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=REPLACE_WITH_COLLECTION_CALIBRATION_ID \
  --robot.cameras='{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}' \
  --device=cuda \
  --fps=30 \
  --interpolation_multiplier=1 \
  --task='Push the button' \
  --duration=30
```

Replace all hardware placeholders and verify the initial pose, visible target, camera orientation, emergency stop, travel limits, and command units before allowing movement. The configured first-chunk latent action is the median first action from the newer batch (`[-11.098,-98.851,93.891,64.509,-3.59,0.866]` in dataset units). It is not directly sent, but a mismatched start pose can make the first prediction poor. Use an initial pose close to the demonstrated starts or override the rollout initial action based on the deployment setup and validate offline first. Reset the policy between episodes; `lerobot-rollout` does this for a new run.

An RTX 5090 model-only synthetic benchmark at 240×320, batch one, CUDA autocast, after warmup measured median 35.7 ms per new `16_8` chunk and 19.3 ms per new `8_1` chunk. This excludes camera capture, motor I/O, Python framing, and preprocessing; it is not a measured robot-control rate. The `8_1` policy replans every 33.3 ms tick, leaving only about 14 ms for the rest of the loop on that machine. Measure the complete loop before using `8_1` at 30 FPS. A slower loop changes the real time span of predicted actions and can defeat the intended temporal alignment; reducing the FPS without retraining is not an equivalent fix.

With the selected 20k checkpoints, the actual LeRobot synchronous inference engine (raw dataset observation to postprocessed six-motor action; no live camera capture or motor I/O) took about 1.1 ms on cached `16_8` ticks and 37–40 ms on replan ticks. The `8_1` policy took about 20–22 ms on every tick. The `16_8` replan tick already exceeds the 33.3 ms 30 FPS budget before hardware I/O; the `8_1` policy leaves only 11–13 ms for camera and motor operations. A complete hardware-loop timing and action-limit review is required before 30 FPS deployment.
