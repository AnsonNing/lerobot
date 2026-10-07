# Push-button Streaming Flow V2 training result (2026-09-14)

The matched Diffusion Policy runs and cross-policy offline comparison are documented in `DIFFUSION_COMPARISON.md`.

Two independent policies trained from scratch on the same 201-episode, 32,501-frame source-stratified training split. The raw merged dataset has 224 episodes and 36,393 frames. Twenty-two episodes (18 old, 4 new) were held out; the 42-frame near-static episode 165 was excluded from training but retained in the raw merge. Both policies used seed 100000, batch size 16, two independent fully trainable ImageNet ResNet18 encoders, `per_dim` action normalization, 240×320 resize, 228×304 random training crop, two observation frames, previous-action alignment, EMA, and local logging with W&B disabled.

| Variant | Aligned path length | New actions predicted | Actions executed before replan | Last complete checkpoint | Best offline checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: |
| `16_8` | 17 (`a_(t-1)` + 16 future) | 16 | 8 | 40,000 | 20,000 |
| `8_1` | 9 (`a_(t-1)` + 8 future) | 8 | 1 | 40,000 | 20,000 |

Held-out teacher-forced SFP loss using the EMA model and center cropping:

| Variant | Step | Old batch: 2,895 frames | New batch: 955 frames |
| --- | ---: | ---: | ---: |
| `16_8` | 10,000 | 0.08399 | 0.05397 |
| `16_8` | **20,000** | **0.07246** | **0.04885** |
| `16_8` | 30,000 | 0.07455 | 0.05082 |
| `16_8` | 40,000 | 0.07918 | 0.05911 |
| `8_1` | 10,000 | 0.06212 | 0.03925 |
| `8_1` | **20,000** | **0.05356** | **0.03550** |
| `8_1` | 30,000 | 0.05836 | 0.04068 |
| `8_1` | 40,000 | 0.06091 | 0.04188 |

Both source groups worsened for both variants after 20k, so training was stopped after the 40k checkpoints instead of completing the configured 50k. Main processes stopped after approximately 41,600 (`16_8`) and 41,300 (`8_1`) logged optimizer steps; steps after 40k have no saved weights. `checkpoints/best_offline` links to `020000`, and `checkpoints/last` to `040000`. After the 2026-10-01 disk cleanup, only the 20k best and 40k last checkpoint directories remain for each variant; the table above and held-out JSON results still record the removed 10k/30k evaluations. Training logs are under `/mnt/data/ningan/button_datasets/outputs/logs/train_button_*.log`; per-step local metrics are in each output directory's `train_metrics.jsonl`; held-out results are under `outputs/logs/val_button_*_*.json`.

Use these policy directories, each including `model.safetensors`, `config.json`, and saved pre/post-processors:

- `/mnt/data/ningan/button_datasets/outputs/streaming_flow_v2_button_16_8_crop95/checkpoints/best_offline/pretrained_model`
- `/mnt/data/ningan/button_datasets/outputs/streaming_flow_v2_button_8_1_crop95/checkpoints/best_offline/pretrained_model`

The absolute flow losses are not directly comparable across different predicted horizon lengths. This is an offline fitting metric, not a button-press success or safety measurement. The merged dataset metadata statistics include held-out episodes, so the offline metric has minor normalization-statistics leakage. The archives contain no success labels. Before robot movement, verify the exact `front`/`side` camera viewpoints, motor ordering/calibration, start pose, command units, travel limits, emergency stop, and complete 30 FPS loop latency. On the tested RTX 5090, the selected checkpoint's LeRobot synchronous policy pipeline (raw saved observation to postprocessed action, without live camera/motor I/O) took 37–40 ms for each `16_8` replan tick and about 1.1 ms for each cached tick. `8_1` took 20–22 ms every tick. The full hardware-loop timing remains unmeasured, and `16_8` replan ticks already exceed a 33.3 ms frame budget before I/O. The LeRobot rollout command and options are documented in `/home/ningan/lerobot/scripts/README_push_button_streaming_flow_v2.md`.
