# Streaming Flow Policy

This policy integrates an image-based Streaming Flow Policy into LeRobot.

The ResNet-based variants (`streaming_flow`, `streaming_flow_v2`, and
`streaming_flow_mamba_lite`) default to a separate, fully trainable encoder for
each camera. V3 and V4 use a shared, frozen CLIP vision encoder by default. V5
keeps the shared encoder but fine-tunes its vision backbone at a reduced
learning rate by default; its CLIP text encoder remains frozen.

## Files

- `configuration_streaming_flow.py`: policy config and CLI-facing hyperparameters.
- `modeling_streaming_flow.py`: original single-point ResNet/UNet implementation.
- `modeling_streaming_flow_v2.py`: multi-point ResNet/UNet implementation and aligned rollout support.
- `modeling_streaming_flow_v3.py`: pooled CLIP image/text conditioning with the UNet expert.
- `modeling_streaming_flow_v4.py`: token-level CLIP conditioning with a transformer expert.
- `modeling_streaming_flow_v5.py`: frequency-scaled V4 transformer and direct frequency prediction.
- `modeling_streaming_flow_mamba.py`: Mamba3, cross-chunk attention, and cached streaming inference.
- `modeling_streaming_flow_mamba_lite.py`: native v2 rollout/training wrapper with the
  temporal ResNet frontend and compact Mamba3 expert.
- `processor_streaming_flow.py`: LeRobot pre/post-processing hooks.

## Version Guide

All numbered versions use the same high-level SFP contract: observations
condition a velocity field, training samples points along the demonstration
trajectory, and rollout integrates the predicted velocity with Euler steps to
produce an action chunk. They also share min-max state/action normalization,
optional EMA, adaptive-frequency controls, and the same LeRobot pre/post-
processor. The versions are different model architectures, not interchangeable
configuration presets.

| CLI policy type | Observation/task encoder | Velocity expert | Main addition | Default rollout behavior |
| --- | --- | --- | --- | --- |
| `streaming_flow_v2` | independent ImageNet ResNet18 temporal encoder per camera; flattened state; no language | conditional 1-D UNet | multi-point SFP supervision plus optional previous-action alignment and a separate execution horizon | predict and execute 8 actions; legacy alignment unless explicitly enabled |
| `streaming_flow_v3` | shared frozen CLIP vision CLS features, pooled CLIP text, and flattened state | conditional 1-D UNet | natural-language task conditioning while retaining the V2 flow objective | predict and execute 8 actions; legacy alignment |
| `streaming_flow_v4` | shared frozen CLIP patch/text tokens, state tokens, and pooled summaries | 8-layer AdaLN-Zero cross-attention transformer | token-level visual/language grounding instead of one global conditioning vector | predict and execute 8 actions; legacy alignment |
| `streaming_flow_v5` | V4 tokens; shared CLIP vision is fine-tuned by default, CLIP text stays frozen | frequency-scaled V4 transformer | direct frequency prediction, frequency-scaled residual branches, and reduced-LR vision fine-tuning | predict and execute 8 actions; legacy alignment |

The table describes configuration defaults: `n_obs_steps=2`, `chunk_size=16`,
`n_action_steps=8`, `execution_horizon=None`, and
`use_previous_action_alignment=false`. `execution_horizon=None` means that the
whole `n_action_steps` prefix is executed before replanning. The MimicGen V2
recipes deliberately override the last two settings with
`execution_horizon=1` and `use_previous_action_alignment=true`.

### V2: multi-point ResNet SFP

V2 is the practical image-only version for PushT and MimicGen-style tasks. For
each camera, the temporal ResNet frontend combines the latest feature with its
first difference; with three or more observation steps it also includes a
second difference. Camera features and the observation-state window are
concatenated into one global conditioning vector for the conditional 1-D UNet.

Compared with the original `streaming_flow`, V2 samples
`sfp_num_train_points` flow locations per trajectory (default 4) instead of one.
It also supports:

- an independent, fully trainable ResNet18 for every camera by default;
- optional encoder freezing with `freeze_vision_encoder=true`;
- configurable initial rollout state (`auto`, `zero`, `constant`, or `state`);
- `use_previous_action_alignment=true`, which trains the path as
  `[a_{t-1}, a_t, ...]` and makes the first Euler update predict `a_t`;
- `execution_horizon`, which decouples the number of predicted actions from the
  number actually executed before the next observation.

For relative-OSC control, use the aligned mode and set the execution horizon
explicitly. Continuity is then updated from the last action actually returned
to the environment, not from the unexecuted end of the predicted chunk.

```bash
lerobot-train \
  --policy.type=streaming_flow_v2 \
  --policy.n_action_steps=8 \
  --policy.execution_horizon=1 \
  --policy.use_previous_action_alignment=true \
  --policy.use_separate_rgb_encoder_per_camera=true \
  --dataset.repo_id=<dataset-repo-id>
```

### V3: pooled CLIP image and text conditioning

V3 retains the V2 conditional UNet and multi-point objective but replaces the
ResNet features with CLIP. It temporally fuses CLIP vision CLS features and
adds a projected pooled CLIP text feature. The processor automatically adds a
tokenizer when text conditioning is enabled.

The default V3 contract uses one shared vision encoder for all cameras and
freezes both CLIP vision and text backbones (`sfp_freeze_clip=true`). The small
temporal/text projections and the UNet remain trainable. V3 therefore provides
language conditioning without the token-level cross-attention or larger
transformer expert introduced in V4.

```bash
lerobot-train \
  --policy.type=streaming_flow_v3 \
  --policy.vision_encoder_name=openai/clip-vit-base-patch16 \
  --policy.text_encoder_name=openai/clip-vit-base-patch16 \
  --dataset.repo_id=<language-conditioned-dataset>
```

V3 requires the `transformers` dependency and a task string that the processor
can tokenize. It keeps the legacy rollout defaults; the V2-specific aligned
MimicGen behavior should not be assumed merely because V3 inherits the V2
configuration class.

### V4: token-grounded transformer SFP

V4 keeps pooled CLIP summaries for global modulation and additionally exposes
compressed vision patch tokens, CLIP text tokens, and one token per state step.
The velocity expert is an AdaLN-Zero transformer whose action queries use
self-attention and cross-attend to that observation/task memory.

The default transformer has hidden size 768, 8 layers, 12 attention heads, a
3072-wide feed-forward network, and 16 pooled patch tokens per observation
frame. `transformer_visual_tokens_per_frame` must be a positive square because
the patch grid is adaptively pooled to a square spatial layout. CLIP vision and
text backbones are shared/frozen by default; their output projections and the
transformer are trainable.

```bash
lerobot-train \
  --policy.type=streaming_flow_v4 \
  --policy.transformer_hidden_dim=768 \
  --policy.transformer_num_layers=8 \
  --policy.transformer_num_heads=12 \
  --policy.transformer_visual_tokens_per_frame=16 \
  --dataset.repo_id=<language-conditioned-dataset>
```

### V5: frequency-scaled transformer and vision fine-tuning

V5 builds on V4 and makes the predicted SFP frequency an explicit control over
every transformer residual branch. A direct predictor is initialized at
`sfp_freq_init=1.0`; the active frequency is optionally clamped to
`[sfp_freq_min, sfp_freq_max]` (defaults 0.2 to 5.0). The residual multiplier is
the predicted frequency times a learned positive scale, initialized so that
the learned factor is 1.

Unlike V3/V4, V5 fine-tunes the shared CLIP vision backbone by default with a
separate optimizer group at `vision_encoder_lr_multiplier=0.1` times the main
learning rate. Set `sfp_finetune_clip_image=false` to freeze it. The text
backbone is still controlled by `sfp_freeze_clip` and remains frozen by
default.

```bash
lerobot-train \
  --policy.type=streaming_flow_v5 \
  --policy.sfp_freq_init=1.0 \
  --policy.sfp_finetune_clip_image=true \
  --policy.vision_encoder_lr_multiplier=0.1 \
  --dataset.repo_id=<language-conditioned-dataset>
```

### Choosing a version

- Use V2 for image/state tasks, independent camera encoders, and the aligned
  one-step execution recipe used by MimicGen.
- Use V3 when pooled natural-language conditioning is sufficient and the UNet
  expert is preferred.
- Use V4 when spatial patch tokens and token-level language cross-attention are
  important.
- Use V5 when using the V4 transformer with explicit frequency-scaled updates
  and CLIP vision fine-tuning.

Checkpoints are version-specific because the encoder and velocity-expert
parameter structures change between versions. V2 checkpoints trained without
previous-action alignment must also not be used to validate the aligned
relative-action contract; retrain after changing that action-time convention.

## Training

### Environment setup

Install the training and evaluation dependencies together with the target
environment. V3-V5 also need the CLIP/transformer extra.

```bash
# V2 with PushT
uv sync --locked --extra training --extra evaluation --extra pusht

# V3-V5 with LIBERO
uv sync --locked --extra training --extra evaluation --extra libero --extra multi_task_dit
```

MimicGen additionally requires a compatible manual installation of MimicGen,
robomimic, robosuite, and `h5py`; see the task-specific section below.

### Dataset contract

A training dataset must provide `action`, at least one visual observation, and
optionally `observation.state`. All configured camera features must have the
same shape. The policy builds its observation window from `n_obs_steps` and its
target window from `chunk_size`.

Images should enter the Streaming Flow processor as `uint8` in `[0, 255]` or as
floating point in `[0, 1]`. The processor converts `uint8` to `[0, 1]`; it does
not reinterpret an already floating-point `[-1, 1]` tensor. ResNet variants
apply ImageNet normalization internally when
`sfp_use_imagenet_visual_norm=true`, while CLIP variants apply CLIP's own mean
and standard deviation.

V3-V5 additionally require a meaningful per-episode task string. The
preprocessor tokenizes it into `observation.language.tokens` and
`observation.language.attention_mask`. An empty task fallback is technically
accepted by the evaluation loop but does not provide useful language
conditioning.

### Standard recipe

This V2 example trains locally, evaluates periodically, saves a checkpoint,
and keeps W&B disabled while preserving local JSONL logs:

```bash
CUDA_VISIBLE_DEVICES=0 uv run lerobot-train \
  --policy.type=streaming_flow_v2 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.n_obs_steps=2 \
  --policy.chunk_size=16 \
  --policy.n_action_steps=8 \
  --policy.sfp_num_train_points=8 \
  --policy.use_separate_rgb_encoder_per_camera=true \
  --policy.freeze_vision_encoder=false \
  --dataset.repo_id=lerobot/pusht_image \
  --dataset.image_transforms.enable=false \
  --env.type=pusht \
  --batch_size=256 \
  --num_workers=16 \
  --steps=20000 \
  --eval_freq=20000 \
  --eval.n_episodes=10 \
  --save_checkpoint=true \
  --save_freq=10000 \
  --log_freq=100 \
  --wandb.enable=false \
  --output_dir=outputs/train/pusht_streaming_flow_v2
```

For V3-V5, change `policy.type`, use a language-conditioned dataset, and select
the CLIP settings appropriate for the version. A representative V5 overlay is:

```text
--policy.type=streaming_flow_v5
--policy.sfp_num_train_points=8
--policy.sfp_use_clip_image_conditioning=true
--policy.sfp_use_clip_text_conditioning=true
--policy.vision_encoder_name=openai/clip-vit-base-patch16
--policy.text_encoder_name=openai/clip-vit-base-patch16
--policy.use_separate_rgb_encoder_per_camera=false
--policy.transformer_hidden_dim=768
--policy.transformer_num_layers=8
--policy.transformer_num_heads=12
--policy.transformer_ffn_dim=3072
--policy.sfp_freq_init=1.0
--policy.sfp_finetune_clip_image=true
--policy.vision_encoder_lr_multiplier=0.1
```

V3 uses the UNet settings (`down_dims`, `kernel_size`, and
`updownsample_type`) instead of transformer depth/width settings. V4 uses the
transformer settings but has neither `sfp_freq_init` nor
`sfp_finetune_clip_image`. Increasing `sfp_num_train_points` gives denser flow
supervision but also increases the number of velocity queries and training
memory. Reduce batch size when moving from V2/V3 to the larger V4/V5 expert.

If no validation environment is configured, set `eval_freq=0`. Training still
writes `train_metrics.jsonl`; environment validation writes
`eval_metrics.jsonl`. Each saved `pretrained_model` directory contains the
policy config, safetensor weights, training config, and the saved pre/post-
processors needed to reproduce normalization at inference time.

### Resume training

Resume from the checkpoint's saved `train_config.json`. This restores the
checkpoint configuration and training state, including optimizer, scheduler,
step, and random-number-generator state.

```bash
uv run lerobot-train \
  --resume=true \
  --config_path=outputs/train/<run>/checkpoints/last/pretrained_model/train_config.json
```

Do not change policy version, encoder sharing, action alignment, or observation
and action shapes when resuming. Start a new run and retrain when one of those
contracts changes.

Complete task-specific commands are maintained in:

- `pusht_v2.txt` for PushT V2;
- `libero_spatial.txt` for LIBERO V2/V4/V5 training and evaluation;
- `mimicgen_square_threading.txt` and
  `scripts/train_mimicgen_streaming_flow_v2_long.sh` for aligned MimicGen V2.

## Inference and Evaluation

### Simulator evaluation

Point `lerobot-eval` at a saved `pretrained_model` directory. The policy type,
model hyperparameters, feature shapes, normalization statistics, and processor
configuration are loaded from that directory, so do not also pass a different
`policy.type`.

```bash
POLICY_PATH=outputs/train/<run>/checkpoints/last/pretrained_model

CUDA_VISIBLE_DEVICES=0 uv run lerobot-eval \
  --policy.path="$POLICY_PATH" \
  --policy.device=cuda \
  --env.type=pusht \
  --eval.batch_size=1 \
  --eval.n_episodes=20 \
  --eval.use_async_envs=false \
  --seed=100000 \
  --output_dir=outputs/eval/<run>
```

For LIBERO, add the task suite and relative-control environment settings:

```text
--env.type=libero
--env.task=libero_spatial
--env.control_mode=relative
--env.max_parallel_tasks=1
--env.observation_width=256
--env.observation_height=256
```

For MimicGen, use `env.type=mimicgen`, the exact task name and source HDF5 path,
and the same episode length used during training. `lerobot-eval` writes summary
and per-episode results to `eval_info.json`, including success, reward, episode
length, policy latency, and available predicted-frequency diagnostics. Rendered
episodes are stored below the evaluation output directory when video rendering
is enabled.

### Rollout state and action queue

Call `policy.reset()` once at every environment reset. Then call
`policy.select_action()` once per control step; it maintains the observation
window, action queue, EMA rollout model, and previous action state. The returned
action is still normalized and must pass through the saved postprocessor before
`env.step()` so it is unnormalized, clipped to dataset bounds, and moved to CPU.

V2's `select_action()` honors `execution_horizon`: it predicts
`n_action_steps`, queues only the configured prefix, and replans when that
prefix has been executed. V3-V5 currently preserve the legacy queue
implementation and execute all `n_action_steps`; their inherited
`execution_horizon` field does not shorten that queue. Use V2 for the current
aligned one-action replan contract.

V3-V5 inference must receive the same kind of task description used in
training. The standard evaluation loop first looks for the environment's
`task_description`, then `task`, and passes that string through the saved CLIP
tokenizer.

### Minimal programmatic inference

The normal evaluation command handles environment processors as well. For a
custom loop, the policy-side sequence is:

```python
from pathlib import Path

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors

checkpoint = Path("outputs/train/<run>/checkpoints/last/pretrained_model")
config = PreTrainedConfig.from_pretrained(checkpoint)
config.device = "cuda"

policy_cls = get_policy_class(config.type)
policy = policy_cls.from_pretrained(checkpoint, config=config).eval()
preprocessor, postprocessor = make_pre_post_processors(
    config,
    pretrained_path=str(checkpoint),
)

policy.reset()  # repeat at the start of every episode
raw_observation = env.reset()[0]
raw_observation["task"] = "the same task description convention used for training"

while True:
    batch = preprocessor(raw_observation)
    with torch.inference_mode():
        normalized_action = policy.select_action(batch)
    action = postprocessor(normalized_action)
    raw_observation, reward, terminated, truncated, info = env.step(action.numpy())
    if terminated or truncated:
        break
```

Real environments may need their own environment pre/post-processors around
the policy processors, as `lerobot-eval` does. Preserve the training camera-key
order, image range, state definition, action representation, task wording
convention, and control frequency. A tensor with the right shape but a changed
meaning is not a compatible inference input.

## MimicGen Square_D1 and Threading_D1

`convert_mimicgen_hdf5_to_lerobot.py` preserves the native 20 Hz relative-OSC
actions and converts both 84x84 cameras plus the 9-D
`eef_pos + eef_quat + gripper_qpos` observation. `env.type=mimicgen` provides a
robomimic/robosuite validation environment with seeded resets, a 400-step
horizon, and success reporting. It needs the released MimicGen environment
package, robomimic, robosuite, and `h5py` in the LeRobot Python environment.

Complete conversion, training, periodic validation, and standalone evaluation
commands are in `mimicgen_square_threading.txt`.

## Rollout Logic

The MimicGen `streaming_flow_v2` launch enables `use_previous_action_alignment`
and aligns relative-control training and rollout as follows:

1. Encode the last `n_obs_steps` RGB observations.
2. Train the flow path as `[a_{t-1}, a_t, ...]`; the previous command is the path state at time zero.
3. At rollout, start from the last action actually returned to the environment. A task-specific
   constant is used only at episode start.
4. Integrate `n_action_steps` predictions. The first Euler update produces `a_t`, rather than
   skipping ahead to `a_{t+1}`.
5. Commit only `execution_horizon` actions before observing and replanning. The MimicGen commands
   use an eight-action prediction with a one-action execution horizon.

## Streaming Flow Mamba

`streaming_flow_mamba` keeps ControlFlow's interpolation targets, multi-point
velocity supervision, Euler integration, and receding-horizon action queue. It
changes the velocity expert as follows:

```text
executed action tail -> [a, delta-a, delta2-a] -> Mamba3 -> history K/V
current flow points  -> action/time/lambda tokens -> observation attention
                                              -> history attention
                                              -> current Mamba3 -> velocity
```

Training uses an ordered `(B, M, A)` sequence of multi-point flow queries by
default. Inference processes one `(B, 1, A)` token at a time with chunk-local
Mamba state. Observation K/V, history K/V, and the previous-tail encoding are
computed once per chunk. The state is discarded after the chunk; only actions
actually returned by `select_action` enter the next history tail.

The complex MIMO trapezoidal recurrence has:

- a Triton forward/backward scan for training;
- a fused Triton state-update kernel for incremental inference;
- a numerically equivalent PyTorch CPU/CUDA fallback.

Useful ablations:

```text
--policy.multi_point_mode=ordered_sequence|independent
--policy.history_encoder=mamba3|linear|none
--policy.history_training_mode=demonstration|generated|scheduled_mix
--policy.generated_history_probability=0.0
--policy.use_cross_chunk_attention=true|false
--policy.use_observation_cross_attention=true|false
--policy.lambda_condition_token=true|false
--policy.lambda_condition_gate=true|false
--policy.lambda_condition_step_size=true|false
--policy.inference_mode=incremental|full_prefix
--policy.mamba_use_cuda_kernel=true|false
```

`generated` and `scheduled_mix` expect `generated_action_history` (and
optionally `generated_action_history_is_pad`) in the training batch. These
histories should come from an offline sequential rollout cache and are detached
before use. This avoids doubling online training compute while allowing a
scheduled transition from demonstration history to model-generated, executed
history.

Example:

```bash
lerobot-train \
  --policy.type=streaming_flow_mamba \
  --policy.mamba_hidden_dim=256 \
  --policy.mamba_history_depth=2 \
  --policy.mamba_current_depth=4 \
  --policy.previous_tail_len=4 \
  --policy.sfp_num_train_points=8 \
  --policy.multi_point_mode=ordered_sequence \
  --policy.inference_mode=incremental \
  --dataset.repo_id=lerobot/pusht_image \
  --env.type=pusht
```

## Streaming Flow Mamba Lite

`streaming_flow_mamba_lite` is the lower-cost, closed-set multi-task variant.
Its processor selection, observation queues, and initial-action utilities come
from `modeling_streaming_flow_v2.py`, while its temporal history and rollout
contract remain model-specific. It does not
inherit `DiSPoConfig` or use `DiSPoPolicy`; only the low-level fused Mamba3
recurrence is shared.
It preserves the parts of `streaming_flow_mamba` that model action dynamics:

- ordered multi-point ControlFlow supervision;
- previous executed-action tail with position, velocity, and acceleration;
- Mamba3 history and current-action streams;
- observation and cross-chunk SDPA cross-attention;
- cached K/V, recurrent state, and the fused Triton inference step.

It replaces CLIP image and text encoders with:

1. The `streaming_flow_v2` ResNet18 temporal encoder. Each camera contributes
   one compact token. For two observations the feature contains the latest
   frame and its temporal difference; with three or more observations it also
   contains acceleration.
2. One token for each proprioceptive observation step.
3. One learned `task_index` token. LeRobot datasets already carry
   `task_index`, so this provides explicit multi-task disambiguation without a
   language transformer.

The token count per control chunk is therefore only
`num_cameras + n_obs_steps + 1`, and its observation/history K/V is projected
once and reused for all Euler steps.

With two cameras, 14-D state/action, and the defaults, the shared-ResNet model
is approximately 16.3M parameters. A local RTX 5090
microbenchmark measured the Mamba velocity path, including cache construction
and eight incremental steps, at 9.9 ms median for batch size one. The shared
ResNet18 temporal encoder measured 1.46 ms for one camera with two 64x64
frames. These are component measurements, not environment end-to-end latency.

| Property | `streaming_flow_mamba` | `streaming_flow_mamba_lite` |
| --- | --- | --- |
| Vision | CLIP patch tokens | ResNet18, one token/camera |
| Task signal | CLIP language tokens | learned dataset `task_index` |
| Open-vocabulary task interface | yes; transfer is data-dependent | no; closed set |
| Default action hidden size | 256 | 192 |
| Default Mamba depth | history 2 + current 4 | history 1 + current 3 |
| Default MIMO rank/state | 4 / 16 | 4 / 8 |
| Transformers package at model runtime | required | not required |
| Incremental CUDA Mamba3 step | yes | yes |

The lite task table defaults to 256 entries. Set it to the dataset's actual
task count to avoid unused parameters. For a strictly single-task policy,
disable the table:

```text
--policy.lite_num_tasks=10
--policy.lite_use_task_embedding=true
# or, for a single task:
--policy.lite_use_task_embedding=false
```

Example:

```bash
lerobot-train \
  --policy.type=streaming_flow_mamba_lite \
  --policy.lite_num_tasks=10 \
  --policy.sfp_num_train_points=8 \
  --policy.multi_point_mode=ordered_sequence \
  --policy.inference_mode=incremental \
  --policy.mamba_use_cuda_kernel=true \
  --dataset.repo_id=<multi-task-dataset>
```

`task_index` must also be supplied during rollout when task embedding is
enabled for multi-task evaluation. A single-task environment that does not
provide it falls back to task 0 during evaluation; training still requires a
real task index by default. This version can learn many known tasks in one
policy, but use the full `streaming_flow_mamba` when natural-language
instructions or unseen task phrasing are required.

Training action samples include `previous_tail_len` negative action deltas
followed by the current trajectory. The Streaming Flow v2 policy wrapper consumes
this layout during training, while `select_action` stores only actions actually
dequeued by the environment for the next cross-chunk history.

The recurrent dimensions use the Streaming Flow Mamba config names:

```text
--policy.mamba_hidden_dim=192
--policy.mamba_current_depth=3
--policy.mamba_d_state=8
--policy.mamba_mimo_rank=4
```

PushT and LIBERO training/evaluation commands are available in
`policies/streaming_flow/streaming_flow_mamba_lite.txt`.
