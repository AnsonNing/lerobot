# Streaming Flow Policy

This policy integrates an image-based Streaming Flow Policy into LeRobot.

## Files

- `configuration_streaming_flow.py`: policy config and CLI-facing hyperparameters.
- `modeling_streaming_flow.py`: training loss, rollout integration, and the SFP backbone.
- `processor_streaming_flow.py`: LeRobot pre/post-processing hooks.

## Rollout Logic

The rollout path follows the original notebook-style chunked integration loop:

1. Encode the last `n_obs_steps` RGB observations.
2. Predict an adaptive rollout frequency.
3. Start from the previous action state. On the first chunk, initialize from the current
   agent/state observation when available, otherwise from the image centroid, matching the
   notebook rollout behavior.
4. Integrate the velocity field for `n_action_steps`.
5. Cache the final action state for the next chunk.

## Example

```bash
lerobot-train \
  --policy.type=streaming_flow \
  --dataset.repo_id=lerobot/pusht_image \
  --env.type=pusht
```
