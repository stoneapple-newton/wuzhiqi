# Single Process Pipeline

The current training workflow is implemented in `src/wuzhiqi/train.py`.

The default config is currently set for a warm-start smoke reset: it loads
`checkpoints/current_policy_15x15.pt` as model weights, but does not resume the
old optimizer, replay buffer, batch counter, random states, or adaptive
learning-rate multiplier.

## Startup

1. Load `config/default.yaml`.
2. Seed Python, NumPy, and Torch if `evaluation.seed` is configured.
3. Create the `Board` and `Game`.
4. Create the `PolicyValueNet`.
5. Create a self-play `MCTSPlayer` using the policy-value network.
6. Optionally resume from `training.checkpoint_path` if `training.resume` is enabled.

## Per-Batch Loop

For each batch:

1. Collect one or more self-play games.
2. Augment positions with board symmetries.
3. Append samples to the replay buffer.
4. Train the policy-value network if the replay buffer is larger than `training.batch_size`.
5. Periodically evaluate, save model weights, and save the full training checkpoint.

## Self-Play

Self-play uses `Game.start_self_play()` and the neural-network-guided `MCTSPlayer`.

For each move:

1. Run `mcts.simulations` PUCT playouts.
2. Convert root visit counts into move probabilities.
3. Sample a move with Dirichlet exploration noise.
4. Store the current state, move probabilities, and later the game outcome target.

## Training Update

`policy_update()` samples from the replay buffer and trains for up to `training.epochs` epochs.

Logged values include:

- `kl`: policy shift between old and new network outputs.
- `lr_multiplier`: adaptive multiplier for the configured learning rate.
- `effective_lr`: actual learning rate applied to the optimizer.
- `loss`: combined value loss and policy loss.
- `policy_loss`: policy cross-entropy component.
- `value_loss`: value MSE component.
- `entropy`: spread of the policy distribution.
- `replay_buffer`: current augmented replay-buffer size.
- `target_counts`: sampled value-target balance for `-1`, `0`, and `1`.
- `explained_var_old` and `explained_var_new`: value-head fit before and after the update.

## Evaluation

Evaluation runs every `training.check_freq` batches.

The current config uses a fast heuristic opponent:

```yaml
evaluation:
  opponent: heuristic
```

The older pure MCTS baseline is still available by setting:

```yaml
evaluation:
  opponent: pure_mcts
```

## Checkpointing

The trainer saves:

- current model weights
- best model weights
- full training checkpoint

The full checkpoint includes model weights, optimizer state, replay buffer, batch counter, adaptive learning-rate multiplier, best win ratio, and random number generator state.

## Scaling Limitations

The current workflow is simple and testable, but it is not hardware-efficient:

- Self-play is sequential.
- MCTS playouts are sequential.
- Network inference during MCTS is one board state at a time.
- GPU inference is not batched across workers.
- CPU cores are not used for parallel self-play.

See [Parallel Training Investigation](../ParallelTrainingInvestigation.md) for the recommended scaling direction.
