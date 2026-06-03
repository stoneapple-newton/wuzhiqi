# Training Hyperparameters

Initial defaults should favor correctness and quick iteration.

## Local Smoke Defaults

- **Board size:** `6 x 6`
- **Win length:** `4`
- **MCTS simulations per move:** `100`
- **Self-play games per iteration:** `1`
- **Batch size:** `32`
- **Replay buffer size:** `10_000` positions
- **Learning rate:** `1e-3`
- **Weight decay:** `1e-4`
- **cpuct:** `5.0`
- **Temperature:** `1.0` for template-style self-play collection.

## Full Gomoku Target

- **Board size:** `15 x 15`
- **Win length:** `5`
- **MCTS simulations per move:** start at `50` to `100`, then scale upward.

## Current Warm-Start Smoke Defaults

The default 15x15 configuration is currently set for a short warm-start reset run:

- **Initial model:** `checkpoints/current_policy_15x15.pt`
- **Resume checkpoint:** disabled; optimizer state, replay buffer, batch counter, and adaptive LR multiplier reset.
- **Output checkpoints:** `checkpoints/warm_reset_*_15x15.pt`
- **MCTS simulations per move:** `50`
- **Batch size:** `128`
- **Replay buffer size:** `20_000` positions
- **Learning rate:** `5e-4`
- **Epochs per update:** `2`
- **KL target:** `0.01`
- **Evaluation games:** `4`

## Scaled Defaults

- **MCTS simulations per move:** `400` to `800`
- **Self-play games per iteration:** `100+`
- **Batch size:** `256`
- **Replay buffer size:** `100_000` to `200_000` positions
- **Experiment tracking:** TensorBoard, W&B, or local CSV logs.

## Related

- [Network Architecture](NetworkArchitecture.md)
- [Experiments](Experiments.md)
