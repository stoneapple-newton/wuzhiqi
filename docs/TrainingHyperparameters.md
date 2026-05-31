# Training Hyperparameters

Initial defaults should favor correctness and quick iteration.

## Local Smoke Defaults

- **Board size:** `15`
- **MCTS simulations per move:** `50` to `100`
- **Self-play games per iteration:** `4` to `16`
- **Batch size:** `32`
- **Replay buffer size:** `10_000` positions
- **Learning rate:** `1e-3`
- **Weight decay:** `1e-4`
- **cpuct:** `1.5`
- **Temperature:** `1.0` for opening moves, then near-greedy later.

## Scaled Defaults

- **MCTS simulations per move:** `400` to `800`
- **Self-play games per iteration:** `100+`
- **Batch size:** `256`
- **Replay buffer size:** `100_000` to `200_000` positions
- **Experiment tracking:** TensorBoard, W&B, or local CSV logs.

## Related

- [Network Architecture](NetworkArchitecture.md)
- [Experiments](Experiments.md)
