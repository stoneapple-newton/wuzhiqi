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

## Current Clean-Start Training Defaults

The default 15x15 configuration is currently set for a clean random restart:

- **Initial model:** none; start from random weights.
- **Resume checkpoint:** disabled; optimizer state, replay buffer, batch counter, and adaptive LR multiplier start clean.
- **Output checkpoints:** `checkpoints/clean_start_*_15x15.pt`
- **Structured log:** `experiments/clean_start_training_15x15.jsonl`
- **MCTS simulations per move:** `400`
- **Self-play start mode:** alternate between player 1 and player 2 starts for exploration diversity.
- **Batch size:** `128`
- **Replay buffer size:** `20_000` positions
- **Learning rate:** `5e-4`
- **Epochs per update:** `5`
- **KL target:** `0.01`
- **Evaluation games:** `10`

## Smoke Run Incident And Fix

The previous warm-reset smoke run used fixed-start self-play. Its replay buffer became perfectly separable by the turn/parity plane: one parity contained only `-1` value targets and the other contained only `+1` targets. The value head then reported near-perfect replay fit while the model still lost evaluation games.

Warm-starting from that model did not recover the run because MCTS reinforced the bad value shortcut. The current fix quarantines the warm-reset and mixed-start checkpoint families, disables the fourth input plane, starts from random weights, alternates self-play start player, logs the initial empty-board value, and writes JSONL diagnostics with reserved-plane target counts plus first/second mover outcome counts.

## Scaled Defaults

- **MCTS simulations per move:** `400` to `800`
- **Self-play games per iteration:** `100+`
- **Batch size:** `256`
- **Replay buffer size:** `100_000` to `200_000` positions
- **Experiment tracking:** TensorBoard, W&B, or local CSV logs.

## Related

- [Network Architecture](NetworkArchitecture.md)
- [Experiments](Experiments.md)
