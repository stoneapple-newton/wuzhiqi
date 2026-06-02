# Network Architecture

The current implementation uses the compact PyTorch policy-value CNN from the
`AlphaZero_Gomoku` template. The longer-term target remains a residual network
inspired by AlphaZero.

## Input

- Tensor from [State Representation](StateRepresentation.md).
- Default spatial size is `6 x 6` for local smoke training; `15 x 15` remains the full Gomoku target.

## Body

- Current implementation: three convolution layers over four input planes.
- Future scaled implementation: initial convolution over input planes.
- Stack of residual convolution blocks.
- Candidate starting point: 6 to 10 blocks with 64 to 128 channels for local development.
- Scale toward 10 to 20 blocks with 128 to 256 channels after game logic, MCTS, and training tests are stable.

## Heads

- **Policy head:** Outputs one logit per board point.
- **Value head:** Outputs a scalar value in `[-1, 1]` from the current player's perspective.

## Related

- [State Representation](StateRepresentation.md)
- [Training Hyperparameters](TrainingHyperparameters.md)
