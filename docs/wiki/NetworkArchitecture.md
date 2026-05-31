# Network Architecture

The target model is a policy-value residual network inspired by AlphaZero.

## Input

- Tensor from [State Representation](StateRepresentation.md).
- Default spatial size is `15 x 15`.

## Body

- Initial convolution over input planes.
- Stack of residual convolution blocks.
- Candidate starting point: 6 to 10 blocks with 64 to 128 channels for local development.
- Scale toward 10 to 20 blocks with 128 to 256 channels after game logic, MCTS, and training tests are stable.

## Heads

- **Policy head:** Outputs one logit per board point.
- **Value head:** Outputs a scalar value in `[-1, 1]` from the current player's perspective.

## Related

- [State Representation](StateRepresentation.md)
- [Training Hyperparameters](TrainingHyperparameters.md)
