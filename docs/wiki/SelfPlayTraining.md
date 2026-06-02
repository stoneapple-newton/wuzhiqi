# Self-Play Training

Self-play is the core data-generation loop for the AlphaZero-style Gomoku agent. The current local workflow is intentionally small; this page describes the target pipeline imported from the raw research notes.

## Training Sample

Each move in a self-play game produces one sample:

- `state`: encoded board from the current player's perspective.
- `policy`: MCTS visit-count distribution over board actions.
- `value`: final game outcome from the sampled state's current-player perspective.

The standard target tuple is `(state, policy, value)`.

## Iteration Loop

1. Load the current policy-value network.
2. Run MCTS-guided self-play games.
3. Store all move samples in a replay buffer.
4. Train the network on mini-batches from recent buffer samples.
5. Evaluate the updated network against baselines or the previous best checkpoint.
6. Promote and checkpoint the model when evaluation criteria are met.

## Loss

Use a combined policy and value loss:

```text
loss = value_mse(predicted_value, target_value)
     + policy_cross_entropy(predicted_policy_logits, target_policy)
     + weight_decay
```

The value target is usually `-1`, `0`, or `1`. The policy target is the normalized MCTS visit-count distribution.

## Replay Buffer

- Keep recent positions in a bounded ring buffer.
- Sample uniformly for the initial implementation.
- Prefer storing metadata with each generated game: board size, win length, config hash, model checkpoint, and random seed.
- Do not commit generated self-play data.

## Data Augmentation

Gomoku board symmetries can multiply samples without changing outcomes:

- Rotations by 90, 180, and 270 degrees.
- Horizontal, vertical, and diagonal reflections.
- Apply the same transform to the board tensor and policy vector.

## Promotion Gate

Evaluation should be deterministic and repeatable before it controls checkpoint promotion. Candidate gates include:

- Win rate against the previous checkpoint.
- Win rate against fixed random or heuristic baselines.
- Elo estimate from a checkpoint ladder.
- Fixed tactical test positions.

## Related

- [Current Training Workflow](training-workflow/README.md)
- [MCTS Design](MCTSDesign.md)
- [Training Hyperparameters](TrainingHyperparameters.md)
- [Evaluation Metrics](EvaluationMetrics.md)
- [Checkpoints](Checkpoints.md)

