# MCTS Design

Monte Carlo Tree Search selects moves by combining accumulated search value with policy priors from the neural network. The target algorithm is AlphaZero-style PUCT.

## Edge Statistics

Each searched action from a state tracks:

- `P`: prior probability from the policy head.
- `N`: visit count.
- `W`: total backed-up value.
- `Q`: mean value, usually `W / N`.

Values are from the perspective of the player to move at the node being evaluated.

## PUCT Selection

During selection, choose the action with the largest score:

```text
Q(s, a) + cpuct * P(s, a) * sqrt(sum_b N(s, b)) / (1 + N(s, a))
```

The `cpuct` constant controls exploration. Larger values give policy priors more influence, especially early in search.

## Search Loop

1. **Selection:** Walk from the root by repeatedly choosing the highest PUCT score.
2. **Expansion:** When an unexpanded non-terminal state is reached, query the policy-value network.
3. **Masking:** Remove illegal actions from the policy prior and renormalize the remaining moves.
4. **Evaluation:** Use the value head output for non-terminal leaves, or the terminal game result for completed positions.
5. **Backpropagation:** Update `N`, `W`, and `Q` along the path, flipping perspective at each ply.

## Root Policy

After simulations finish, convert root visit counts into the training target policy:

```text
pi(a | s) = N(s, a) ** (1 / temperature) / sum_b N(s, b) ** (1 / temperature)
```

Use higher temperature during early self-play moves for exploration and near-zero temperature for deterministic evaluation.

## Test Focus

- Legal moves are the only expandable actions.
- Terminal states return exact outcomes without network calls.
- Visit counts increase once per simulation.
- Backpropagation flips player perspective correctly.
- PUCT selection is deterministic when scores are tied by a defined tie-break rule.

## Related

- [Self-Play Training](SelfPlayTraining.md)
- [Training Hyperparameters](TrainingHyperparameters.md)
- [Evaluation Metrics](EvaluationMetrics.md)

