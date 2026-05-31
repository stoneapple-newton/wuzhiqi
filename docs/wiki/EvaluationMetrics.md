# Evaluation Metrics

Evaluation should compare agents under controlled settings.

## Core Metrics

- **Win rate vs random:** Basic sanity check.
- **Win rate vs previous checkpoint:** Regression check.
- **Win rate vs fixed MCTS baseline:** Search-quality check.
- **Average game length:** Detects abnormal early wins, illegal states, or stalled games.
- **Policy loss and value loss:** Training diagnostics.

## Protocol Notes

- Alternate colors evenly.
- Use fixed random seeds for reproducible eval suites.
- Keep evaluation games separate from training self-play games.

## Related

- [Checkpoints](Checkpoints.md)
- [Reproducibility](Reproducibility.md)
