# Scaling Roadmap

The project should scale only after the rules, MCTS math, and single-process training path are correct and tested.

## Local Development

- Use CPU or a single local GPU for correctness and smoke training.
- Keep board size, network size, and MCTS simulations small while validating behavior.
- Prefer fast deterministic tests for rules, state encoding, MCTS selection, and replay sampling.
- Use local logs or lightweight TensorBoard/W&B runs for experiment tracking.

## Current Bottleneck

The current single-process pipeline runs self-play sequentially. MCTS makes many small neural-network inference calls, which underuses both CPU cores and GPU throughput.

See [Parallel Training Investigation](ParallelTrainingInvestigation.md) for the current bottleneck analysis.

## Near-Term Scaling

- Run multiple self-play worker processes.
- Batch neural-network inference requests from workers.
- Centralize checkpoint writes to avoid race conditions.
- Keep one trainer process responsible for network updates.
- Add an evaluator process that runs on a schedule instead of blocking self-play.

## Multi-GPU Scaling

- Use data-parallel training when model updates become the bottleneck.
- Keep self-play generation and training decoupled so actors can keep producing games while the learner trains.
- Use mixed precision where supported.
- Increase batch size through gradient accumulation when GPU memory is limited.

## Operations

- Keep generated games under `data/`.
- Keep model files under `checkpoints/`.
- Keep experiment metadata under `experiments/`.
- Do not commit generated training data, checkpoints, or local logs.
- Record config, code version, seed, checkpoint ID, and evaluation result for every serious run.

## Related

- [Self-Play Training](SelfPlayTraining.md)
- [Training Hyperparameters](TrainingHyperparameters.md)
- [Reproducibility](Reproducibility.md)
- [Experiments](Experiments.md)

