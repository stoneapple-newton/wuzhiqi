# Project Roadmap

This roadmap summarizes the raw project plan in `docs/raw/overall-structure.md` and keeps the working structure aligned with the current repository.

## Target System

- Gomoku rules and environment logic.
- Deterministic state encoding and legal move masks.
- Neural policy-value network.
- PUCT Monte Carlo Tree Search.
- MCTS-guided self-play data generation.
- Replay buffer and training loop.
- Evaluation, checkpointing, and experiment tracking.
- Scaling path from local smoke runs to multi-process and multi-GPU training.

## Implementation Order

1. Establish the repository scaffold, documentation, config files, and test harness.
2. Implement dependency-light game rules, legal move generation, terminal detection, and state encoding.
3. Add policy-value network interfaces and small local model defaults.
4. Implement PUCT MCTS and verify its math with deterministic tests.
5. Generate self-play samples and store `(state, policy, value)` training data.
6. Train from replay buffer samples and checkpoint reproducibly.
7. Evaluate checkpoints against baselines and prior models.
8. Investigate multiprocessing self-play and batched inference for scale.

## Documentation Map

- [Rules](Rules.md)
- [State Representation](StateRepresentation.md)
- [Network Architecture](NetworkArchitecture.md)
- [MCTS Design](MCTSDesign.md)
- [Self-Play Training](SelfPlayTraining.md)
- [Training Hyperparameters](TrainingHyperparameters.md)
- [Scaling Roadmap](ScalingRoadmap.md)
- [Evaluation Metrics](EvaluationMetrics.md)
- [Checkpoints](Checkpoints.md)
- [Reproducibility](Reproducibility.md)

## Planning Map

- [SAFe Epics](../SAFe/Epics.md)
- [SAFe Features](../SAFe/Features.md)
- [User Stories](../SAFe/UserStories.md)
- [PI Planning](../SAFe/PI_Planning.md)
- [Definition of Done](../SAFe/DefinitionOfDone.md)

