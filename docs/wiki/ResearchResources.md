# Research Resources

This page imports the useful resource map from the raw research notes while keeping the working docs concise.

## Foundational Papers

- Silver et al., 2017, AlphaGo Zero: policy-value network, self-play, and MCTS-guided improvement.
- Silver et al., 2018, AlphaZero: generalization of the approach to chess, shogi, and Go.
- Gomoku-specific AlphaZero papers and implementations can guide board-size choices, first-player bias handling, and evaluation protocols.

## Implementations To Study

- `junxiaosong/AlphaZero_Gomoku`: compact educational AlphaZero Gomoku implementation.
- `Nagi-ovo/AlphaZero-Gomoku`: PyTorch self-play implementation with replay and experiment logging.
- `suragnair/alpha-zero-general`: general AlphaZero tutorial implementation.
- KataGo and KataGomo: useful references for high-performance search, batching, and large-scale training.
- OpenSpiel: useful reference for game environments, evaluation, and RL experiments.

## Frameworks

- PyTorch for the neural policy-value network and training loop.
- NumPy for dependency-light game logic and state encoding.
- Ray or multiprocessing for future parallel self-play.
- TensorBoard, W&B, MLflow, or local CSV logs for experiment tracking.

## Research Questions

- What board size and win length should be used for each development milestone?
- How many MCTS simulations are needed before self-play data quality improves meaningfully?
- What network size fits the available GPU memory without starving search throughput?
- Which evaluation baselines are stable enough for checkpoint promotion?
- When does batched inference become more valuable than increasing model capacity?

## Related

- [Project Roadmap](ProjectRoadmap.md)
- [Network Architecture](NetworkArchitecture.md)
- [MCTS Design](MCTSDesign.md)
- [Scaling Roadmap](ScalingRoadmap.md)

