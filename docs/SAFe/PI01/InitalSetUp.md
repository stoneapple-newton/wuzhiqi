# PI01 User Stories

These stories cover the first implementation slice for the Wuzhiqi AlphaZero project. Each story has its own page with scope, acceptance criteria, implementation notes, and test guidance.

## Stories

- [US-201: Reset Game State](US-201-reset-game-state.md)
- [US-202: Detect Wins](US-202-detect-wins.md)
- [US-301: Run MCTS for a Move](US-301-run-mcts-for-a-move.md)
- [US-401: Compare Two Agents](US-401-compare-two-agents.md)

## Planning Notes

- Keep the game engine stories dependency-light so they can be tested without neural network or training dependencies.
- Treat deterministic game-rule tests as required before connecting the environment to MCTS or self-play.
- Keep evaluation and MCTS interfaces explicit enough that later network-backed agents can replace simple baselines without changing the runner contract.
