# US-401: Compare Two Agents

## User Story

- **As an** experiment owner,
- **I want** a repeatable evaluation match runner,
- **So that** I can detect regressions between checkpoints.

## Feature and Epic

- **Feature:** FTR-401: Evaluation Harness
- **Epic:** EPI-004: Evaluation and Experimentation
- **PI:** PI-3: Evaluation and Scale

## Scope

This story defines a repeatable match runner for comparing two agents under the same Gomoku rules. The runner should support baseline agents, MCTS agents, and future checkpoint-backed agents through a common move-selection interface.

In scope:

- Agent-vs-agent match execution.
- Alternating first-player color across games.
- Deterministic seeding.
- Result aggregation for wins, losses, draws, and game length.
- Basic validation that agents only choose legal moves.

Out of scope:

- Elo calculation.
- Tournament scheduling across many agents.
- Distributed evaluation workers.
- Model checkpoint loading details beyond accepting an agent implementation.
- Experiment dashboard integration.

## Acceptance Criteria

- Agents alternate first-player color.
- Evaluation accepts a random seed.
- Results include wins, losses, draws, and average game length.
- Illegal agent moves are rejected or reported as evaluation failures.
- Match counts and color assignments are reproducible for the same seed and configuration.
- The result format is structured enough for later experiment tracking.

## Implementation Notes

- Define a small agent interface that receives a game state and returns a move.
- Keep the runner separate from training so evaluation can run against random, heuristic, MCTS, or checkpoint-backed agents.
- Record results from a consistent perspective, such as agent A versus agent B, while also tracking color-specific outcomes.
- Make maximum game length explicit, even if the default is the full board capacity.
- Avoid writing local experiment logs unless the caller explicitly asks for an output path.

## Test Guidance

- Use deterministic stub agents for predictable outcomes.
- Verify first-player assignment alternates across an even number of games.
- Verify the same seed and agents produce the same aggregate result.
- Verify draws are counted when the board fills with no winner.
- Verify illegal moves fail clearly and do not silently corrupt the match result.

## Dependencies

- Depends on [US-201: Reset Game State](US-201-reset-game-state.md).
- Depends on [US-202: Detect Wins](US-202-detect-wins.md).
- Benefits from [US-301: Run MCTS for a Move](US-301-run-mcts-for-a-move.md), but can start with stub or random agents.
