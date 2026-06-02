# US-301: Run MCTS for a Move

## User Story

- **As an** agent developer,
- **I want** MCTS to return a visit-count policy,
- **So that** self-play can produce policy targets.

## Feature and Epic

- **Feature:** FTR-302: PUCT MCTS
- **Epic:** EPI-003: Implement AlphaZero Core
- **PI:** PI-2: AlphaZero Core

## Scope

This story establishes the minimum tree-search behavior needed for AlphaZero-style self-play. Given a non-terminal game state and an evaluator that supplies priors and values, MCTS should run a configured number of simulations and return a policy target derived from child visit counts.

In scope:

- MCTS node state, prior probability, value sum, and visit count.
- PUCT child selection.
- Expansion from legal moves only.
- Value backup through the visited path.
- Visit-count policy over the full action space.
- Terminal-state handling without child expansion.

Out of scope:

- Batched neural network inference.
- Dirichlet noise at the root.
- Temperature scheduling for move sampling.
- Parallel tree search.
- Replay buffer storage.

## Acceptance Criteria

- Illegal moves receive zero probability.
- Visit counts sum to the configured simulation count after search for a non-terminal root.
- Terminal states return no expanded children.
- The returned policy uses the project action indexing convention.
- Search is deterministic when evaluator outputs and tie-breaking are deterministic.
- Backups alternate value perspective correctly between players.

## Implementation Notes

- Keep MCTS independent from a concrete neural network class by depending on a small evaluator interface.
- Use legal-action masks from the game environment rather than recomputing move legality inside MCTS.
- Normalize or validate evaluator priors over legal moves so illegal priors cannot leak into policy targets.
- Make exploration constants and simulation count configurable.
- Store enough node statistics for debugging, tests, and later training diagnostics.

## Test Guidance

- Use a deterministic fake evaluator with fixed priors and values.
- Verify illegal actions have zero probability even when the evaluator assigns them prior mass.
- Verify visit-count totals for a small board and small simulation count.
- Verify terminal roots do not expand children.
- Include a backup-perspective test where known values produce expected root statistics.

## Dependencies

- Depends on [US-201: Reset Game State](US-201-reset-game-state.md).
- Depends on [US-202: Detect Wins](US-202-detect-wins.md) for terminal-state behavior.
- Depends on a stable action-indexing convention from the state representation work.
