# US-201: Reset Game State

## User Story

- **As a** developer,
- **I want** a Gomoku environment that resets to an empty board,
- **So that** tests and self-play can start from a known state.

## Feature and Epic

- **Feature:** FTR-201: Game Engine
- **Epic:** EPI-002: Implement Gomoku Environment
- **PI:** PI-1: Foundations

## Scope

This story establishes the baseline environment lifecycle. A caller must be able to create or reset a game state and receive a deterministic, empty board with the correct player to move and a complete legal-move set.

In scope:

- Empty board initialization for the configured board size.
- Current-player initialization with black moving first.
- Legal move generation for every empty intersection.
- Reset behavior that clears prior moves, terminal flags, and winner state.

Out of scope:

- Win detection after moves.
- Neural network state encoding.
- MCTS integration.
- Rendering or UI concerns.

## Acceptance Criteria

- Reset returns a board with no stones placed.
- Black is always the first player after reset.
- Legal moves include every board intersection on an empty board.
- Reset after a partially played game removes all prior moves.
- Reset returns a non-terminal state with no winner.
- The result is deterministic for the same board-size configuration.

## Implementation Notes

- Prefer a small, typed game-state model under `src/wuzhiqi/` that can be reused by rules, encoders, and MCTS.
- Represent players and empty intersections with explicit constants or enums rather than ambiguous magic values.
- Keep reset side effects narrow: it should only restore game state, not seed randomness, load models, or initialize training components.
- Legal move order should be stable so tests and downstream policy indexing are reproducible.

## Test Guidance

- Add deterministic pytest coverage for default board reset.
- Test a non-default board size if the environment supports configurable sizes.
- Apply at least one move, call reset, and verify the board, player, legal moves, terminal state, and winner are restored.
- Verify legal move count equals `board_size * board_size`.

## Dependencies

- None. This should be one of the first game-engine stories implemented.
