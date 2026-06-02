# US-202: Detect Wins

## User Story

- **As an** ML researcher,
- **I want** terminal wins detected exactly,
- **So that** self-play rewards are correct.

## Feature and Epic

- **Feature:** FTR-201: Game Engine
- **Epic:** EPI-002: Implement Gomoku Environment
- **PI:** PI-1: Foundations

## Scope

This story defines terminal win detection for standard Gomoku lines. The environment must identify exactly when a player has five connected stones in a row and must avoid declaring terminal states for shorter or interrupted lines.

In scope:

- Horizontal wins.
- Vertical wins.
- Main diagonal wins.
- Anti-diagonal wins.
- Edge and corner wins.
- Non-winning states with fewer than five connected stones.

Out of scope:

- Rule variants such as Renju forbidden moves, overline restrictions, or swap rules.
- Neural network value assignment beyond exposing terminal winner information.
- MCTS backup logic.

## Acceptance Criteria

- Horizontal, vertical, and both diagonal wins are detected.
- Wins at board edges and corners are detected.
- Non-winning lines shorter than five are not terminal.
- Interrupted lines with mixed players are not terminal.
- The winning player is reported unambiguously.
- Legal moves are empty or unavailable once the state is terminal.
- Draw detection remains separate from win detection and does not mask a win on the final move.

## Implementation Notes

- Prefer checking from the last move when available, because it keeps move application efficient.
- Keep a board-wide scan helper only if it is useful for tests, validation, or loading arbitrary positions.
- Use direction pairs for win checks: horizontal, vertical, main diagonal, and anti-diagonal.
- Avoid hard-coding a 15x15 board if the environment supports configurable board sizes.
- Keep terminal state, winner, and legal moves consistent immediately after a winning move is applied.

## Test Guidance

- Cover each direction with exactly five stones.
- Cover each direction at an edge or corner.
- Cover four-in-a-row and separated line cases that must not win.
- Cover a final-board win to ensure it is reported as a win, not a draw.
- Include both black and white wins.

## Dependencies

- Depends on [US-201: Reset Game State](US-201-reset-game-state.md) for a reliable game-state baseline.
