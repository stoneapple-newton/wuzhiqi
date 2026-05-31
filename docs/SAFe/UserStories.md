# User Stories

## US-201: Reset Game State

- **As a** developer,
- **I want** a Gomoku environment that resets to an empty board,
- **So that** tests and self-play can start from a known state.

### Acceptance Criteria

- Reset returns an empty board.
- Black is the first player.
- Legal moves include every board intersection.

## US-202: Detect Wins

- **As an** ML researcher,
- **I want** terminal wins detected exactly,
- **So that** self-play rewards are correct.

### Acceptance Criteria

- Horizontal, vertical, and both diagonal wins are detected.
- Wins at board edges are detected.
- Non-winning lines shorter than five are not terminal.

## US-301: Run MCTS for a Move

- **As an** agent developer,
- **I want** MCTS to return a visit-count policy,
- **So that** self-play can produce policy targets.

### Acceptance Criteria

- Illegal moves receive zero probability.
- Visit counts sum to the configured simulation count after search.
- Terminal states return no expanded children.

## US-401: Compare Two Agents

- **As an** experiment owner,
- **I want** a repeatable evaluation match runner,
- **So that** I can detect regressions between checkpoints.

### Acceptance Criteria

- Agents alternate first-player color.
- Evaluation accepts a random seed.
- Results include wins, losses, draws, and average game length.
