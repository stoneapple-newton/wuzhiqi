# Features

## FTR-101: Repository Scaffold

- **Epic:** EPI-001
- **Description:** Base folders, documentation pages, and package placeholders exist.

## FTR-201: Game Engine

- **Epic:** EPI-002
- **Description:** Implement board state, move application, win detection, draw detection, and legal move generation.

## FTR-202: State Encoder

- **Epic:** EPI-002
- **Description:** Convert game state to deterministic tensor inputs and legal action masks.

## FTR-301: Policy-Value Network

- **Epic:** EPI-003
- **Description:** Implement a PyTorch model that returns policy logits and value estimates.

## FTR-302: PUCT MCTS

- **Epic:** EPI-003
- **Description:** Implement tree search using network priors and value estimates.

## FTR-303: Self-Play Pipeline

- **Epic:** EPI-003
- **Description:** Generate training samples from MCTS-guided self-play games.

## FTR-401: Evaluation Harness

- **Epic:** EPI-004
- **Description:** Run checkpoint-vs-baseline and checkpoint-vs-checkpoint matches.
