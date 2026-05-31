# AGENTS.md

## Project

This repository is for an AlphaZero-style Gomoku (wuzhiqi, five-in-a-row) AI built in Python. The target architecture is:

- Gomoku rules and environment logic.
- Neural policy-value network.
- PUCT Monte Carlo Tree Search.
- Self-play data generation.
- Replay buffer and training loop.
- Evaluation, checkpointing, and experiment tracking.

Use `docs/raw/overall-structure.md` as the original project roadmap. Use the structured docs under `docs/` as the working source of truth.

## Working Rules

- Prefer small, testable changes.
- Keep code and docs synchronized when behavior or interfaces change.
- Do not commit generated training data, model checkpoints, or local experiment logs.
- Preserve user changes in the worktree. Do not revert unrelated edits.
- Use deterministic tests for game rules, win detection, state encoding, and MCTS math.

## Repository Map

- `src/wuzhiqi/`: application and research code.
- `tests/`: pytest test suite.
- `config/`: reproducible configuration files.
- `docs/`: project wiki and planning artifacts.
- `docs/raw/`: raw research or planning inputs.
- `data/`: local datasets and self-play samples, ignored by git.
- `checkpoints/`: local model checkpoints, ignored by git.
- `experiments/`: experiment notes and lightweight metadata.

## Commands

Use these commands from the repository root:

```powershell
uv run python main.py
uv run pytest
```

If dependencies are not installed yet, run:

```powershell
uv sync
```

## Coding Conventions

- Target Python `>=3.13`, as declared in `pyproject.toml`.
- Keep core game logic dependency-light and easy to test.
- Put package code under `src/wuzhiqi/`.
- Keep neural network and distributed training dependencies isolated from pure game logic where practical.
- Prefer typed dataclasses and clear interfaces for game state, moves, MCTS nodes, and training samples.

## Documentation Conventions

- Each major concept gets a focused Markdown page in `docs/`.
- Use concise headings and bullets.
- Link related pages rather than duplicating long explanations.
- SAFe planning artifacts live under `docs/SAFe/`.

## Definition of Done

A change is done when:

- Relevant tests pass.
- Public interfaces are documented or self-explanatory.
- User-facing docs are updated when behavior changes.
- New generated artifacts are either ignored or intentionally tracked.
