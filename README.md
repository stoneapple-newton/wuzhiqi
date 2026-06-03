# Wuzhiqi

AlphaZero-style Gomoku implementation based on the `tempalte/AlphaZero_Gomoku`
reference project.

The local smoke baseline uses a 6 x 6 board with 4 in a row and a PyTorch
policy-value network. The code remains configurable for full 15 x 15
five-in-a-row Gomoku.

See [AGENTS.md](AGENTS.md) for agent working instructions and [docs/README.md](docs/README.md) for the project wiki.

## Local dashboard

Run the training and evaluation dashboard from the repository root:

```powershell
uv run python -m wuzhiqi.dashboard
```

Then open <http://127.0.0.1:8765>. The dashboard reads local logs and
checkpoints, can launch curated Codex CLI log analyses when `codex` is
available, and stores generated analysis/evaluation outputs under ignored
`experiments/` subfolders.
