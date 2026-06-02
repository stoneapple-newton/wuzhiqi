# Parallel Training Investigation

This note summarizes the current bottleneck analysis for using more CPU cores and GPU VRAM during 15x15 training.

## Current Runtime Shape

- Training currently runs in a single process through `TrainPipeline.run()`.
- Self-play games are collected sequentially in `collect_selfplay_data()`.
- Each self-play move runs PUCT MCTS with `mcts.simulations` playouts.
- Each MCTS playout eventually calls the policy-value network for one board state.
- The GPU receives many small single-position inference calls instead of larger batches.

The effective pattern is:

```text
CPU Python MCTS loop -> tiny GPU inference -> CPU Python MCTS loop -> tiny GPU inference
```

This underuses both many-core CPUs and large GPUs.

## Main Bottlenecks

- Python MCTS is serialized and CPU-heavy.
- `copy.deepcopy(state)` is called for every MCTS playout.
- Neural inference is not batched during tree search.
- Python threads are unlikely to help much because the tree-search loop is Python-heavy.
- Running multiple independent `uv run python main.py` processes against the same checkpoint paths is unsafe because they can overwrite or corrupt shared checkpoint files.

## Useful Parallel Architecture

A stronger training architecture would separate the pipeline into coordinated workers:

- Multiple self-play worker processes generate games in parallel.
- A shared GPU inference service batches policy-value requests from workers.
- A trainer process samples replay data and updates the network.
- An evaluator process runs occasional evaluation without blocking self-play.
- Checkpoint writes are centralized so workers do not race on the same files.

This is the common direction for scaling AlphaZero-style training on a machine with many CPU cores and a GPU.

## Simple Knobs Without Architecture Work

These may help, but they do not solve the main self-play bottleneck:

- Increase `training.batch_size` to use more GPU memory during training updates.
- Keep evaluation cheap with a heuristic opponent and larger `training.check_freq`.
- Reduce `mcts.simulations` for faster but lower-quality self-play.

Increasing `self_play.games_per_iteration` does not add parallelism in the current implementation. It only runs more games sequentially before the next training update.

## Recommendation

The next meaningful performance investigation should focus on multiprocessing self-play plus batched GPU inference. That change is larger than a config tweak, but it targets the actual underused hardware: CPU cores for search and GPU VRAM/compute for batched network calls.
