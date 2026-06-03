# Training Status Checks

This page documents how to read the current single-process AlphaZero-style training logs, with emphasis on 15x15 Gomoku runs.

## Quick Status Read

For each recent log window, record:

- **Current batch:** latest `batch i`.
- **Episode length range and average:** from `episode_len`.
- **KL range:** from `kl`.
- **Learning-rate multiplier:** from `lr_multiplier`.
- **Effective learning rate:** from `effective_lr`.
- **Loss ranges:** from `loss`, `policy_loss`, and `value_loss`.
- **Entropy range:** from `entropy`.
- **Replay target balance:** from `target_counts`.
- **Explained variance:** from `explained_var_old` and `explained_var_new`.
- **Evaluation result:** most recent `opponent`, wins, losses, ties, and promoted checkpoint status.

For example, this log window:

```text
batch i:709-716
episode_len:53-88
kl:0.03509-0.10112
lr_multiplier:0.088
loss:3.516-3.959
entropy:2.575-2.916
explained_var_old/new:0.000
```

means training is still producing plausible self-play games and the policy has not collapsed, but updates are heavily throttled and the value head is not yet explaining outcome targets.

## Where The Metrics Come From

The current implementation logs these fields from `TrainPipeline.policy_update()` in `src/wuzhiqi/train.py`.

- `batch i`: count of self-play collection iterations completed.
- `episode_len`: number of moves in the most recent self-play game before augmentation.
- `kl`: average KL divergence between the policy distribution before and after the update.
- `lr_multiplier`: adaptive multiplier applied to `training.learning_rate`.
- `effective_lr`: actual learning rate applied after the adaptive multiplier.
- `loss`: combined policy-value network loss returned by `PolicyValueNet.train_step()`.
- `policy_loss`: policy cross-entropy against MCTS visit targets.
- `value_loss`: value MSE against final outcome targets.
- `entropy`: policy entropy returned by `PolicyValueNet.train_step()`.
- `replay_buffer`: number of augmented samples currently available.
- `target_counts`: sampled value target counts for losses, draws, and wins.
- `explained_var_old`: value-head explained variance before the update.
- `explained_var_new`: value-head explained variance after the update.

Evaluation logs come from `TrainPipeline.policy_evaluate()`. Current config uses:

```yaml
board_width: 15
board_height: 15
n_in_row: 5
mcts:
  simulations: 50
training:
  batch_size: 128
  kl_target: 0.01
  check_freq: 20
evaluation:
  opponent: heuristic
  games: 4
```

## How To Interpret Logs

### Batch

Use `batch i` to measure run progress, not playing strength. With `self_play.games_per_iteration: 1`, one batch means one newly generated self-play game plus one network update once the replay buffer is large enough.

Current checkpoints are configured as:

- `checkpoints/warm_reset_training_checkpoint_15x15.pt`
- `checkpoints/warm_reset_current_policy_15x15.pt`
- `checkpoints/warm_reset_best_policy_15x15.pt`
- `checkpoints/warm_reset_best_training_checkpoint_15x15.pt`

With `check_freq: 20`, expect evaluation and checkpointing at batches `20`, `40`, `60`, `80`, and so on during the smoke reset.

### Episode Length

For 15x15 Gomoku, normal self-play lengths can vary widely. Use trends rather than single games.

Healthy signs:

- Varied lengths rather than the same length repeatedly.
- No frequent games ending in just a handful of moves.
- No constant near-full-board games unless draws are expected.

Investigate if:

- Most games end extremely early.
- Most games run close to 225 moves.
- A sudden shift appears after a code or config change.

Likely causes include win-detection bugs, illegal move handling, temperature settings, MCTS selection errors, or an opponent/player perspective mismatch.

### KL

`kl` measures how much the network policy changed during the update. The configured warm-reset target is `training.kl_target: 0.01`.

Current adaptive rule:

- If `kl > kl_target * 4`, the epoch loop stops early.
- If `kl > kl_target * 2` and `lr_multiplier > 0.1`, the multiplier is divided by `1.5`.
- If `kl < kl_target / 2` and `lr_multiplier < 10`, the multiplier is multiplied by `1.5`.

Interpretation for `kl_target: 0.01`:

- `< 0.005`: updates may be too small; LR multiplier may rise.
- `0.005-0.02`: generally controlled.
- `0.02-0.04`: aggressive; watch for repeated spikes.
- `> 0.04`: large policy movement; repeated values usually mean training instability.

If `lr_multiplier` is below `0.1`, the current code will not reduce it further because of the `> 0.1` guard. A run sitting near `0.088` is already heavily throttled.

### Loss

Loss combines policy imitation of MCTS visit probabilities, value prediction error, and regularization. It is useful as a stability signal, but it is not enough to judge playing strength.

Healthy signs:

- Loss does not explode or become `nan`.
- Loss trends downward over longer windows.
- Loss changes are consistent with KL and entropy.

Investigate if:

- Loss suddenly doubles or becomes `nan`.
- Loss falls while evaluation strength gets worse.
- Loss is flat for many checkpoints while evaluation never improves.

### Entropy

Entropy summarizes how spread out the policy distribution is.

Healthy signs:

- Early training has high entropy because the policy is exploratory.
- Entropy gradually decreases as tactical patterns become clearer.
- Entropy does not collapse abruptly to near zero.

Investigate if:

- Entropy drops sharply while evaluation does not improve.
- Entropy stays nearly constant for thousands of batches.
- Entropy becomes inconsistent with episode lengths, such as very deterministic policy but erratic games.

### Explained Variance

Explained variance tracks how much better the value head predicts winner targets than a constant baseline.

Interpretation:

- Near `0.0`: value predictions are not explaining outcomes.
- Positive and rising: value head is learning useful signal.
- Negative: predictions are worse than a constant mean baseline.

If `explained_var_old` and `explained_var_new` stay at `0.000`, check:

- Winner targets are balanced and non-constant in sampled batches.
- Values are encoded from the state player's perspective.
- Draws, wins, and losses use the expected labels.
- Value output shape and flattening are correct.
- Value loss contributes gradients.
- Replay buffer is not dominated by identical or near-identical outcomes.

Do not rely on explained variance alone for small batches. The current warm-reset 15x15 config uses `batch_size: 128`, which is still diagnostic rather than definitive.

## Evaluation Checklist

Check training status at each checkpoint interval:

1. Confirm the run reached an evaluation batch such as `20` or `40`.
2. Record the opponent label, win count, loss count, tie count, and win ratio.
3. Confirm `warm_reset_current_policy_15x15.pt` was saved.
4. Confirm `warm_reset_training_checkpoint_15x15.pt` was saved.
5. If the run prints `New best policy`, confirm `warm_reset_best_policy_15x15.pt` and `warm_reset_best_training_checkpoint_15x15.pt` were saved.
6. Compare results against the previous checkpoint, not just against the raw loss.

Use evaluation games that are separate from self-play. Alternate starting player, keep seeds reproducible for regression checks, and increase `evaluation.games` when a promotion decision matters. Four games is enough for a smoke check, but not enough for a reliable strength estimate.

## Current 15x15 Triage Rules

Use these rules for the current config:

- **Continue watching:** episode lengths vary, loss is finite, entropy is nonzero, and KL spikes are isolated.
- **Inspect value pipeline:** explained variance remains `0.000` for several evaluation intervals.
- **Inspect LR/KL behavior:** `kl` repeatedly exceeds `0.04`, especially if `lr_multiplier` is already below `0.1`.
- **Inspect MCTS/search quality:** evaluation does not improve after several checkpoint intervals despite stable training loss.
- **Increase evaluation games:** any checkpoint promotion based on `4` games is still only a smoke signal.
- **Scale cautiously:** before increasing network or MCTS size, verify value targets, legal moves, winner perspective, checkpoint resume, and deterministic evaluation.

## Useful Commands

Run training:

```powershell
uv run python main.py
```

Run the full test suite:

```powershell
uv run pytest
```

Evaluate the configured checkpoint:

```powershell
uv run python -m wuzhiqi.evaluate
```

Inspect recent logs manually if the terminal captured them:

```powershell
# Replace training.log with the actual log file if stdout was redirected.
Get-Content training.log -Tail 80
```

## What To Record In Experiments

For each run, record lightweight metadata in `experiments/`:

- Date and machine.
- Git commit or working tree note.
- Config snapshot or config hash.
- Board size and win length.
- MCTS simulations.
- Batch size and replay buffer size.
- Latest batch.
- Latest training log window.
- Latest evaluation result.
- Checkpoint paths.
- Any manual interpretation or intervention.

Do not commit generated training data, model checkpoints, or large logs.

## External References

- DeepMind's AlphaGo Zero paper describes self-play reinforcement learning with MCTS-generated policy targets, final game outcome value targets, and a combined value/policy/regularization loss: <https://www.nature.com/articles/nature24270>
- OpenSpiel's AlphaZero documentation summarizes the practical pipeline as actors generating self-play data, a learner updating from a FIFO replay buffer, evaluators measuring progress against standard MCTS, and checkpoints/logs for analysis: <https://openspiel.readthedocs.io/en/latest/alpha_zero.html>
- AlphaZero search documentation describes MCTS nodes using network policy and value outputs, selecting by `Q(s,a) + U(s,a)`, and returning visit-probability targets for self-play: <https://alphazero.readthedocs.io/en/latest/search_algo.html>
- AlphaZero evaluator documentation describes the neural evaluator interface as returning a `(policy, value)` pair for MCTS: <https://alphazero.readthedocs.io/en/latest/evaluator.html>

## Related

- [Self-Play Training](SelfPlayTraining.md)
- [Training Hyperparameters](TrainingHyperparameters.md)
- [Evaluation Metrics](EvaluationMetrics.md)
- [Checkpoints](Checkpoints.md)
- [Current Training Workflow](training-workflow/README.md)
