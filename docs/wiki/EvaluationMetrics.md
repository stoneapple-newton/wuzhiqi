# Evaluation Metrics

Evaluation compares agents under controlled settings. Training loss can show whether optimization is numerically stable, but playing strength must be measured through games, checkpoint comparisons, tactical tests, and search diagnostics.

## Current Implementation

The current evaluation entry point is `wuzhiqi.evaluate.evaluate_checkpoint()`.

It loads the configured checkpoint, creates an AlphaZero-style `MCTSPlayer`, and plays against the configured baseline:

- `heuristic`: `HeuristicPlayer(search_radius=evaluation.heuristic_search_radius)`
- `pure_mcts`: `PureMCTSPlayer(n_playout=evaluation.pure_mcts_playouts)`

The result is an `EvaluationResult`:

```text
wins
losses
draws
games
win_rate = (wins + 0.5 * draws) / games
```

Current `config/default.yaml` uses a small smoke-evaluation setup:

```yaml
evaluation:
  opponent: heuristic
  heuristic_search_radius: 2
  games: 4
  seed: 20260531
  pure_mcts_playouts: 200
  model_file: checkpoints/warm_reset_current_policy_15x15.pt
```

Four games are useful for checking that evaluation runs, but they are not enough to make reliable strength or checkpoint-promotion decisions.

## Metric Groups

Use metrics in groups. No single metric is sufficient for AlphaZero-style training.

- **Playing strength:** win rate, score, Elo-style rating, checkpoint ladder.
- **Reliability:** confidence interval, number of games, color balance, seed stability.
- **Game quality:** average game length, draw rate, illegal move count, termination reason.
- **Training health:** total loss, policy loss, value loss, entropy, KL divergence, explained variance.
- **Search quality:** MCTS visit concentration, root value, policy-search agreement, simulation count.
- **Data quality:** replay buffer size, winner-target balance, unique positions, augmentation count.
- **Operational health:** self-play speed, update speed, checkpoint cadence, hardware utilization.

## Playing Strength Metrics

### Win Rate

Win rate is the main direct evaluation metric.

```text
win_rate = wins / games
```

Use this when draws are impossible or should be counted separately.

Interpretation:

- High win rate against a weak opponent is only a sanity check.
- Stable improvement against the same opponent indicates learning.
- A single evaluation window can be misleading when `games` is small.

### Score Rate

The current implementation uses score rate and names it `win_rate`:

```text
score_rate = (wins + 0.5 * draws) / games
```

This is useful when draws are meaningful. For Gomoku, draws can indicate full-board games, overly defensive play, or search/training issues depending on the position distribution.

### Loss Rate

```text
loss_rate = losses / games
```

Track loss rate separately from score rate. A model with many draws and few losses may be improving defensively, while a model with fewer draws and more wins may be learning to convert advantages.

### Draw Rate

```text
draw_rate = draws / games
```

Investigate a rising draw rate if:

- Games frequently reach the full 225 moves on 15x15.
- The agent avoids tactical commitments.
- The evaluation opponent is too weak to punish passive play.

### Checkpoint-Vs-Checkpoint Win Rate

Compare a candidate checkpoint against the previous best or a fixed historical checkpoint.

Purpose:

- Regression detection.
- Promotion decisions.
- Measuring whether current training improves over the deployed/best model.

Protocol:

- Use the same board size, MCTS simulations, temperature, and rules.
- Alternate starting player.
- Run enough games to reduce noise.
- Promote only when the confidence level is acceptable for the project stage.

### Baseline Win Rate

Use multiple baselines because each baseline catches different failures.

- **Random baseline:** verifies legality and basic tactical learning.
- **Heuristic baseline:** checks practical Gomoku tactics.
- **Pure MCTS baseline:** checks whether the policy-value network plus PUCT beats search without neural guidance.
- **Previous checkpoint:** checks training regression.
- **Fixed checkpoint ladder:** tracks long-term progress.
- **Human-play smoke tests:** useful qualitatively, not statistically reliable.

## Statistical Reliability

Evaluation is noisy because self-play agents, MCTS tie-breaking, and starting colors can change outcomes.

### Number Of Games

Use game count based on the decision:

- `2-4` games: smoke test only.
- `20-50` games: rough progress check.
- `100+` games: stronger checkpoint comparison.
- `400+` games: more reliable rating or promotion decisions.

For expensive 15x15 evaluations, start small during development and increase games before promoting a checkpoint.

### Confidence Interval

For a rough score-rate uncertainty estimate:

```text
standard_error ~= sqrt(score_rate * (1 - score_rate) / games)
95_percent_interval ~= score_rate +/- 1.96 * standard_error
```

This approximation is imperfect when draws are common, but it is useful for intuition. With only `4` games, uncertainty is so large that the result should not be treated as a strength estimate.

### Color Balance

Alternate starting player evenly. In the current implementation, evaluation uses:

```python
start_player = i % 2
```

Report first-player and second-player results separately when possible. A large color gap can indicate first-move advantage, perspective encoding bugs, or asymmetric search behavior.

### Seed Stability

Use fixed seeds for regression tests. Also run occasional different-seed evaluations to ensure the model is not only performing well under one deterministic sequence.

## Game Quality Metrics

### Average Game Length

```text
avg_game_length = total_moves / games
```

Meaning:

- Very short games can indicate tactical dominance, but also rule or search bugs.
- Very long games can indicate weak attack conversion, draw-heavy play, or overly diffuse policy.
- Sudden shifts after code changes require investigation.

For 15x15 Gomoku, inspect the distribution, not only the average:

- Minimum length.
- Maximum length.
- Median length.
- Percent of games ending before 20 moves.
- Percent of games ending near full board.

### Termination Reason

Record whether games ended by:

- Five-in-a-row win.
- Full-board draw.
- Illegal move or exception.
- Manual interruption.

Illegal moves and exceptions should be zero. Any nonzero count is a correctness issue, not a weak-agent issue.

### Winner Distribution

Track wins by player id and starting color.

Useful for detecting:

- First-player advantage.
- Current-player perspective bugs.
- Value target sign errors.
- Evaluation opponent assignment mistakes.

## Training Health Metrics

These metrics come from training updates, not standalone evaluation games. They are still part of training status because they help explain why evaluation is improving or failing.

### Total Loss

Current network training computes:

```text
loss = value_mse + policy_cross_entropy
```

Meaning:

- Lower loss usually means the network better matches replay-buffer targets.
- Lower loss does not guarantee stronger play.
- Sudden spikes, `nan`, or flat loss across many intervals require investigation.

Current code logs total loss, `policy_loss`, and `value_loss` separately.

### Policy Loss

Policy loss measures how well the raw network policy matches MCTS visit-count targets.

Expected target:

```text
policy target = normalized MCTS visit distribution
```

High policy loss can mean:

- Search targets are sharp and the network has not caught up.
- MCTS simulations are too low or noisy.
- Learning rate is unstable.
- Replay data is stale or inconsistent.

### Value Loss

Value loss measures how well the value head predicts final game outcome from the current player's perspective.

Expected target:

```text
win:  1
draw: 0
loss:-1
```

High or stagnant value loss can mean:

- Winner labels are wrong.
- Perspective encoding is wrong.
- Replay buffer has low outcome diversity.
- The value head is undertrained relative to the policy head.

### Explained Variance

Explained variance estimates whether value predictions are better than predicting a constant target mean.

```text
explained_variance = 1 - var(target - prediction) / var(target)
```

Interpretation:

- `1.0`: perfect value prediction.
- `0.0`: no better than predicting the target mean.
- `< 0.0`: worse than the target mean.

If explained variance stays at `0.000`, inspect winner target distribution, value output shape, perspective encoding, and whether value gradients are flowing.

### Entropy

Entropy measures policy spread.

Meaning:

- Higher entropy means more diffuse move probabilities.
- Lower entropy means more concentrated policy.
- Early training usually has high entropy.
- Abrupt collapse can indicate overconfidence or a bad update.

Entropy should be interpreted with board state. A forced tactical move should have low entropy; an open early-board position should usually have higher entropy.

### KL Divergence

KL divergence measures how much the policy changed from before to after an update.

The current adaptive learning-rate logic uses `training.kl_target`.

For the current warm-reset `kl_target: 0.01`:

- `< 0.005`: conservative update.
- `0.005-0.02`: near target range.
- `0.02-0.04`: aggressive update.
- `> 0.04`: large update; repeated spikes are a concern.

KL is a stability metric, not a strength metric.

### Learning-Rate Multiplier

`lr_multiplier` scales the base learning rate.

Meaning:

- High multiplier: the update loop is trying to make larger changes.
- Low multiplier: the update loop has throttled learning because KL was high.
- Very low values can slow progress even if training remains stable.

If `lr_multiplier` is low and KL remains high, inspect target quality, replay mix, and base learning rate.

## Search Quality Metrics

These are target metrics for future instrumentation. They are useful because AlphaZero training depends on MCTS improving the raw network policy.

### Root Visit Distribution

Track concentration of MCTS visits at the root:

- Top-1 visit share.
- Top-3 visit share.
- Visit entropy.

Meaning:

- Very flat visits can indicate uncertain search or too few simulations.
- Very sharp visits can indicate clear tactics or premature policy collapse.

### Policy-Search Agreement

Compare the raw network policy with the MCTS visit distribution.

Useful measurements:

- KL between MCTS visits and raw policy.
- Top-1 agreement.
- Whether MCTS changes the network's preferred move.

Meaning:

- If MCTS often changes the move and evaluation improves, search is adding value.
- If MCTS never changes the move early in training, search may be too weak or too policy-dominated.

### Root Value

Track the value estimate at the root before move selection.

Useful for:

- Seeing whether the value head is calibrated.
- Comparing value prediction against final outcome.
- Detecting systematic overconfidence.

### Search Budget Sensitivity

Evaluate the same checkpoint with different MCTS simulation counts:

- `50`
- `100`
- `200`
- `400+`

If strength improves sharply with more simulations, the network may be useful but search-limited. If strength does not improve, the policy/value estimates may be poor or the opponent may be too weak to reveal differences.

## Data Quality Metrics

These metrics describe the replay data used for training.

### Replay Buffer Size

Track current replay buffer length and maximum capacity.

Meaning:

- Too small: high variance and overfitting to recent games.
- Too large: stale data may slow adaptation.

### Winner Target Balance

Track counts of `1`, `0`, and `-1` value targets in sampled batches and in the buffer.

Meaning:

- All one target: explained variance becomes meaningless and value learning can stall.
- Severe imbalance: value head may learn a biased constant.

### Unique Position Count

Track duplicate positions when possible.

High duplication can happen if:

- MCTS is deterministic too early.
- Temperature is too low.
- Replay buffer is too small.
- Self-play games are too similar.

### Augmentation Ratio

Track generated samples before and after symmetry augmentation.

Expected Gomoku augmentations usually include rotations and reflections. Confirm the policy vector is transformed consistently with the board state.

## Operational Metrics

These metrics show whether training is using time and hardware effectively.

### Self-Play Throughput

```text
self_play_games_per_hour
self_play_positions_per_hour
average_seconds_per_game
```

Use this to compare MCTS settings and board sizes.

### Training Throughput

```text
updates_per_hour
samples_per_second
seconds_per_policy_update
```

Use this to detect slow data loading, GPU underuse, or overly expensive model updates.

### Evaluation Cost

```text
seconds_per_eval_game
total_eval_seconds
```

Evaluation can dominate runtime as game count and MCTS simulations increase.

### Checkpoint Cadence

Track:

- Last saved checkpoint.
- Last best checkpoint.
- Batch of last evaluation.
- Time since last checkpoint.

This helps resume long 15x15 runs safely.

## Recommended Evaluation Protocols

### Smoke Evaluation

Purpose: verify the code runs after a change.

- Games: `2-4`
- Opponent: heuristic or small pure MCTS
- Required result: no exceptions, legal moves only, finite metrics.

### Regression Evaluation

Purpose: verify a candidate did not get worse.

- Games: `20-50`
- Opponent: previous checkpoint and fixed heuristic baseline.
- Start players: alternate evenly.
- Required result: no clear regression in score rate or game quality.

### Promotion Evaluation

Purpose: decide whether to update `best_policy`.

- Games: `100+` if compute allows.
- Opponents: previous best, fixed checkpoint ladder, heuristic, pure MCTS.
- Report: score rate, confidence interval, draw rate, average game length, color split.

### Scaling Evaluation

Purpose: understand whether the model benefits from more search.

- Evaluate the same checkpoint at several MCTS simulation counts.
- Keep opponents and seeds controlled.
- Compare strength gain versus runtime cost.

## Minimum Evaluation Report

Every meaningful checkpoint evaluation should report:

- Checkpoint path.
- Config path or config hash.
- Board size and `n_in_row`.
- Opponent type and opponent parameters.
- MCTS simulations for the evaluated agent.
- Number of games.
- Wins, losses, draws.
- Score rate.
- Starting color split.
- Average game length.
- Seed.
- Runtime.
- Notes on illegal moves, exceptions, or abnormal terminations.

## Current Gaps To Implement

The current code already reports wins, losses, draws, games, and score rate. The next useful additions are:

- Average game length during evaluation.
- Color-split results.
- Replay-buffer target distribution.
- Confidence interval for score rate.
- Evaluation runtime.
- JSON or CSV evaluation logs under `experiments/`.
- Checkpoint-vs-checkpoint evaluation.
- Fixed tactical position test suite.
- Search diagnostics: root visit entropy, top move visit share, policy-search KL.

## External Context

AlphaZero-style systems are evaluated through the interaction of self-play data generation, neural policy-value learning, MCTS-guided play, and checkpoint comparison. OpenSpiel's AlphaZero documentation describes this structure as actors generating self-play games, a learner updating from a replay buffer, evaluators measuring progress against MCTS opponents, and checkpoint/log outputs for analysis. The AlphaGo Zero paper describes training the network to match MCTS search probabilities and final game outcomes, which is why policy, value, and playing-strength metrics all matter.

## Related

- [Training Status Checks](TrainingStatusChecks.md)
- [Self-Play Training](SelfPlayTraining.md)
- [Training Hyperparameters](TrainingHyperparameters.md)
- [Checkpoints](Checkpoints.md)
- [Reproducibility](Reproducibility.md)
