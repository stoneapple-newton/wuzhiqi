# Checkpoints

Model checkpoints are generated artifacts and should not be committed by default.

## Location

- Store local checkpoints under `checkpoints/`.
- Keep lightweight metadata in `experiments/` when it is useful for review.

## Naming

Recommended naming:

- `ckpt_iter_000001.pt`
- `ckpt_best.pt`
- `ckpt_eval_<rating-or-date>.pt`

## Metadata

Each checkpoint should record:

- Training iteration.
- Git commit or working tree marker.
- Config file path or config hash.
- Evaluation results.

## Related

- [Evaluation Metrics](EvaluationMetrics.md)
- [Reproducibility](Reproducibility.md)
