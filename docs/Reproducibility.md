# Reproducibility

Reproducible runs require stable code, config, seeds, and environment metadata.

## Required Metadata

- Git commit or dirty worktree status.
- Python version.
- Dependency lockfile when available.
- Config file and overrides.
- Random seeds.
- Hardware summary for training runs.

## Practices

- Keep default configs under `config/`.
- Record experiment decisions in [Experiments](Experiments.md).
- Do not rely on unchecked local notebook state for canonical results.
- Keep deterministic unit tests for rules and state encoding.

## Related

- [Datasets](Datasets.md)
- [Experiments](Experiments.md)
