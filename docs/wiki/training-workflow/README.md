# Current Training Workflow

This folder tracks how the current single-process training loop works and where future scaling work can be investigated.

## Pages

- [Single Process Pipeline](SingleProcessPipeline.md)
- [Training Status Checks](../TrainingStatusChecks.md)

## Current Command

Run training from the repository root:

```powershell
uv run python main.py
```

`main.py` calls `wuzhiqi.train.main()`, which creates `TrainPipeline` with `config/default.yaml`.
