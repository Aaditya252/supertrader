# Supertrader MLOps pipeline

The pipeline turns the existing baseline training code into a repeatable candidate workflow. It never treats a training run as an approved live model.

## Run it

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m backend.cli mlops run
```

To skip optional MLflow tracking:

```powershell
.\.venv\Scripts\python.exe -m backend.cli mlops run --no-mlflow
```

Inspect the most recent run:

```powershell
.\.venv\Scripts\python.exe -m backend.cli mlops status
```

## GitHub Actions

The workflow at `.github/workflows/mlops.yml` runs manually, every Monday at 04:00 UTC, and on pushes that change the training pipeline, model, feature, backtest, data-download, or configuration code. It downloads the current public Binance 1-hour candle history, runs the same promotion-gated pipeline, and publishes the run record, candidate, MLflow files, and report as a 90-day `mlops-state` Actions artifact. Later runs on the same branch restore the last successful state so an approved champion can be evaluated as a challenger.

After pushing this workflow to the repository's default branch, open **Actions → MLOps candidate pipeline → Run workflow** for a manual run. The run summary reports whether a candidate passed promotion gates; a rejected candidate leaves the champion unchanged. The workflow does not deploy artifacts to the local website or place orders.

The API exposes the same run summary at `GET /mlops/status`. The existing service scheduler now launches this pipeline after `retrain_days` (seven days by default), in a separate process from live inference.

## Stages

1. **Snapshot and preflight:** copies the source Parquet candles into a run-specific immutable snapshot and verifies schema, timestamps, OHLC bounds, freshness, feature warmup, label coverage and hashes.
2. **Train candidate:** calls the existing calibrated LightGBM baseline trainer using only that snapshot. Chronological walk-forward folds and a purged holdout include configured trading costs.
3. **Quality gate:** checks the trainer's trade-count, net-return, baseline Sharpe, drawdown and walk-forward gates. When a champion exists, it also checks compatibility and evaluates the challenger and champion on the same newer holdout.
4. **Register/promote:** records every completed run. Failed candidates remain under `artifacts/mlops/runs/`; they do not change the active model pointers. Only a passing candidate is copied to the serving artifact registry and set as champion.
5. **Monitor:** the runtime `/health` endpoint reports the currently served model, matured-outcome calibration and feature drift. Monitoring is evidence for review; distribution drift alone does not automatically mean a model has lost its edge.

## Outputs

- `artifacts/mlops/runs/<run-id>/pipeline.json` records every stage, dataset/config hashes, gate results and candidate paths.
- `artifacts/mlops/runs/<run-id>/data_snapshot/` retains the exact input candles used for training.
- `artifacts/mlops/registry.json` is the local run registry; `latest-run.json` points to its latest record.
- `artifacts/mlops/mlflow.sqlite` tracks training metrics and report/model artifacts when MLflow is installed and enabled.
- A promoted model is copied into `artifacts/<model-version>/`; only successful promotion updates `champion.json` and `latest.json`.

The scheduled retraining interval comes from `retrain_days` in `configs/default.yaml`. Candidate training does not enable real orders. If validation fails, the report records the failed gates and the running champion remains in place; without an approved champion the signal layer stays research-only/HOLD.
