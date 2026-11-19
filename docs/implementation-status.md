# Implementation scope

This is a functional research-and-paper platform. It is not certified production-ready or proven profitable.

## Implemented

- Resumable SOL spot history downloads through native Binance REST and optional ccxt, archive checksums, monthly Parquet partitions, manifests, quarantine, gap repair, complete-bucket resampling across all six requested intervals.
- Sequenced live spot order books/trades with reconnect and snapshot rebuilding; optional futures funding/liquidation streams and REST open-interest polling.
- Causal OHLCV feature groups, optional statistical transforms, triple-barrier labels, chronological purging, calibrated classifier, return-quantile regressors and cost-aware baseline comparisons.
- Experimental causal HMM, PyTorch TCN, Gymnasium PPO/SAC/RecurrentPPO scripts. Unvalidated experiments are excluded from serving.
- Spot backtest and paper simulators, exposure/risk gates, conservative stops, immutable signal/outcome/feedback events, drift and calibration monitoring, weekly candidate retraining and rollback/suspension.
- Promotion-gated MLOps candidate pipeline with immutable input snapshots, local run registry, optional MLflow tracking, and the service's scheduled retraining delegated to a separate subprocess.
- FastAPI read endpoints, WebSocket updates, protected feedback, Next.js trading terminal, explicit empty/stale/error states, model/portfolio/backtest pages, setup instructions, CI and Docker definitions.

## Remaining before a production or live-money claim

- Prospective paper validation, a predeclared research-trial registry and untouched future evaluation after any tuning; strong gates are intentionally not proof of universal profitability.
- Full independent benchmark/ablation campaigns for sequence/RL/HMM/GARCH, out-of-fold deep-model calibration and meta-ensemble promotion. The active research model uses OHLCV only.
- Historical point-in-time news/L2 archives and optional macro/on-chain adapters. FinBERT is optional and needs a pinned revision and domain validation.
- A separately reviewed live execution adapter, reconciliation, credential handling and explicit user activation. Current live mode is locked.
- Multiworker distributed locking, adversarial security/load tests, backup/restore drills, deployment observability and operational runbooks. Docker must be verified on a host with Docker available.
- PBO needs a sufficient comparable strategy search grid; no fake PBO statistic is returned. Deflated Sharpe is marked approximate, with block-bootstrap and cost stress reports.

## Build evidence

The local test/build and actual model/data evidence are recorded in `docs/verification.json` when verification finishes. No test count or profitable result is invented in advance.
