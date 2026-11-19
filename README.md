# Supertrader

See [the detailed prototype description](docs/prototype-description.md) for the product walkthrough, model explanations, verified scope and roadmap. The **Strategy finder** dashboard tab now discovers public GitHub repositories/READMEs and arXiv abstracts, provides source/license and rule-review details, saves a browser-local watchlist, and compares the existing local research experiment with its baselines. Web discoveries remain unvalidated and are never executed or automatically promoted.

A SOL/USDT research terminal with historical data, causal features, calibrated supervised predictions, cost-aware validation, risk gates, an audit/outcome ledger and paper trading. The default is **signals only**. A trained model is not automatically an approved strategy.

See `docs/architecture.md` for the design and `docs/implementation-status.md` for the actual scope and validation evidence. Models may fail to beat baselines, in which case the dashboard deliberately shows HOLD/research-only rather than invented profitable signals.

## MLOps with GitHub Actions

The promotion-gated candidate pipeline can be run locally with `python -m backend.cli mlops run` or from **Actions → MLOps candidate pipeline → Run workflow**. GitHub Actions downloads the public Binance candles, restores the latest successful run state, records a candidate and report, and preserves the champion only when the validation gates pass. See [`docs/mlops-pipeline.md`](docs/mlops-pipeline.md) for the workflow, outputs, and validation scope. The workflow does not place orders or deploy a candidate to a live service.

## Native Windows setup

Use Python 3.11–3.13 and Node 22. From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,research,rl]"
Set-Location frontend
npm.cmd ci
npm.cmd run build
Set-Location ..
```

The supplied local installation may use `--system-site-packages` to reuse the machine's PyTorch installation; the commands above create a standalone environment for a new machine. Optional research libraries are not necessary to open the dashboard; they are needed for sequence/RL/HMM/GARCH experiments. `news` is a separate extra, and sentiment requires a pinned Hugging Face commit in configuration.

Download all requested timeframes efficiently using checksum-verified monthly one-minute archives plus current-month REST backfill:

```powershell
.\.venv\Scripts\python.exe -m backend.cli download-archives
```

Alternatively, download directly through native REST or ccxt. With no `--start`, it begins before SOL's listing and discovers the first available bar:

```powershell
.\.venv\Scripts\python.exe -m backend.cli download --timeframes 1h 4h 1d
.\.venv\Scripts\python.exe -m backend.cli download --ccxt --timeframes 1m 5m 15m 1h 4h 1d
```

Commands are resumable. Current unfinished bars are excluded. Maintenance-truncated bars create gaps; malformed timestamps are quarantined. Monthly Parquet partitions and SHA-256 archive receipts are retained. Historical close-time availability is an explicit latency assumption, not a claim that today's downloaded bars were received in real time years ago.

Train the baseline model and obtain a real report:

```powershell
.\.venv\Scripts\python.exe -m backend.cli train
# Optional experiment tracking; point this to your local MLflow server/store.
$env:MLFLOW_TRACKING_URI = 'sqlite:///data/mlflow.sqlite'
.\.venv\Scripts\python.exe -m backend.cli train --mlflow
```

Reports are in `artifacts/<version>/report.json`. `latest.json` points to the latest experiment. `champion.json` is created only after configured gates pass. Reusing the same holdout after changing parameters does not produce another untouched test. Treat such runs as exploratory and reserve a new future interval for prospective confirmation.

Start the backend and frontend in separate terminals:

```powershell
.\.venv\Scripts\python.exe -m backend.cli serve
# Separate terminal:
Set-Location frontend
npm.cmd run start
```

Or run `./start.ps1` after building the frontend. It launches hidden background processes and records their IDs and logs under `data/logs`. Dashboard: http://localhost:3000; API docs: http://localhost:8000/docs. The startup helper does not send orders. Shut down the recorded processes when finished. Do not expose the API publicly without authentication, TLS, rate limits and an operational review.

## Research commands

```powershell
.\.venv\Scripts\python.exe -m backend.cli train-sequence --epochs 5
.\.venv\Scripts\python.exe -m backend.cli train-rl --timesteps 10000 --seeds 42 43 44
.\.venv\Scripts\python.exe -m backend.cli train-rl --recurrent --timesteps 10000 --seeds 42 43 44
```

The TCN outputs direction logits and return quantiles. Its uncalibrated predictions remain excluded from live signals. PPO/SAC experiments use continuous target exposure under spot long/flat semantics; negative actions mean flat. RecurrentPPO is optional. Experimental artifacts do not get promoted automatically. This release does not claim a successfully validated deep ensemble, shorting strategy or profitable RL agent.

## Paper trading and risk

Edit `configs/default.yaml` to set `mode: paper`, then restart the backend. Paper equity starts at 10,000 USDT unless changed before the first snapshot. The simulator applies an eligible signal at the next bar open and uses conservative stop-first handling when both barriers touch. This is a closed-bar simulation, not a tick-level exchange matching engine.

Risk gates include maximum exposure, risk-per-trade, volatility targeting, stale feeds, invalid books, extreme spreads, daily losses and drawdown. Stops can slip through gaps. Fractional Kelly is available as a sizing function only when validated return moments are supplied; the default does not invent those estimates. HOLD preserves an existing position unless a separate stop/kill-switch acts. Changing initial equity after paper events exist requires a new database, not silently rewriting history.

Live execution is intentionally locked. No order-placement code runs, no private API keys are requested, and setting live flags raises an error. A separately reviewed execution adapter, reconciliation and explicit activation are required before real trading. This is a research/paper release, not a completed live-money trading product.

## How learning works

Signals store causal inputs, calibrated probabilities, features, model version, reasons and risk overrides. Resolved prediction horizons create separate append-only outcomes. The health screen shows accuracy, Brier score, regime error patterns and feature PSI. It includes abstentions to avoid scoring only trades the model chose to take.

Weekly candidate training runs as a subprocess. A candidate must pass the research gates before replacing the champion. Rolling calibration degradation triggers compatible rollback or persistent suspension. These checks require enough matured outcomes; they cannot establish that a newly started application has already learned. Human notes can be submitted to `/feedback` with `X-Admin-Token` matching `SUPERTRADER_ADMIN_TOKEN`; notes are review material, never automatic market labels.

## Optional data

RSS ingestion works without a sentiment model. Sentiment remains null until `news_enabled: true` and a pinned `news_revision` are configured with the `news` extra installed. The default OHLCV model does not use news or L2; retrospective news fetches are not inserted into old training history. Derivatives collection is behind `derivatives_enabled`; open interest uses REST polling, funding and observed liquidations use futures streams. Liquidation snapshots are incomplete and are identified as such.

Optional on-chain/macro coverage, paid historical news/L2 archives, and a validated enriched feature model require additional adapters/data. Flags are reserved but unsupported inputs are not fabricated. FinBERT sentiment is a financial-language measure, not a verified Solana price forecast.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-local
.\.venv\Scripts\python.exe -m ruff check backend tests
.\.venv\Scripts\python.exe -m mypy backend
Set-Location frontend
npm.cmd run typecheck
npm.cmd run build
```

Tests cover future perturbation invariance, point-in-time news joins, label purging, gap warmup, final-bar exclusion, order-book sequencing, next-open fills, stop gaps, cost accounting, stale signals, risk caps, ledger idempotency, API protection and RL environment accounting. Tests use synthetic fixtures explicitly and never present them as market history.

## Docker

Docker Desktop is optional. Set strong database/admin secrets in the environment, then:

```powershell
$env:POSTGRES_PASSWORD = 'set-a-strong-local-password'
$env:SUPERTRADER_ADMIN_TOKEN = 'set-a-long-random-token'
docker compose -f infra/compose.yaml up --build
```

Compose uses PostgreSQL for operational audit events and shared Parquet/artifact volumes. Redis is provisioned for future distributed workers; the current single-process asyncio service does not depend on it. Retraining is isolated in a subprocess. Run one API worker to avoid duplicate ingestion; distributed scheduling/locks are not implemented in this release. Back up `data` and `artifacts` together. Docker deployment needs independent verification if Docker is unavailable on the host.

Lightweight Charts attribution is displayed in the chart. Financial models and backtests are research evidence, not financial advice or promises of future returns.
