# Supertrader: prototype description, models and feature roadmap

Prepared 24 October 2026 from the local source code and saved experiment reports. Historical evidence is dated below; it is not a fresh claim about the running application.

## Product description for a presentation or proposal

Supertrader is an AI-assisted quantitative research terminal for Solana markets. It brings market data, technical analysis, statistical diagnostics, machine-learning forecasts, strategy research, risk controls and simulated execution into one workspace. Its purpose is to make the entire decision process inspectable: what the system observed, what a model predicted, which risks blocked action, and what happened after the forecast matured.

The core workflow starts with public exchange data and ends with an auditable BUY, SELL or HOLD decision. Historical and streaming observations pass through quality checks and causal feature calculations. A calibrated statistical model estimates the probabilities of defined market outcomes. An independent risk layer checks market freshness, liquidity conditions, volatility and portfolio limits before the result can become actionable. Signals and later outcomes are recorded separately, supporting error analysis, model comparison and controlled retraining.

The prototype also contains a more ambitious research engine: seven temporal convolutional encoders, attention across timeframes, supervised pretraining, an offline reinforcement-learning warm start and several reinforcement-learning experiments. These components provide an experimental framework for studying allocation decisions across market conditions. Their presence does not establish a validated trading edge.

The new strategy finder expands the research workflow beyond internally designed models. Users can discover public strategy repositories and research abstracts, examine source and license information, identify gaps in the described trading rules and save candidates for further investigation. A separate benchmark panel exposes the results of the existing local experiment. Web discoveries remain explicitly unvalidated until reproduced and tested.

## What is implemented, experimental or proposed

| Capability | Status and practical meaning |
|---|---|
| Historical SOL ingestion, checksums, Parquet storage and gap checks | Implemented research infrastructure |
| Streaming trades and sequenced order books | Implemented adapters; availability depends on network and exchange access |
| Calibrated OHLCV classifier and return quantiles | Implemented training/serving pipeline; the inspected baseline failed promotion |
| Technical analysis and multi-timeframe confluence | Implemented calculations and research panels |
| Advanced quant estimators | Implemented research functions; several need optional dependencies or extra inputs and do not run in the fast path |
| TCN, attention, HMM, IQL, PPO/SAC/RecurrentPPO/QR-DQN | Research code and saved experiments; no approved deep trading model established |
| Paper trading, risk gates and outcome ledger | Implemented, with entries dependent on eligible approved signals |
| Broker terminal integrations | Removed from the supported platform; the core service uses Binance data only |
| Web strategy discovery | Added in this update: on-demand GitHub/README and arXiv abstract collection |
| Provenance, rule triage, watchlist and local comparison | Added in this update |
| Automatically translating, backtesting and promoting web strategies | Proposed next stage; not implemented by the finder |
| Real-money execution | Locked in the normal service configuration |

## How the whole system works

### 1. Acquire and validate market observations

The initial research path obtains SOL/USDT spot candles from Binance using archive downloads and REST backfill, with optional ccxt support. Archive receipts retain checksums, allowing downloaded files to be traced to their source. Candles are stored in monthly Parquet partitions rather than kept only in an application session.

The Phase 2 path adds aggregated-trade archives and tick-derived bars. Supported bar intervals span 30 seconds, 1 minute, 5 minutes, 15 minutes, 1 hour, 2 hours, 4 hours, 6 hours, 12 hours and 1 day. The neural model uses a subset of seven intervals. Exchange aggregated trades group executions; they should not be described as a complete tape of individual fills.

Validation checks timestamps, duplicates, invalid prices or volumes, missing intervals and incomplete bars. Empty periods remain gaps. An outage must not become an invented candle that appears tradable in a backtest. Feature warmup restarts after gaps where required.

For live order books, the collector combines an exchange snapshot with sequenced updates. A sequence break invalidates the book and requires rebuilding it. The resulting spread and depth can help determine whether a market observation is suitable for a signal.

### 2. Preserve the time at which information was usable

Features must use information available at the decision time. A completed daily bar can provide context only after that bar has closed. An article fetched today cannot silently become an input to a historical prediction from last year.

Historical availability is sometimes modeled from exchange timestamps plus an explicit latency assumption. That is different from having actual receipt timestamps from the original market session. Prospective collection records when information is observed. This distinction is essential to interpreting backtest realism.

### 3. Convert observations into causal features

The baseline classifier uses 21 OHLCV-derived inputs. They include returns over several trailing horizons, RSI, normalized MACD, rate of change, Donchian position, ADX, momentum, realized volatility, ATR fraction, Parkinson and Garman–Klass volatility, volatility of volatility, volume and price z-scores, autocorrelation, VWAP deviation and rule-based regime flags.

These inputs describe complementary aspects of the market: direction, strength, variation, unusual activity and location relative to recent history. A high RSI is an observation rather than a universal instruction to sell. The model learns combinations of inputs under a particular labeling and cost contract.

Phase 2 adds a wider technical-analysis library: moving averages, Supertrend, SAR, oscillators, bands, VWAP, volume-flow indicators, confirmed pivots and approximate volume profile. Multi-timeframe confluence summarizes agreement across intervals. Equal weights are an explicit fallback; fitted reliability weights require training data and validation.

A volume profile reconstructed from bar prices is approximate. A swing pivot requiring bars to its right becomes usable only after confirmation. Visual plotting offsets do not authorize future information in model inputs.

### 4. Define the target before training

The baseline uses cost-aware triple-barrier labels. A hypothetical trade resolves when it reaches an upper barrier, a lower barrier or a maximum time horizon. The resulting classes are down, flat and up under those rules. This differs from simply asking whether the next closing price is higher.

The default profile uses completed hourly bars with a four-bar horizon. The separate minute profile uses one-minute bars and a one-bar horizon, with isolated artifacts and ledger. Hourly weights cannot be presented as a validated minute model.

Historical labels are training targets only after the future outcome has resolved. Training and evaluation splits remove overlapping label windows so that future outcomes cannot leak through the split boundary.

### 5. Train, calibrate and evaluate

The baseline learns on earlier observations and calibrates on a later chronological portion of development data. Walk-forward folds test the pipeline on subsequent intervals. A final historical holdout provides another evaluation, but becomes exploratory once repeatedly inspected and used for tuning.

Evaluation includes fees, spread assumptions and slippage. Comparisons include buy-and-hold, matched-exposure holding, moving-average crossover and momentum. Useful outputs include return, Sharpe, drawdown, turnover, trade count, probability calibration and return-quantile loss. Cost stress tests and block-bootstrap diagnostics expose sensitivity to assumptions.

Promotion gates check trade count, positive net holdout return, matched-risk baseline Sharpe, configured drawdown and consistency across walk-forward folds. The current baseline gate is not a complete certification of calibration, infrastructure readiness or future performance. Additional prospective review is still necessary.

### 6. Produce a forecast and apply independent risk checks

The service returns a structured signal with probabilities, horizon, decision time, expiry, model version, regime, sizing information, explanation fields and blocking reasons. Forecast direction and trading action are separate. A model can predict an upward outcome while the action remains HOLD because the model is unapproved or the market feed is stale.

The active signal path calls a voting helper with the eligible baseline probability vector. Although ensemble utilities exist, this does not establish a deployed ensemble combining every neural and statistical model.

For spot trading, BUY increases a long position and SELL reduces it toward flat. SELL does not open a leveraged short. HOLD preserves an existing position unless a separate stop or kill switch intervenes.

### 7. Simulate execution and record outcomes

Paper execution uses the next eligible bar open, with fees and slippage. If stop and target both touch within a bar, conservative stop-first handling avoids choosing the favorable sequence without evidence. This is a closed-bar simulator, not a full exchange matching engine.

Signals, paper events, outcomes and human feedback are append-only event records. Outcomes mature separately from the initial prediction, preserving what was known at the time. Evaluation includes abstained predictions rather than scoring only trades selected by the model.

### 8. Monitor and retrain

The service tracks empirical accuracy, Brier score, errors by regime and feature distribution changes using Population Stability Index. Drift indicates a reason to investigate; it does not independently prove that a model has lost its edge.

Scheduled candidate training runs separately from the API process. Candidates must pass configured gates before replacing a champion. Compatible rollback and suspension mechanisms address degraded calibration. Human notes support review; they are not automatically converted into true market labels. Learning is a measured retraining workflow rather than a promise that every losing trade makes the model better.

## Models behind the prototype

| Model or method | Role | Important qualification |
|---|---|---|
| LightGBM classifier | Nonlinear prediction from tabular OHLCV features | Primary baseline when dependency is installed |
| Histogram gradient boosting | Classifier fallback | Used if LightGBM is unavailable |
| Sigmoid calibration | Adjusts raw class probabilities using held-out chronological data | Calibration quality must still be measured |
| LightGBM quantile regression | Estimates 10th, 50th and 90th percentile barrier-exit return outcomes | Optional dependency; these are not guaranteed price bounds |
| Single-timeframe TCN | Learns temporal patterns from causal convolution windows | Experimental, uncalibrated sequence outputs excluded from normal signals |
| Multi-timeframe TCN plus attention | Combines seven temporal representations with state features | Research architecture, not an approved live policy |
| HMM / filtered regime model | Estimates latent market states | Whole-sequence smoothing would leak future information; fast dashboard rules are separately labeled |
| Kalman filtering | Estimates a changing trend or beta | A statistical estimate rather than proof of market direction |
| HAR, GARCH-family and EVT functions | Volatility forecasts and tail diagnostics | Coverage, convergence and sample size determine availability |
| IQL | Offline warm start from simulator transitions | No claim of training on real profitable trader logs |
| PPO and RecurrentPPO | Learn exposure policies from simulator interaction | RecurrentPPO adds recurrent memory; training success is not profit validation |
| SAC | Continuous-action exposure policy research | Policy output is not a calibrated probability |
| QR-DQN | Distributional value learning over discretized exposures | Implemented in Phase 2 research, despite older documents listing it as future work |
| FinBERT | Optional financial-text sentiment | Disabled by default; requires a pinned revision and domain validation |

### Inside the multi-timeframe network

The seven encoder intervals are 30 seconds, 1 minute, 5 minutes, 15 minutes, 1 hour, 4 hours and 1 day. Each encoder reads a trailing window using causal convolutions, producing a representation of recent behavior. Multihead attention combines those representations, with masks for missing intervals. A separate state projection and learned gate combine market embeddings with the configured state vector.

The shared representation feeds heads for target exposure, value, three direction logits, ordered return quantiles, nonnegative risk measures, volatility, regime logits and family summaries. Quantile ordering in this architecture uses a median plus positive lower/upper distances. The research dataset uses compact features and a specific auxiliary input contract; it does not feed every dashboard diagnostic or unavailable news item into the network.

Supervised pretraining combines quantile pinball loss, direction cross-entropy, volatility loss, regime loss and optional risk-target loss. Its direction labels use return thresholds; they must not be conflated with the baseline classifier's triple-barrier probabilities. IQL adds expectile value regression, a Bellman-style Q loss and advantage-weighted behavior cloning. Subsequent RL experiments optimize exposure in a research simulator.

The perpetual simulator includes research support for signed exposure, funding, leverage, maintenance liquidation and trading costs. This is distinct from the normal spot service and does not enable real-money futures trading. A reward shaped by risk and turnover is an optimization signal; account equity remains the authoritative P&L measure.

### Explainability

The baseline can expose LightGBM TreeSHAP contributions for the raw up-class score. These contributions are not contributions to the calibrated probability and are not evidence of causation. Phase 2 includes integrated-gradient research utilities and structured explanation templates. Attention weights describe weighting inside a model; they are not a reliable standalone ranking of economic causes. Explanations must identify their actual basis and show unavailable states where appropriate.

## Standout features already present

The strongest differentiator is the connection between forecasts and evidence. Data receipts, feature versions, model versions, chronological validation and append-only outcomes make it possible to investigate a decision later.

The second is separation of prediction from permission to act. Research-only models, stale inputs, abnormal spreads and portfolio limits can override a directional forecast. The product can explain why it is holding instead of inventing a confident entry.

The third is the combination of short-horizon observations with wider context. Tick-derived flow, live books, hourly analysis and daily trends can be inspected within the same research environment, with explicit missing-data and proxy labels.

The fourth is cost-aware comparison. A model must be assessed against simple strategies and realistic costs rather than against an attractive gross-return chart alone. The recorded failed baseline demonstrates why this distinction matters.

## Risk controls and broker integration

Default settings cap suggested exposure at 20% of paper equity, risk per trade at 0.5%, UTC daily loss at 2% and drawdown at 10%. The annual volatility target is 15%; the configured spread threshold is 15 basis points. These values are profile defaults, not measured guarantees.

Position sizing takes the minimum of exposure, stop-distance risk and volatility caps. Optional quarter-Kelly sizing needs validated net-return moments. The baseline proposes a stop 1.5 ATR away and a target 3 ATR away when an eligible BUY exists. Gaps and execution costs can exceed an intended loss budget.

The broker terminal connector, demo order code, API endpoints and configuration profile have been removed from the supported platform. Existing broker experiment artifacts are retained as archival files and are not loaded by the core service. The service uses Binance SOLUSDT observations, with real-money execution locked.

## New feature: live web strategy finder

### What was implemented

The Strategy finder tab provides on-demand searches against public GitHub repository metadata and README content, plus arXiv research abstracts. The collector uses official API endpoints instead of an unrestricted scraper. This provides live web discovery and text extraction while keeping source access bounded and inspectable.

Each candidate includes a stable source-based ID, original source link, collected timestamp, content hash, description excerpt, source category and available license information. GitHub results include the SPDX license identifier when supplied by the API; missing licensing remains explicitly unknown. Paper abstracts are metadata, not a grant to redistribute full papers.

Lightweight keyword triage assigns broad families such as trend, mean reversion, arbitrage and machine learning. It checks for mentions of entry, exit, risk and costs, highlighting missing categories. This is intentionally labeled keyword triage: a README mentioning a stop loss has not thereby defined a reproducible strategy. The finder supplies no profitability or quality score.

Results are deduplicated by stable ID within a response and cached for ten minutes. Each provider is paced, provider errors are surfaced, and external text is rendered as text rather than executable markup. Downloaded repositories are not installed or executed. Searches accept topics rather than arbitrary destination URLs.

### Additional features implemented with it

1. **Research watchlist:** save or remove up to 50 candidate cards in browser storage, preserving the candidate's collected source details. It is local to that browser, not a shared research database.
2. **Rule-review flags:** highlight absent entry, exit, risk and cost mentions so a researcher can prioritize manual review.
3. **Local benchmark comparison:** display the existing local candidate and baseline returns, Sharpe, drawdown and trade counts, alongside approval failures. These numbers do not belong to web-discovered strategies.
4. **Source transparency:** retain original links, license metadata and collection details rather than presenting an extracted strategy as an original platform invention.

“Live” currently means a user-triggered provider search, subject to caching and quotas. Background continuous crawling, scheduled alerts and automated translation into backtests are future work.

### Source-access checks

GitHub documents repository/code discovery through its [search API](https://docs.github.com/en/rest/search/search) and publishes [request limits](https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api). Repository search is usable without credentials under the public quota; code search has separate authentication requirements. The prototype uses repository search and individual README retrieval, not authenticated global code search.

arXiv supports metadata discovery, with legacy API pacing limited to one request every three seconds and one connection at a time across the operator's machines. Full-paper redistribution has separate restrictions. See the [arXiv API terms](https://info.arxiv.org/help/api/tou.html). The current single-process finder paces searches per source; multiple deployments would need shared quota coordination.

TradingView's [terms](https://www.tradingview.com/policies/) restrict automated processing and non-display use of its content and market data. Its [script-publishing documentation](https://www.tradingview.com/pine-script-docs/writing/publishing/) also sets reuse conditions for open-source scripts. A public page or visible script does not by itself establish permission for a strategy-harvesting product. TradingView is therefore not an implemented collection source. Any later adapter needs suitable authorized access and reuse terms.

### How the complete future discovery pipeline should work

Discovery should produce hypotheses for testing. A later pipeline could move candidates through: discovered → source reviewed → rules specified → reproduced → historically evaluated → forward paper tested → eligible for promotion review. Each transition should preserve the evidence and the person or process responsible.

A structured strategy specification would define instrument, market, timeframe, indicator parameters, entry and exit conditions, stop policy, sizing, cost model, warmup, source revision and required data. A language model could draft that specification with source references. Missing rules should remain unresolved rather than being filled in and attributed to the author.

After review, an internal allowlisted strategy language could map the specification to deterministic calculations. Reproduction should run in an isolated environment with bounded resources. Evaluation should use the existing causal pipeline, chronological purging, costs, multiple-testing records and untouched future paper observations. Arbitrary downloaded Python should never become an import in the signal service.

The date of discovery is especially important. An attractive strategy collected after a historical market move is not evidence that the platform could have selected it before that move. Forward paper testing from the discovery date is the cleanest next confirmation step.

## Further high-value upgrades

| Proposed feature | User benefit | What it requires |
|---|---|---|
| Structured strategy builder | Turn reviewed ideas into explicit reproducible rules | A bounded rule schema, deterministic compiler and validation fixtures |
| Prospective paper tournament | Compare newly discovered strategies on the same future interval | Frozen trial registry, aligned costs, timestamps and scheduled evaluation |
| Regime-aware strategy allocation | Study when trend or reversion candidates work better | Calibrated regime inference and out-of-sample allocation evaluation |
| Scenario laboratory | Inspect fee spikes, gaps, feed outages and volatility shocks | Clearly labeled scenarios, reproducible replay and risk accounting |
| Evidence-linked research assistant | Explain decisions and retrieve related experiments | Retrieval over trusted reports, source links and separation from execution |
| Point-in-time news and event intelligence | Research event-sensitive models | Licensed historical coverage, observed-time joins and incremental-value tests |
| Multi-asset portfolio research | Study correlated exposure across SOL, BTC and ETH | Instrument-generalized contracts, coverage, correlation and portfolio constraints |

## Technology and deployment

The backend uses Python, FastAPI, asyncio, Pydantic, pandas/NumPy, scikit-learn and optional research libraries. PyTorch supports the neural experiments; Stable-Baselines3 and sb3-contrib supply RL algorithms. Parquet stores bulk history. SQLAlchemy provides an append-only operational event store using SQLite locally, with PostgreSQL support in deployment configuration.

The frontend uses Next.js, React, TypeScript and Lightweight Charts. REST endpoints provide canonical state, while WebSockets push signal and health updates. The interface includes overview, Deep Quant, strategy discovery, backtests, model health and paper trading views.

MLflow and experiment artifacts support tracking; Phase 2 also includes Optuna research support. Redis and Docker definitions should be described according to their actual use: Redis provisioning is not proof of a distributed job system. The current service is designed around a single API worker, and production authentication, distributed scheduling, backup drills and operational testing remain additional work.

## Actual evidence and limitations

The saved verification record dated 8 October 2026 reports 37 passing tests for that earlier code snapshot, successful frontend checks and 3,237,305 minute rows across 74 archive months. These are historical recorded results, not a new rerun or the total coverage of every later Phase 2 dataset.

The baseline report at `artifacts/20261008T090625Z-b63e3de7/report.json` records 53,372 eligible feature/label rows and an unapproved candidate. Its holdout had zero trades, zero exposure and zero return. Promotion failures included insufficient trades, nonpositive return and inconsistent walk-forward results. Zero return here describes inactivity, not capital-preserving skill established by evidence.

The Phase 2 pilot report records a one-epoch, one-seed integration pilot with 512 RL steps per algorithm. It remains unapproved and explicitly identifies its historical holdout as previously observed. The root Phase 2 status files still show a larger run in a training stage with October 8 timestamps, and no completed root validation report was found during this inspection. Those stale files do not prove a process is currently training or that the larger campaign completed.

The prototype therefore supports a strong claim about implemented research workflow, auditability and experimental breadth. It does not yet support claims of proven profitability, a validated deep ensemble, universal self-improvement, unrestricted web strategy mining or a production-ready autonomous trading service.

## Source map for technical review

Validation for the 24 October update: 15 targeted strategy/API tests passed; Ruff and mypy passed for the changed backend modules; frontend TypeScript checks passed. Live provider checks returned five GitHub repository/README candidates and five finance-category arXiv abstracts. The production frontend build passed with `npm run build -- --webpack`; the default Turbopack path encountered a CSS worker startup failure. The build also reported a Next.js dependency security advisory, which remains unresolved. Browser visual verification was unavailable because the browser automation runtime could not initialize. These checks do not validate discovered strategies or establish trading performance.

- `backend/data/`: archives, candles, streaming books, derivatives and news.
- `backend/features/`: causal baseline features and triple-barrier labels.
- `backend/models/train.py`: baseline training, calibration, reports and promotion checks.
- `backend/signals.py` and `backend/risk.py`: forecast contracts, eligibility and risk overrides.
- `backend/paper.py`, `backend/outcomes.py`, `backend/store.py`: simulation and append-only outcomes.
- `backend/phase2/network.py`, `learning.py`, `rl.py`, `research.py`: neural and RL research workflow.
- `backend/phase2/ta.py`, `quant.py`, `runtime.py`: indicators, diagnostics and live research snapshots.
- `backend/strategies.py`: newly added source collection, triage, deduplication and pacing.
- `backend/api/app.py`: API endpoints, including discovery and comparison.
- `frontend/components/StrategyFinder.tsx`: discovery, review cards, watchlist and comparison interface.
- `docs/ft-trading.md`, `docs/minute-signals.md`: separate broker-demo and minute-profile contracts.

## Shareable closing description

Supertrader brings the research lifecycle into one terminal: collect trustworthy observations, calculate causal features, evaluate models against realistic baselines, explain decisions, apply independent risk controls and measure subsequent outcomes. Its strategy discovery workspace adds a traceable route from public research to a candidate review queue. The next development milestone is prospective validation and reproducible strategy testing, allowing the platform's measured evidence to grow alongside its feature set.
