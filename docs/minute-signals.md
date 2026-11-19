# Minute SOL research

Use the separate minute profile to train on one-minute candles and forecast
net-cost triple-barrier outcomes over the next one-minute bar. These are barrier
outcomes, not a guarantee of the next closing-price direction. FLAT is a valid
prediction. Direction and direction_confidence describe the model forecast;
action remains HOLD when approval or risk gates fail.

```powershell
.\.venv\Scripts\python.exe -m backend.cli --config configs/minute-paper.yaml train
.\.venv\Scripts\python.exe -m backend.cli --config configs/minute-paper.yaml serve
```

Training requires existing SOLUSDT one-minute history. If it is missing, run
the download command with this configuration and `--timeframes 1m` first.
The service polls every five seconds and uses only finished candles, producing
one forecast per new available minute. Feed outages can prevent a forecast.
Minute artifacts and the paper ledger are isolated from hourly experiments.
Existing hourly weights must not be used as a minute model.

Candidate retraining runs serially every 24 hours after service start, with the
same configuration as the service. Only candidates passing existing historical
promotion gates can become champions. Repeated historical evaluation is
exploratory; future paper results are still required to establish performance.
No approved model means no paper entries. Short forecasts mean reducing a spot
position, not leveraged short selling. Live execution remains locked.

The requested external platform name has not been identified. No exchange or
external trading account is connected by this configuration.
