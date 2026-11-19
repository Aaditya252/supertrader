# Research lessons and safeguards

Recorded during the initial build on 2026-10-08. These are engineering lessons, not evidence of a profitable trading edge.

- Historical exchange archives contained three malformed candle timestamp records and truncated outage candles. Quarantine preserves the raw evidence; incomplete records are excluded and resulting gaps stay visible. Indicators restart their warm-up across gaps.
- A Gymnasium termination value initially used a NumPy Boolean. Stable-Baselines environment validation exposed it; the environment now returns Python Booleans and a regression test runs the environment checker.
- Execution and labels must agree about when a decision becomes tradable. Labels and simulations use the following bar's open, include costs, and use conservative stop-first handling when both barriers are touched. Gap exits use observable prices.
- Optional model experiments cannot bypass calibration and promotion. The actual supervised candidate failed its gates; its absence of qualifying trades is reported as abstention, not a profitable result. Experimental TCN probabilities are uncalibrated. Short PPO/SAC trials lost value and remain excluded.
- A stale browser signal must not stay actionable after a fetch failure. Both the API and terminal expose stale data as HOLD. Version/configuration mismatches also block action.
- Startup risk checks can differ from recovered live checks within the same decision bar. The current view refreshes independently of immutable, idempotent audit records; a regression test prevents a startup outage reason from remaining latched after recovery.

## Learning from subsequent outcomes

Signals keep their inputs, version, probability definition and risk reasons. Outcomes are attached only after the label horizon matures, using idempotent event keys. Monitoring reports classification accuracy, Brier score and errors by regime. Prediction outcomes are not presented as executed trade P&L.

Weekly retraining creates a separately versioned candidate. Chronological validation and promotion gates decide whether it can serve. A degradation monitor can suspend an approved model or roll back to a compatible approved predecessor. This is controlled evaluation and retraining; feedback does not instantly rewrite weights or guarantee that the next prediction improves.

Before tuning thresholds after inspecting results, reserve new untouched evaluation data. Collect prospective paper evidence and register research trials; a favorable historical score alone must not justify real-money deployment.
