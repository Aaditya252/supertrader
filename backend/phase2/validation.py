"""Cost-aware common-episode evaluation and registered, purged family comparisons."""

import numpy as np
from typing import Any
from backend.backtest.engine import metrics, deflated_sharpe, block_bootstrap
from backend.phase2.rl import MultiTimeframeEnv, DiscreteExposure


def longest_segment(bars, indices):
    indices = np.asarray(indices, dtype=int)
    splits = (
        np.flatnonzero((np.diff(indices) != 1) | (np.diff(bars.open_time.to_numpy()[indices]) != 3600000)) + 1
    )
    parts = np.split(indices, splits)
    return max(parts, key=len)


def evaluate(windows, state, bars, policy, funding=None, algorithm=None, seed=42, regimes=None):
    """Same prices/costs for all policies; stops flatten remaining assessment period."""
    env: Any = MultiTimeframeEnv(windows, state, bars, randomize=False, funding=funding, drawdown_stop=0.99)
    if algorithm == "QRDQN":
        env = DiscreteExposure(env)
    observation, _ = env.reset(seed=seed)
    done = False
    records = []
    lstm_state = None
    starts = np.array([True])
    while not done:
        index = env.unwrapped.i
        if algorithm == "RecurrentPPO":
            action, lstm_state = policy.predict(
                observation, state=lstm_state, episode_start=starts, deterministic=True
            )
            starts[:] = False
        elif algorithm:
            action, _ = policy.predict(observation, deterministic=True)
        else:
            action = np.array([policy(index, observation)], dtype=np.float32)
        observation, _, done, _, info = env.step(action)
        records.append({"index": index, "time": int(bars.open_time.iloc[index + 2]), **info})
    assessed = len(bars) - env.unwrapped.window - 1
    # Remaining balance earns no return after a liquidation/kill event.
    returns = np.r_[[r["net_return"] for r in records], np.zeros(max(0, assessed - len(records)))]
    exposures = np.r_[[r["exposure"] for r in records], np.zeros(max(0, assessed - len(records)))]
    result = metrics(np.maximum(returns, -1 + 1e-12), np.abs(exposures), bars_per_year=8760)
    result["net_directional_exposure"] = float(np.mean(exposures))
    if (returns <= -1).any():
        result.update(total_return=-1.0, cagr=-1.0, max_drawdown=-1.0)
    result["turnover"] = float(sum(r["turnover"] for r in records))
    result["liquidations"] = sum(r["liquidated"] for r in records)
    per_regime = {}
    if regimes is not None:
        labels = np.asarray(regimes)[env.unwrapped.window - 1 : env.unwrapped.window - 1 + len(returns)]
        for label in np.unique(labels):
            select = labels == label
            subset = returns[: len(labels)][select]
            if len(subset) >= 2:
                per_regime[str(label)] = metrics(np.maximum(subset, -1 + 1e-12), bars_per_year=8760)
    stride = max(1, len(records) // 600)
    return {
        "metrics": result,
        "returns": returns.tolist(),
        "curve": [{"time": r["time"], "equity": r["equity"]} for r in records[::stride]],
        "regime_metrics": per_regime,
        "execution_coverage_bars": len(records),
        "assessment_bars": len(returns),
        "stopped_equity_is_flat": True,
        "funding_total": float(sum(r["funding_cost"] for r in records)),
    }


def compact(result):
    return {k: v for k, v in result.items() if k != "returns"}


def diagnostics(result, registered_trials, seed=42):
    r = np.asarray(result["returns"])
    return {
        "deflated_sharpe": deflated_sharpe(r, registered_trials),
        "bootstrap": block_bootstrap(r, seed),
        "pbo": {"status": "unavailable: independently comparable nested research search not completed"},
        "interpretation": "Historical evaluation after initial registration; repeated inspection forfeits untouched status.",
    }
