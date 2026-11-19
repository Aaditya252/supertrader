from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis


@dataclass
class Costs:
    fee_bps: float = 10
    slippage_bps: float = 5
    spread_bps: float = 2

    @property
    def one_way(self) -> float:
        return (self.fee_bps + self.slippage_bps + self.spread_bps/2) / 10000


def metrics(returns: np.ndarray, exposures: np.ndarray | None = None,
            turnover: np.ndarray | None = None, bars_per_year: float = 8760) -> dict[str, Any]:
    r = np.asarray(returns, dtype=float)
    if len(r) < 2 or not np.isfinite(r).all() or (r <= -1).any():
        raise ValueError("Need finite, solvent returns")
    equity = np.cumprod(1+r)
    peaks = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    drawdown = equity/peaks-1
    std = r.std(ddof=1)
    sharpe = float(r.mean()/std*np.sqrt(bars_per_year)) if std > 0 else 0.
    downside = np.sqrt(np.mean(np.minimum(r, 0)**2))
    sortino = float(r.mean()/downside*np.sqrt(bars_per_year)) if downside > 0 else None
    cagr = float(np.expm1(np.log(equity[-1])*bars_per_year/len(r)))
    mdd = float(drawdown.min())
    negative, positive = r[r < 0], r[r > 0]
    var = float(np.quantile(r, .05))
    return {"total_return": float(equity[-1]-1), "cagr": cagr, "sharpe": sharpe,
            "sortino": sortino, "max_drawdown": mdd,
            "calmar": cagr/abs(mdd) if mdd < 0 else None,
            "bar_win_rate": float((r > 0).mean()),
            "bar_profit_factor": float(positive.sum()/abs(negative.sum())) if len(negative) else None,
            "var_95": var, "cvar_95": float(r[r <= var].mean()),
            "exposure": float(np.mean(exposures)) if exposures is not None else None,
            "turnover": float(np.sum(turnover)) if turnover is not None else None,
            "observations": len(r), "annualization": bars_per_year}


def simulate(d: pd.DataFrame, target: np.ndarray | pd.Series, costs: Costs = Costs(),
             stop_atr: float | None = 1.5, take_atr: float = 3., bars_per_year: float = 8760) -> dict:
    """Spot cash/units ledger. Decision i-1 executes at i open; stops include gap slippage."""
    target = np.asarray(target, dtype=float)
    if len(target) != len(d) or not np.isfinite(target).all() or ((target < 0) | (target > 1)).any():
        raise ValueError("Spot target exposure must be finite and between zero and one")
    cash, units, previous_equity = 1., 0., 1.
    stop, take = None, None
    returns, equity, exposure, turnover = [], [], [], []
    trades: list[dict[str, Any]] = []
    entry_equity = None
    interval_ms = int(np.median(np.diff(d.open_time)))
    for i in range(1, len(d)):
        row, prior = d.iloc[i], d.iloc[i-1]
        requested = float(target[i-1])
        if int(row.open_time)-int(prior.open_time) != interval_ms:
            # The previous prediction expired during an outage; flatten at first observable open.
            requested = 0.
        opening_equity = cash + units*float(row.open)
        desired_units = requested * opening_equity / float(row.open)
        change = desired_units - units
        notional = change*float(row.open)
        cost = abs(notional)*costs.one_way
        if change > 0:
            change = min(change, cash/(float(row.open)*(1+costs.one_way)))
            notional = change*float(row.open)
            cost = abs(notional)*costs.one_way
        was_open = units > 1e-12
        cash -= notional+cost
        units += change
        traded = abs(notional)
        if not was_open and units > 1e-12:
            entry_equity = opening_equity
            if stop_atr and np.isfinite(prior.atr):
                stop = float(row.open)-stop_atr*float(prior.atr)
                take = float(row.open)+take_atr*float(prior.atr)
        if was_open and units <= 1e-12:
            assert entry_equity is not None
            trades.append({"exit_time": int(row.open_time), "net_return": cash/entry_equity-1,
                           "reason": "rebalance"})
            stop = take = entry_equity = None
        exit_price, reason = None, None
        if units > 1e-12 and stop is not None:
            assert take is not None
            if float(row.low) <= stop:
                exit_price, reason = min(float(row.open), stop), "stop"
            elif float(row.high) >= take:
                exit_price, reason = max(float(row.open), take), "target"
        if exit_price is not None:
            assert entry_equity is not None
            proceeds = units*exit_price
            cash += proceeds*(1-costs.one_way)
            traded += proceeds
            units = 0.
            trades.append({"exit_time": int(row.close_time), "net_return": cash/entry_equity-1,
                           "reason": reason})
            stop = take = entry_equity = None
        closing_equity = cash+units*float(row.close)
        returns.append(closing_equity/previous_equity-1)
        equity.append(closing_equity)
        turnover.append(traded/max(opening_equity, 1e-12))
        exposure.append(units*float(row.close)/max(closing_equity, 1e-12))
        previous_equity = closing_equity
    # Charge terminal liquidation so comparison includes the exit cost.
    if units > 1e-12:
        assert entry_equity is not None
        final = cash+units*float(d.close.iloc[-1])*(1-costs.one_way)
        returns[-1] = (1+returns[-1])*final/equity[-1]-1
        equity[-1] = final
        turnover[-1] += units*float(d.close.iloc[-1])/max(final, 1e-12)
        trades.append({"exit_time": int(d.close_time.iloc[-1]), "net_return": final/entry_equity-1,
                       "reason": "terminal"})
    result = metrics(np.array(returns), np.array(exposure), np.array(turnover), bars_per_year)
    wins = [t["net_return"] for t in trades if t["net_return"] > 0]
    losses = [t["net_return"] for t in trades if t["net_return"] < 0]
    result.update({"trade_count": len(trades), "win_rate": len(wins)/len(trades) if trades else None,
                   "profit_factor": sum(wins)/abs(sum(losses)) if losses else None})
    eq = np.array(equity)
    return {"metrics": result, "returns": returns, "equity": equity,
            "drawdown": (eq/np.maximum.accumulate(np.r_[1., eq])[1:]-1).tolist(),
            "time": d.open_time.iloc[1:].astype(int).tolist(), "trades": trades,
            "regimes": {regime: metrics(np.array(returns)[mask], bars_per_year=bars_per_year)
                         for regime in d.regime.iloc[1:].unique()
                         if (mask := (d.regime.iloc[1:].to_numpy() == regime)).sum() >= 2}}


def baseline_targets(d: pd.DataFrame, cap: float = .2) -> dict[str, np.ndarray]:
    return {"buy_hold": np.ones(len(d)), "risk_matched_hold": np.full(len(d), cap),
            "ma_crossover": np.where(d.ema20 > d.ema50, cap, 0),
            "momentum": np.where(d.momentum > 0, cap, 0)}


def deflated_sharpe(returns: np.ndarray, trials: int, bars_per_year: float = 8760) -> dict:
    """Approximate DSR with explicit trial-count and independent-return assumptions."""
    r = np.asarray(returns)
    if len(r) < 30 or r.std() == 0:
        return {"probability": None, "reason": "insufficient nonconstant returns"}
    sr = r.mean()/r.std(ddof=1)
    se = np.sqrt(max(1-skew(r)*sr+(kurtosis(r, fisher=False)-1)*sr*sr/4, 1e-12)/(len(r)-1))
    expected_max = norm.ppf(1-1/max(trials, 2)) / np.sqrt(len(r)-1) if trials > 1 else 0
    return {"probability": float(norm.cdf((sr-expected_max)/se)), "trials": trials,
            "annualized_sharpe": float(sr*np.sqrt(bars_per_year)),
            "assumption": "approximate independent observations; inspect block bootstrap too"}


def block_bootstrap(returns: np.ndarray, seed: int = 42, samples: int = 200, block: int = 24) -> dict:
    rng = np.random.default_rng(seed)
    r = np.asarray(returns)
    paths = []
    for _ in range(samples):
        chunks = [r[int(s):int(s)+block] for s in rng.integers(0, max(1, len(r)-block+1),
                                                            size=int(np.ceil(len(r)/block)))]
        sample = np.concatenate(chunks)[:len(r)]
        paths.append(np.prod(1+sample)-1)
    return {"return_p05": float(np.quantile(paths, .05)), "return_p50": float(np.median(paths)),
            "return_p95": float(np.quantile(paths, .95)), "samples": samples, "block_bars": block}

