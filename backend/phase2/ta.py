"""Trailing TA; plotting shifts never introduce future information."""

import numpy as np
import pandas as pd
from backend.phase2.bars import INTERVALS

VERSION = "phase2-ta-v1"


def divide(a, b):
    return a / b.replace(0, np.nan) if isinstance(b, pd.Series) else a / np.where(b == 0, np.nan, b)


def ema(x, n):
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def wilder(x, n=14):
    return x.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def sar(high, low):
    """Recursive Wilder SAR; reversal uses only current/past extremes."""
    h, lows = np.asarray(high), np.asarray(low)
    out = np.full(len(h), np.nan)
    if len(h) < 3:
        return out
    bull, ep, acc, value = True, h[0], 0.02, lows[0]
    for i in range(1, len(h)):
        value += acc * (ep - value)
        if bull:
            value = min(value, lows[i - 1], lows[max(0, i - 2)])
            if lows[i] < value:
                bull, value, ep, acc = False, ep, lows[i], 0.02
            elif h[i] > ep:
                ep, acc = h[i], min(0.2, acc + 0.02)
        else:
            value = max(value, h[i - 1], h[max(0, i - 2)])
            if h[i] > value:
                bull, value, ep, acc = True, ep, h[i], 0.02
            elif lows[i] < ep:
                ep, acc = lows[i], min(0.2, acc + 0.02)
        out[i] = value
    return out


def supertrend(close, high, low, atr):
    c, h, lows, a = map(np.asarray, [close, high, low, atr])
    upper, lower = (h + lows) / 2 + 3 * a, (h + lows) / 2 - 3 * a
    trend = np.full(len(c), np.nan)
    band = np.full(len(c), np.nan)
    start = np.flatnonzero(np.isfinite(a))
    if not len(start):
        return trend, band
    first = int(start[0])
    trend[first] = 1
    band[first] = lower[first]
    for i in range(first + 1, len(c)):
        if upper[i] >= upper[i - 1] and c[i - 1] <= upper[i - 1]:
            upper[i] = upper[i - 1]
        if lower[i] <= lower[i - 1] and c[i - 1] >= lower[i - 1]:
            lower[i] = lower[i - 1]
        trend[i] = 1 if c[i] > upper[i - 1] else -1 if c[i] < lower[i - 1] else trend[i - 1]
        band[i] = lower[i] if trend[i] > 0 else upper[i]
    return trend, band


def profile(price, volume, n=64):
    """Trailing typical-price histogram approximation; not a tick-level profile."""
    p, v = np.asarray(price), np.asarray(volume)
    out = np.full((len(p), 3), np.nan)
    for i in range(n - 1, len(p)):
        values, edges = np.histogram(p[i - n + 1 : i + 1], bins=16, weights=v[i - n + 1 : i + 1])
        if values.sum() <= 0:
            continue
        centers = (edges[:-1] + edges[1:]) / 2
        chosen = np.argsort(values)[::-1]
        covered = np.searchsorted(np.cumsum(values[chosen]), 0.7 * values.sum()) + 1
        selected = chosen[:covered]
        out[i] = [
            centers[np.argmax(values)],
            edges[selected.min()],
            edges[selected.max() + 1],
        ]
    return out


def _segment(bars: pd.DataFrame) -> pd.DataFrame:
    d = bars.copy()
    c, h, lows, v = d.close, d.high, d.low, d.volume
    prev = c.shift()
    delta = c.diff()
    typ = (h + lows + c) / 3
    tr = pd.concat([h - lows, (h - prev).abs(), (lows - prev).abs()], axis=1).max(axis=1)
    d["atr"] = wilder(tr)
    d["atr_fraction"] = divide(d.atr, c)
    for n in [9, 20, 50, 100, 200]:
        d[f"ema_{n}"] = ema(c, n)
        d[f"sma_{n}"] = c.rolling(n).mean()
    up, down = h.diff(), -lows.diff()
    d["dmi_plus"] = 100 * divide(wilder(up.where((up > down) & (up > 0), 0)), d.atr)
    d["dmi_minus"] = 100 * divide(wilder(down.where((down > up) & (down > 0), 0)), d.atr)
    d["adx"] = wilder(100 * divide((d.dmi_plus - d.dmi_minus).abs(), d.dmi_plus + d.dmi_minus))
    gain, loss = wilder(delta.clip(lower=0)), wilder((-delta).clip(lower=0))
    d["rsi"] = 100 - 100 / (1 + divide(gain, loss))
    d.loc[(loss == 0) & (gain > 0), "rsi"] = 100
    d.loc[(loss == 0) & (gain == 0), "rsi"] = 50
    d["stochrsi"] = divide(
        d.rsi - d.rsi.rolling(14).min(),
        d.rsi.rolling(14).max() - d.rsi.rolling(14).min(),
    )
    d["macd"] = ema(c, 12) - ema(c, 26)
    d["macd_signal"] = ema(d.macd, 9)
    d["macd_hist"] = d.macd - d.macd_signal
    d["roc"] = c.pct_change(12)
    mean = typ.rolling(20).mean()
    mad = typ.rolling(20).apply(lambda a: np.mean(np.abs(a - a.mean())), raw=True)
    d["cci"] = divide(typ - mean, 0.015 * mad)
    hh, ll = h.rolling(14).max(), lows.rolling(14).min()
    d["williams_r"] = -100 * divide(hh - c, hh - ll)
    d["tsi"] = 100 * divide(ema(ema(delta, 25), 13), ema(ema(delta.abs(), 25), 13))
    d["awesome_oscillator"] = ((h + lows) / 2).rolling(5).mean() - ((h + lows) / 2).rolling(34).mean()
    mid = c.rolling(20).mean()
    std = c.rolling(20).std(ddof=0)
    d["bollinger_upper"] = mid + 2 * std
    d["bollinger_lower"] = mid - 2 * std
    d["bollinger_percent_b"] = divide(c - d.bollinger_lower, 4 * std)
    d["bollinger_bandwidth"] = divide(4 * std, mid)
    d["squeeze"] = (d.bollinger_bandwidth < d.bollinger_bandwidth.rolling(100).quantile(0.2)).astype(float)
    d["realized_vol"] = np.log(c).diff().rolling(24).std()
    d["donchian_high"] = h.rolling(20).max().shift()
    d["donchian_low"] = lows.rolling(20).min().shift()
    d["keltner_upper"] = ema(typ, 20) + 2 * d.atr
    d["keltner_lower"] = ema(typ, 20) - 2 * d.atr
    for n, name in [(9, "tenkan"), (26, "kijun"), (52, "span_b")]:
        d[f"ichimoku_{name}"] = (h.rolling(n).max() + lows.rolling(n).min()) / 2
    d["ichimoku_span_a"] = (d.ichimoku_tenkan + d.ichimoku_kijun) / 2
    d["supertrend_direction"], d["supertrend"] = supertrend(c, h, lows, d.atr)
    d["parabolic_sar"] = sar(h, lows)
    x = np.arange(20)
    center = x - x.mean()
    denom = np.sum(center**2)
    d["regression_slope"] = c.rolling(20).apply(lambda a: np.dot(center, a - a.mean()) / denom, raw=True)
    d["regression_r2"] = c.rolling(20).apply(
        lambda a: np.dot(center, a - a.mean()) ** 2 / max(denom * np.sum((a - a.mean()) ** 2), 1e-12),
        raw=True,
    )
    d["obv"] = (np.sign(delta).fillna(0) * v).cumsum()
    multiplier = divide(2 * c - h - lows, h - lows).fillna(0)
    d["cmf"] = divide((multiplier * v).rolling(20).sum(), v.rolling(20).sum())
    mf = typ * v
    change = typ.diff()
    positive = mf.where(change > 0, 0).rolling(14).sum()
    negative = mf.where(change < 0, 0).rolling(14).sum()
    d["mfi"] = 100 - 100 / (1 + divide(positive, negative))
    quote = d.quote_volume if "quote_volume" in d else typ * v
    d["vwap_rolling"] = divide(quote.rolling(24).sum(), v.rolling(24).sum())
    weighted_var = divide(((typ - d.vwap_rolling) ** 2 * v).rolling(24).sum(), v.rolling(24).sum())
    d["vwap_band_upper"] = d.vwap_rolling + 2 * np.sqrt(weighted_var)
    d["vwap_band_lower"] = d.vwap_rolling - 2 * np.sqrt(weighted_var)
    day = d.open_time // 86400000
    d["anchored_vwap"] = divide(quote.groupby(day).cumsum(), v.groupby(day).cumsum())
    if "signed_volume_delta" in d:
        d["cvd"] = d.signed_volume_delta.cumsum()
        d["flow_imbalance"] = divide(d.signed_volume_delta, v)
    else:
        d["cvd"] = np.nan
        d["flow_imbalance"] = np.nan
    vp = profile(typ, v)
    d["profile_poc"], d["profile_val"], d["profile_vah"] = vp.T
    pivot_high = h.shift(2).where(h.shift(2) == h.rolling(5).max())
    pivot_low = lows.shift(2).where(lows.shift(2) == lows.rolling(5).min())
    d["swing_high"] = pivot_high.ffill()
    d["swing_low"] = pivot_low.ffill()
    d["resistance"] = d.swing_high
    d["support"] = d.swing_low
    width = d.swing_high - d.swing_low
    for level in [0.236, 0.382, 0.5, 0.618, 0.786]:
        d[f"fib_{str(level).replace('.', '_')}"] = d.swing_high - level * width
    daily = (
        pd.DataFrame({"day": day, "high": h, "low": lows, "close": c})
        .groupby("day")
        .agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
        .shift()
    )
    pivot = day.map((daily.high + daily.low + daily.close) / 3)
    d["pivot"] = pivot
    d["pivot_r1"] = 2 * pivot - day.map(daily.low)
    d["pivot_s1"] = 2 * pivot - day.map(daily.high)
    d["breakout_up"] = (c > d.donchian_high).astype(float)
    d["breakout_down"] = (c < d.donchian_low).astype(float)
    d["fakeout_up"] = ((h > d.donchian_high) & (c <= d.donchian_high)).astype(float)
    body = (c - d.open).abs()
    range_ = h - lows
    d["doji_weak"] = (body <= 0.1 * range_).astype(float)
    d["hammer_weak"] = (
        (np.minimum(c, d.open) - lows >= 2 * body) & (h - np.maximum(c, d.open) <= body)
    ).astype(float)
    d["engulfing_weak"] = (
        (c > d.open) & (prev < d.open.shift()) & (c >= d.open.shift()) & (d.open <= prev)
    ).astype(float)
    pivot_rsi = d.rsi.shift(2).where(pivot_high.notna()).ffill()
    prior_high = pivot_high.ffill().shift()
    prior_rsi = pivot_rsi.shift()
    d["bearish_rsi_divergence"] = ((pivot_high > prior_high) & (pivot_rsi < prior_rsi)).astype(float)
    for family in ["cvd", "open_interest"]:
        if family in d:
            value = d[family].shift(2).where(pivot_high.notna()).ffill()
            d[f"bearish_{family}_divergence"] = ((pivot_high > prior_high) & (value < value.shift())).astype(
                float
            )
    d["trend_score"] = np.tanh(divide(d.ema_20 - d.ema_50, d.atr))
    d["momentum_score"] = ((d.rsi - 50) / 50).clip(-1, 1)
    d["ta_score"] = (d.trend_score + d.momentum_score) / 2
    return d.replace([np.inf, -np.inf], np.nan)


def features(bars: pd.DataFrame, interval: str) -> pd.DataFrame:
    if bars.empty:
        return bars.copy()
    if not bars.open_time.is_monotonic_increasing or bars.open_time.duplicated().any():
        raise ValueError("Sorted unique bars required")
    groups = (bars.open_time.diff().fillna(INTERVALS[interval]) != INTERVALS[interval]).cumsum()
    return pd.concat([_segment(g) for _, g in bars.groupby(groups, sort=False)]).sort_index()


def report(frame: pd.DataFrame, interval: str) -> dict:
    if frame.empty:
        return {"interval": interval, "status": "unavailable", "features": {}}
    row = frame.iloc[-1]
    values = {
        k: float(v) for k, v in row.items() if isinstance(v, (float, int, np.number)) and np.isfinite(v)
    }
    return {
        "interval": interval,
        "status": "ready" if np.isfinite(row.ta_score) else "warming_up",
        "available_at": int(row.available_at),
        "version": VERSION,
        "features": values,
        "interpretation": "Positive score indicates trailing trend/momentum agreement; it does not predict a certain return.",
        "volume_profile_basis": (
            "estimated from trailing 64 bars: 16 bins; each bar's full volume is assigned to "
            "(high + low + close) / 3; value area spans bins covering 70% of estimated volume; "
            "not trade-by-trade volume-at-price"
        ),
        "pattern_basis": (
            "descriptive causal flags on the latest available bar; pivot divergences require "
            "confirmed pivots; not a future-pattern probability model"
        ),
        "ichimoku_basis": "unshifted computable spans; no future plotting values",
        "flow_status": "observed taker flow"
        if np.isfinite(row.get("flow_imbalance", np.nan))
        else "unavailable",
    }


def confluence(reports: dict, weights: dict | None = None) -> dict:
    usable = {tf: r["features"]["ta_score"] for tf, r in reports.items() if r["status"] == "ready"}
    weights = weights or {tf: 1.0 for tf in usable}
    denominator = sum(max(0, weights.get(tf, 0)) for tf in usable)
    score = (
        sum(max(0, weights.get(tf, 0)) * v for tf, v in usable.items()) / denominator if denominator else None
    )
    return {
        "score": score,
        "timeframes": usable,
        "weights": weights,
        "weight_basis": "training-derived reliability"
        if any(v != 1 for v in weights.values())
        else "equal weights; untrained fallback",
    }


def reliability_weights(training_scores: pd.DataFrame, future_return: pd.Series) -> dict:
    """Call inside a training fold only; weights=max(0,Spearman(score,label))."""
    return {
        k: max(0.0, float(training_scores[k].corr(future_return, method="spearman") or 0))
        for k in training_scores
    }
