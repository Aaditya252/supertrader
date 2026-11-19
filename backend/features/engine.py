"""All rolling calculations are trailing; no full-series fitted transforms."""
import numpy as np
import pandas as pd
from scipy.stats import entropy

FEATURE_VERSION = "causal-ohlcv-v1"
FEATURE_COLUMNS = ["return_1", "return_4", "return_12", "return_24", "rsi", "macd",
                   "roc", "donchian", "adx", "momentum", "realized_vol", "atr_fraction",
                   "parkinson", "garman_klass", "vol_of_vol", "volume_z", "price_z",
                   "autocorrelation", "vwap_deviation", "regime_trend", "regime_crash"]


def safe_div(a, b):
    return a / b.replace(0, np.nan)


def build_features(candles: pd.DataFrame, interval_ms: int = 3600000) -> pd.DataFrame:
    d = candles.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True).copy()
    # Reset warmup after every missing bar: indicators may not bridge an outage.
    groups = d.open_time.diff().ne(interval_ms).cumsum()
    if groups.nunique() > 1:
        return pd.concat([_features(group) for _, group in d.groupby(groups)], ignore_index=True)
    return _features(d)


def _features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    c, h, low, o, v = [d[k].astype(float) for k in ["close", "high", "low", "open", "volume"]]
    ret = np.log(c).diff()
    for n in [1, 4, 12, 24]:
        d[f"return_{n}"] = np.log(c / c.shift(n))
    change = c.diff()
    gain = change.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    d["rsi"] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).where(loss != 0, 100)
    d.loc[(gain == 0) & (loss == 0), "rsi"] = 50
    d["ema20"] = c.ewm(span=20, adjust=False).mean()
    d["ema50"] = c.ewm(span=50, adjust=False).mean()
    d["macd"] = (c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()) / c
    d["roc"] = c.pct_change(12)
    upper, lower = h.shift(1).rolling(24).max(), low.shift(1).rolling(24).min()
    d["donchian"] = safe_div(c - lower, upper - lower)
    tr = pd.concat([h - low, (h - c.shift()).abs(), (low - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    d["atr"] = atr
    d["atr_fraction"] = atr / c
    up, down = h.diff(), -low.diff()
    plus = up.where((up > down) & (up > 0), 0).ewm(alpha=1/14, adjust=False).mean()
    minus = down.where((down > up) & (down > 0), 0).ewm(alpha=1/14, adjust=False).mean()
    dx = (plus - minus).abs() / (plus + minus).replace(0, np.nan)
    d["adx"] = dx.ewm(alpha=1/14, adjust=False, min_periods=28).mean()
    d["momentum"] = c / c.shift(48) - 1
    d["realized_vol"] = ret.rolling(24).std(ddof=1)
    d["parkinson"] = np.sqrt((np.log(h/low)**2).rolling(24).mean() / (4*np.log(2)))
    gk = .5*np.log(h/low)**2 - (2*np.log(2)-1)*np.log(c/o)**2
    d["garman_klass"] = np.sqrt(gk.rolling(24).mean().clip(lower=0))
    d["vol_of_vol"] = d.realized_vol.rolling(24).std()
    # References exclude the current bar; no global normalization.
    d["volume_z"] = safe_div(v - v.shift().rolling(48).mean(), v.shift().rolling(48).std())
    d["price_z"] = safe_div(c - c.shift().rolling(48).mean(), c.shift().rolling(48).std())
    d["autocorrelation"] = ret.rolling(48).corr(ret.shift(1))
    typical = (h + low + c) / 3
    vwap = safe_div((typical*v).rolling(24).sum(), v.rolling(24).sum())
    d["vwap_deviation"] = c / vwap - 1
    d["regime_trend"] = ((d.ema20 > d.ema50) & (d.adx > .2)).astype(float)
    historic_vol = d.realized_vol.shift().rolling(168, min_periods=48).quantile(.9)
    d["regime_crash"] = ((d.realized_vol > historic_vol) & (d.return_4 < -2*d.realized_vol)).astype(float)
    d["regime"] = np.select([d.regime_crash == 1, d.regime_trend == 1],
                             ["high-vol crash", "trend"], default="range")
    d[FEATURE_COLUMNS] = d[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return d


def causal_join(left: pd.DataFrame, right: pd.DataFrame, columns: list[str], prefix: str) -> pd.DataFrame:
    if "available_at" not in right or "available_at" not in left:
        raise ValueError("Point-in-time join requires availability timestamps")
    r = right[["available_at", *columns]].rename(columns={c: prefix+c for c in columns})
    return pd.merge_asof(left.sort_values("available_at"), r.sort_values("available_at"),
                         on="available_at", direction="backward")


def fractional_difference(series: pd.Series, order: float = .4, width: int = 48) -> pd.Series:
    weights = [1.0]
    for k in range(1, width):
        weights.append(-weights[-1] * (order-k+1) / k)
    w = np.array(weights)[::-1]
    return series.rolling(width).apply(lambda x: float(np.dot(x, w)), raw=True)


def experimental_statistics(series: pd.Series, window: int = 96) -> pd.DataFrame:
    def hurst(x):
        lags = np.arange(2, min(20, len(x)//3))
        tau = np.array([np.std(x[k:]-x[:-k]) for k in lags])
        if (tau <= 0).any():
            return np.nan
        return np.polyfit(np.log(lags), np.log(tau), 1)[0]
    returns = np.log(series).diff()
    return pd.DataFrame({"hurst": np.log(series).rolling(window).apply(hurst, raw=True),
                         "entropy": returns.rolling(window).apply(
                             lambda x: entropy(np.histogram(x, bins=10)[0] + 1e-9), raw=True),
                         "fractional_difference": fractional_difference(np.log(series))})

