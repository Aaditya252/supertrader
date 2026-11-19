"""Completed-bar as-of windows and train-only scaling for seven encoders."""

import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from backend.phase2.bars import INTERVALS
from backend.phase2.ta import ema, wilder

TIMEFRAMES = ["30s", "1m", "5m", "15m", "1h", "4h", "1d"]
MODEL_FEATURES = [
    "return",
    "trend",
    "rsi",
    "atr_fraction",
    "realized_vol",
    "volume_z",
    "flow_imbalance",
    "vwap_deviation",
]


def compact_features(bars, interval):
    """Shared EMA/Wilder definitions; training subset of the full TA/quant report."""
    groups = (bars.open_time.diff().fillna(INTERVALS[interval]) != INTERVALS[interval]).cumsum()
    outputs = []
    for _, d in bars.groupby(groups, sort=False):
        c = d.close
        delta = c.diff()
        tr = pd.concat(
            [d.high - d.low, (d.high - c.shift()).abs(), (d.low - c.shift()).abs()],
            axis=1,
        ).max(axis=1)
        atr = wilder(tr)
        gain = wilder(delta.clip(lower=0))
        loss = wilder((-delta).clip(lower=0))
        rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
        rsi = rsi.where(loss != 0, 100).where((loss != 0) | (gain != 0), 50)
        ret = np.log(c).diff()
        volume_mean = d.volume.rolling(100).mean()
        volume_std = d.volume.rolling(100).std().replace(0, np.nan)
        flow = (
            d.signed_volume_delta / d.volume.replace(0, np.nan)
            if "signed_volume_delta" in d
            else pd.Series(np.nan, index=d.index)
        )
        quote = d.quote_volume if "quote_volume" in d else (d.high + d.low + c) / 3 * d.volume
        vwap = quote.rolling(24).sum() / d.volume.rolling(24).sum().replace(0, np.nan)
        outputs.append(
            pd.DataFrame(
                {
                    "available_at": d.available_at,
                    "open_time": d.open_time,
                    "return": ret,
                    "trend": np.tanh((ema(c, 20) - ema(c, 50)) / atr.replace(0, np.nan)),
                    "rsi": rsi / 100,
                    "atr_fraction": atr / c,
                    "realized_vol": ret.rolling(24).std(),
                    "volume_z": (d.volume - volume_mean) / volume_std,
                    "flow_imbalance": flow,
                    "vwap_deviation": c / vwap - 1,
                }
            )
        )
    return pd.concat(outputs).sort_index().replace([np.inf, -np.inf], np.nan)


def read_bars(root: Path, market, tf, limit=None):
    import duckdb

    directory = Path(root) / "bars" / market / "SOLUSDT" / tf
    files = sorted(directory.glob("*.parquet"))
    receipt_folder = Path(root) / "ticks" / market / "SOLUSDT"
    if receipt_folder.exists():
        files = [f for f in files if (receipt_folder / (f.stem + ".receipt.json")).exists()]
    if not files:
        return pd.DataFrame()
    with duckdb.connect() as con:
        query = "SELECT * FROM read_parquet(?) ORDER BY open_time" + (
            f" DESC LIMIT {int(limit)}" if limit else ""
        )
        d = con.execute(query, [[f.as_posix() for f in files]]).df()
    return d.sort_values("open_time").drop_duplicates("open_time", keep="last").reset_index(drop=True)


def aligned_windows(frames, decision_times, window=24):
    """Search last availability<=decision; never join on a higher-TF open timestamp."""
    times = np.asarray(decision_times, dtype="int64")
    windows = {}
    valid = np.ones(len(times), dtype=bool)
    for tf, frame in frames.items():
        a = frame.available_at.to_numpy(dtype="int64")
        x = frame[MODEL_FEATURES].to_numpy(dtype=np.float32)
        index = np.searchsorted(a, times, side="right") - 1
        good = index >= window - 1
        safe = np.maximum(index, window - 1)
        offsets = safe[:, None] - np.arange(window - 1, -1, -1)
        data = x[np.minimum(offsets, len(x) - 1)]
        good &= np.isfinite(data).all(axis=(1, 2))
        good &= times - a[np.maximum(index, 0)] <= INTERVALS[tf] * 2
        # A sequence spanning an exchange gap is not a valid encoder window.
        sequence_times = frame.open_time.to_numpy(dtype="int64") if "open_time" in frame else a
        good &= np.all(
            np.diff(sequence_times[np.minimum(offsets, len(a) - 1)], axis=1) == INTERVALS[tf], axis=1
        )
        windows[tf] = data
        valid &= good
    return windows, valid


def chronological_split(times, holdout_days=365, calibration_fraction=0.15, horizon_ms=14400000):
    times = np.asarray(times)
    cutoff = times[-1] - holdout_days * 86400000
    eligible = np.flatnonzero(times < cutoff - horizon_ms)
    cut = int(len(eligible) * (1 - calibration_fraction))
    calibration = eligible[cut:]
    train = eligible[times[eligible] < times[calibration[0]] - horizon_ms]
    holdout = np.flatnonzero(times >= cutoff)
    return train, calibration, holdout


def cpcv(times, label_end, groups=6, test_groups=2, embargo_ms=14400000):
    """Combinatorial split with interval-overlap purge and post-test embargo."""
    t = np.asarray(times)
    end = np.asarray(label_end)
    blocks = np.array_split(np.arange(len(t)), groups)
    for selected in itertools.combinations(range(groups), test_groups):
        test = np.sort(np.concatenate([blocks[i] for i in selected]))
        mask = np.ones(len(t), dtype=bool)
        mask[test] = False
        for i in selected:
            block = blocks[i]
            lo, hi = t[block[0]], end[block[-1]]
            mask &= ~((t <= hi + embargo_ms) & (end >= lo))
        yield np.flatnonzero(mask), test


def fit_scale(windows, train):
    """Fit per-timeframe feature location/scale on training observations only."""
    scaled = {}
    scalers = {}
    for tf, x in windows.items():
        reference = x[train].reshape(-1, x.shape[-1])
        mean = reference.mean(axis=0)
        std = np.maximum(reference.std(axis=0), 1e-6)
        scaled[tf] = np.clip((x - mean) / std, -10, 10).astype(np.float32)
        scalers[tf] = {"mean": mean.tolist(), "std": std.tolist()}
    return scaled, scalers
