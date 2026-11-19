"""Identical OHLCV/flow definitions for archive chunks and live aggregate trades."""

import numpy as np
import pandas as pd

INTERVALS = {
    "30s": 30000,
    "1m": 60000,
    "5m": 300000,
    "15m": 900000,
    "1h": 3600000,
    "2h": 7200000,
    "4h": 14400000,
    "6h": 21600000,
    "12h": 43200000,
    "1d": 86400000,
}
TICK_COLUMNS = [
    "aggregate_id",
    "price",
    "quantity",
    "first_id",
    "last_id",
    "timestamp",
    "buyer_maker",
]


def normalize_ticks(raw: pd.DataFrame, enforce_order=True) -> tuple[pd.DataFrame, dict]:
    """Normalize microseconds to milliseconds; maker-buy is taker-sell."""
    d = raw.iloc[:, :7].copy()
    d.columns = TICK_COLUMNS
    d = d[pd.to_numeric(d.aggregate_id, errors="coerce").notna()].copy()
    for c in TICK_COLUMNS[:-1]:
        d[c] = pd.to_numeric(d[c], errors="raise")
    for c in ["aggregate_id", "first_id", "last_id", "timestamp"]:
        d[c] = d[c].astype("int64")
    d.loc[d.timestamp > 100000000000000, "timestamp"] //= 1000
    maker = d.buyer_maker.astype(str).str.lower()
    valid_maker = maker.isin(["true", "false", "1", "0"])
    valid = valid_maker & (d.price > 0) & (d.quantity > 0) & (d.last_id >= d.first_id)
    bad = int((~valid).sum())
    d = d.loc[valid].copy()
    d["buyer_maker"] = maker.loc[valid].isin(["true", "1"])
    duplicates = int(d.aggregate_id.duplicated().sum())
    d = d.drop_duplicates("aggregate_id", keep="first")
    if enforce_order and (
        (d.timestamp.diff().dropna() < 0).any() or (d.aggregate_id.diff().dropna() <= 0).any()
    ):
        raise ValueError("Trades must be sorted by exchange time and increasing aggregate ID")
    d["trade_count"] = d.last_id - d.first_id + 1
    d["quote_volume"] = d.price * d.quantity
    d["buy_volume"] = np.where(d.buyer_maker, 0.0, d.quantity)
    d["sell_volume"] = np.where(d.buyer_maker, d.quantity, 0.0)
    return d.reset_index(drop=True), {"invalid_rows": bad, "duplicate_rows": duplicates}


def tick_bars(ticks: pd.DataFrame, interval: str) -> pd.DataFrame:
    """O=first p,H=max p,L=min p,C=last p,V=sum q; VWAP=sum(pq)/V."""
    if interval not in INTERVALS:
        raise ValueError("Unsupported bar interval")
    if ticks.empty:
        return pd.DataFrame()
    d = ticks.copy()
    d["open_time"] = d.timestamp // INTERVALS[interval] * INTERVALS[interval]
    g = d.groupby("open_time", sort=True).agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("quantity", "sum"),
        quote_volume=("quote_volume", "sum"),
        buy_volume=("buy_volume", "sum"),
        sell_volume=("sell_volume", "sum"),
        trade_count=("trade_count", "sum"),
        aggregate_trade_count=("aggregate_id", "size"),
    )
    return finish_bars(g.reset_index(), interval)


def finish_bars(g: pd.DataFrame, interval: str) -> pd.DataFrame:
    g = g.copy()
    g["vwap"] = g.quote_volume / g.volume
    g["signed_volume_delta"] = g.buy_volume - g.sell_volume
    g["close_time"] = g.open_time + INTERVALS[interval] - 1
    g["available_at"] = g.close_time + 1
    return g


def merge_fragments(parts: list[pd.DataFrame], interval: str) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts, ignore_index=True)
    g = (
        d.groupby("open_time", sort=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            quote_volume=("quote_volume", "sum"),
            buy_volume=("buy_volume", "sum"),
            sell_volume=("sell_volume", "sum"),
            trade_count=("trade_count", "sum"),
            aggregate_trade_count=("aggregate_trade_count", "sum"),
        )
        .reset_index()
    )
    return finish_bars(g, interval)


def aggregate_bars(bars: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Exact associative OHLCV/flow aggregation; preserves first/last tick ordering."""
    if bars.empty:
        return bars.copy()
    child = bars.copy()
    child["open_time"] = child.open_time // INTERVALS[interval] * INTERVALS[interval]
    return merge_fragments([child], interval)


class IncrementalBars:
    """Finalize at event-time watermark; late events are counted and excluded."""

    def __init__(self, intervals=None, lag_ms=1000):
        self.intervals = intervals or list(INTERVALS)
        self.lag_ms = lag_ms
        self.pending = {tf: pd.DataFrame() for tf in self.intervals}
        self.watermark = -1
        self.late_events = 0
        self.last_id = -1

    def update(self, ticks: pd.DataFrame, observed_at: int) -> dict[str, pd.DataFrame]:
        fresh = ticks[(ticks.timestamp >= self.watermark) & (ticks.aggregate_id > self.last_id)].copy()
        self.late_events += int(
            ((ticks.timestamp < self.watermark) & (ticks.aggregate_id > self.last_id)).sum()
        )
        if len(ticks):
            self.last_id = max(self.last_id, int(ticks.aggregate_id.max()))
        cutoff = max(self.watermark, observed_at - self.lag_ms)
        output = {}
        for tf in self.intervals:
            partial = tick_bars(fresh, tf)
            fragments = [f for f in [self.pending[tf], partial] if not f.empty]
            bars = merge_fragments(fragments, tf)
            if not bars.empty:
                self.pending[tf] = bars[bars.available_at > cutoff].copy()
                completed = bars[(bars.available_at <= cutoff) & (bars.available_at > self.watermark)].copy()
                completed["available_at"] = observed_at
                output[tf] = completed
            else:
                output[tf] = bars
        self.watermark = cutoff
        return output
