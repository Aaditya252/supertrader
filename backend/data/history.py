import asyncio
import json
from pathlib import Path
from typing import Any
import httpx
import pandas as pd

INTERVAL_MS = {"1m": 60000, "5m": 300000, "15m": 900000, "1h": 3600000,
               "4h": 14400000, "1d": 86400000}
COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time"]


def clean_candles(frame: pd.DataFrame, interval: str, now_ms: int) -> pd.DataFrame:
    step = INTERVAL_MS[interval]
    d = frame.copy().drop_duplicates("open_time", keep="last").sort_values("open_time")
    for c in ["open", "high", "low", "close", "volume"]:
        d[c] = pd.to_numeric(d[c], errors="raise")
    for c in ["open_time", "close_time"]:
        d[c] = pd.to_numeric(d[c], errors="raise").astype("int64")
    if ((d.open_time % step) != 0).any():
        raise ValueError("Candles are not aligned to UTC interval boundaries")
    expected_close = d.open_time + step - 1
    if ((d.close_time > expected_close) | (d.close_time < d.open_time)).any():
        raise ValueError("Unexpected candle close timestamp")
    # Historical maintenance can truncate a bar. Exclude it, leaving an auditable gap.
    d = d[d.close_time == expected_close].copy()
    if (d[["open", "high", "low", "close"]] <= 0).any().any() or (d.volume < 0).any():
        raise ValueError("Invalid candle price or volume")
    if ((d.high < d[["open", "close", "low"]].max(axis=1)) |
        (d.low > d[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("OHLC bounds are inconsistent")
    d = d[d.close_time < now_ms].reset_index(drop=True)
    d["available_at"] = d.close_time + 1
    d["availability_basis"] = "exchange_bar_close_assumed"
    return d


def gap_ranges(d: pd.DataFrame, interval: str) -> list[dict]:
    step = INTERVAL_MS[interval]
    times = d.open_time.to_numpy()
    return [{"start_ms": int(a + step), "end_ms": int(b - step),
             "missing_bars": int((b - a) // step - 1)}
            for a, b in zip(times[:-1], times[1:]) if b - a > step]


def candle_path(root: Path, symbol: str, interval: str) -> Path:
    return root / "candles" / "spot" / symbol / f"{interval}.parquet"


def save_candles(root: Path, symbol: str, interval: str, frame: pd.DataFrame, now_ms: int) -> dict:
    file = candle_path(root, symbol, interval)
    file.parent.mkdir(parents=True, exist_ok=True)
    # Preserve malformed historical timestamp rows for inspection, outside tradable data.
    raw_open = pd.to_numeric(frame.open_time).astype("int64")
    raw_close = pd.to_numeric(frame.close_time).astype("int64")
    invalid = ((raw_open % INTERVAL_MS[interval]) != 0) | (raw_close < raw_open) | (raw_close > raw_open+INTERVAL_MS[interval]-1)
    quarantine = file.with_suffix(".quarantine.json")
    anomalies = json.loads(quarantine.read_text()) if quarantine.exists() else {}
    for _, row in frame[invalid].iterrows():
        anomalies[str(int(row.open_time))] = {"reason": "invalid exchange timestamps", "row": row.to_dict()}
    if anomalies:
        quarantine.write_text(json.dumps(anomalies, indent=2, default=str))
    frame = frame[~invalid].copy()
    if file.exists():
        frame = pd.concat([pd.read_parquet(file), frame], ignore_index=True)
    d = clean_candles(frame, interval, now_ms)
    temp = file.with_suffix(".tmp.parquet")
    d.to_parquet(temp, index=False)
    temp.replace(file)
    manifest = {"symbol": symbol, "market": "spot", "interval": interval, "rows": len(d),
                "first_ms": int(d.open_time.iloc[0]) if len(d) else None,
                "last_ms": int(d.close_time.iloc[-1]) if len(d) else None,
                "gaps": gap_ranges(d, interval), "observed_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "quarantined_rows": len(anomalies),
                "availability_basis": "historical exchange close, not verified receipt latency"}
    file.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


class BinanceREST:
    def __init__(self, base: str = "https://data-api.binance.vision"):
        self.base = base.rstrip("/")

    async def get(self, path: str, params: dict | None = None) -> Any:
        async with httpx.AsyncClient(timeout=30) as client:
            for attempt in range(6):
                try:
                    response = await client.get(self.base + path, params=params)
                    if response.status_code in (418, 429):
                        await asyncio.sleep(min(float(response.headers.get("Retry-After", 2 ** attempt)), 60))
                        continue
                    response.raise_for_status()
                    return response.json()
                except (httpx.TransportError, httpx.HTTPStatusError):
                    if attempt == 5:
                        raise
                    await asyncio.sleep(min(2 ** attempt, 30))
        raise RuntimeError("Exchange rate limit retries exhausted")

    async def now_ms(self) -> int:
        return int((await self.get("/api/v3/time"))["serverTime"])

    async def candles(self, symbol: str, interval: str, start_ms: int | None = None,
                      end_ms: int | None = None, limit: int = 1000) -> pd.DataFrame:
        params: dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_ms is not None:
            params["startTime"] = start_ms
        if end_ms is not None:
            params["endTime"] = end_ms
        rows = await self.get("/api/v3/klines", params)
        return pd.DataFrame([r[:7] for r in rows], columns=COLUMNS)


async def download_history(root: Path, symbol: str, interval: str, start: str,
                           base: str, use_ccxt: bool = False) -> dict:
    client = BinanceREST(base)
    now = await client.now_ms()
    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    file = candle_path(root, symbol, interval)
    if file.exists():
        existing = pd.read_parquet(file)
        cursor = max(start_ms, int(existing.open_time.max()) - INTERVAL_MS[interval])
    else:
        cursor = start_ms
    exchange = None
    if use_ccxt:
        import ccxt.async_support as ccxt
        exchange = ccxt.binance({"enableRateLimit": True})
    try:
        while cursor < now:
            if exchange:
                values = await exchange.fetch_ohlcv(symbol.replace("USDT", "/USDT"), interval,
                                                     since=cursor, limit=1000)
                page = pd.DataFrame([v + [v[0] + INTERVAL_MS[interval] - 1] for v in values],
                                    columns=COLUMNS)
            else:
                page = await client.candles(symbol, interval, cursor, now)
            if page.empty:
                break
            last_open = int(page.open_time.max())
            if last_open < cursor:
                raise RuntimeError("Pagination failed to advance")
            save_candles(root, symbol, interval, page, now)
            cursor = last_open + INTERVAL_MS[interval]
            await asyncio.sleep(.12)
        # Repair internal gaps, without fabricating bars when the exchange has no data.
        if not file.exists():
            raise RuntimeError("No historical bars returned")
        d = pd.read_parquet(file)
        for gap in gap_ranges(d, interval):
            repair_cursor = gap["start_ms"]
            while repair_cursor <= gap["end_ms"]:
                page = await client.candles(symbol, interval, repair_cursor,
                                           gap["end_ms"] + INTERVAL_MS[interval] - 1)
                if page.empty:
                    break
                save_candles(root, symbol, interval, page, now)
                repair_cursor = int(page.open_time.max()) + INTERVAL_MS[interval]
        return json.loads(file.with_suffix(".manifest.json").read_text())
    finally:
        if exchange:
            await exchange.close()


def resample_candles(d: pd.DataFrame, source: str, target: str, now_ms: int) -> pd.DataFrame:
    factor = INTERVAL_MS[target] // INTERVAL_MS[source]
    if factor < 1 or INTERVAL_MS[target] % INTERVAL_MS[source]:
        raise ValueError("Target must be a whole multiple of source")
    frame = d.copy()
    frame["bucket"] = (frame.open_time // INTERVAL_MS[target]) * INTERVAL_MS[target]
    groups = frame.groupby("bucket", sort=True)
    agg = groups.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                     close=("close", "last"), volume=("volume", "sum"), count=("open_time", "nunique"))
    agg = agg[agg["count"] == factor].drop(columns="count").reset_index().rename(columns={"bucket": "open_time"})
    agg["close_time"] = agg.open_time + INTERVAL_MS[target] - 1
    return clean_candles(agg, target, now_ms)

