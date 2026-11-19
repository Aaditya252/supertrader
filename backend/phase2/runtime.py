"""Shared feature orchestration, bounded live bars and explicit historical coverage."""

import asyncio
import copy
import json
import time
import httpx
import numpy as np
import pandas as pd
from backend.phase2.bars import INTERVALS, IncrementalBars, normalize_ticks, aggregate_bars
from backend.phase2.ta import features, report, confluence
from backend.phase2.quant import diagnostics, book_features, ofi, liquidation_proxy
from backend.phase2.interpret import explanation
from backend.phase2.dataset import read_bars


def state_vector(hour, derivatives=None, news=None):
    from backend.phase2.quant import volatility_features

    v = volatility_features(hour.close).iloc[-1]
    d = derivatives or {}
    news = news or {}
    flow = (
        float(hour.signed_volume_delta.iloc[-1] / hour.volume.iloc[-1])
        if "signed_volume_delta" in hour and hour.volume.iloc[-1] > 0
        else None
    )
    values = [
        v.rv,
        v.bipower_variation,
        flow,
        d.get("funding_rate"),
        d.get("oi_change"),
        news.get("sentiment"),
        float(flow is not None),
        float(d.get("funding_rate") is not None),
        float(d.get("oi_change") is not None),
        float(news.get("sentiment") is not None),
    ]
    return np.array([float(x) if x is not None and np.isfinite(x) else 0.0 for x in values], dtype=np.float32)


def safe_json(value):
    if isinstance(value, dict):
        return {k: safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class Phase2Runtime:
    def __init__(self, settings, market, store):
        self.settings, self.market, self.store = settings, market, store
        self.root = settings.data_dir / "phase2"
        self.frames = {tf: pd.DataFrame() for tf in INTERVALS}
        self.builder = IncrementalBars(lag_ms=1500)
        self.last_id = -1
        self.previous_book = None
        self.derivatives = {}
        self.snapshot = {"status": "warming_up"}
        self.last_refresh = 0.0
        self.last_features = 0.0
        self.last_derivative = 0.0
        self.error = None
        self.oi_observations = []
        self.capture_start = int(time.time() * 1000)
        self.ta_cache = {}
        self.quant_cache = None
        self.quant_key = None

    async def rest_bars(self, tf, client):
        now = int(time.time() * 1000)
        if tf == "30s":
            values = []
            cursor = now - 120 * 60000
            for _ in range(8):
                r = await client.get(
                    self.settings.api_base + "/api/v3/klines",
                    params={"symbol": "SOLUSDT", "interval": "1s", "startTime": cursor, "limit": 1000},
                )
                r.raise_for_status()
                page = r.json()
                if not page:
                    break
                values.extend(page)
                cursor = int(page[-1][0]) + 1000
                if cursor >= now:
                    break
            child = self.kline_frame(values, 1000, now)
            if child.empty:
                return child
            child["aggregate_trade_count"] = np.nan
            return aggregate_bars(child, "30s").query("available_at <= @now")
        r = await client.get(
            self.settings.api_base + "/api/v3/klines",
            params={"symbol": "SOLUSDT", "interval": tf, "limit": 650},
        )
        r.raise_for_status()
        return self.kline_frame(r.json(), INTERVALS[tf], now)

    @staticmethod
    def kline_frame(values, step, now):
        if not values:
            return pd.DataFrame()
        d = pd.DataFrame(
            values,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trade_count",
                "buy_volume",
                "buy_quote_volume",
                "ignore",
            ],
        )
        d = d.apply(pd.to_numeric)
        d = d[(d.close_time == d.open_time + step - 1) & (d.close_time < now)].copy()
        d["sell_volume"] = d.volume - d.buy_volume
        d["signed_volume_delta"] = d.buy_volume - d.sell_volume
        d["vwap"] = d.quote_volume / d.volume.replace(0, np.nan)
        d["available_at"] = d.close_time + 1
        return d

    async def refresh_context(self):
        async with httpx.AsyncClient(timeout=30) as client:

            async def one(tf):
                existing = await asyncio.to_thread(read_bars, self.root, "spot", tf, 650)
                if (
                    existing.empty
                    or time.time() * 1000 - int(existing.available_at.iloc[-1]) > INTERVALS[tf] * 2
                ):
                    existing = await self.rest_bars(tf, client)
                    source = "native exchange OHLCV context; live ticks share flow definitions"
                else:
                    source = "checksum-verified aggregate-trade bars"
                if not existing.empty:
                    existing["source_basis"] = source
                    current = self.frames[tf]
                    # Keep tick observations when they cover the same bar.
                    self.frames[tf] = (
                        pd.concat([existing, current], ignore_index=True)
                        .sort_values("open_time")
                        .drop_duplicates("open_time", keep="last")
                        .tail(700)
                        .reset_index(drop=True)
                    )

            results = await asyncio.gather(*(one(tf) for tf in INTERVALS), return_exceptions=True)
            for tf, result in zip(INTERVALS, results):
                if isinstance(result, Exception):
                    self.error = f"{tf} context: {type(result).__name__}"
        self.last_refresh = time.time()

    async def derivative_context(self):
        now = int(time.time() * 1000)
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                mark, oi = await asyncio.gather(
                    client.get("https://fapi.binance.com/fapi/v1/premiumIndex", params={"symbol": "SOLUSDT"}),
                    client.get("https://fapi.binance.com/fapi/v1/openInterest", params={"symbol": "SOLUSDT"}),
                )
                mark.raise_for_status()
                oi.raise_for_status()
                m, o = mark.json(), oi.json()
            current = float(o["openInterest"])
            eligible = [o for o in self.oi_observations if o[0] <= now - 300000]
            prior = eligible[-1][1] if eligible and now - eligible[-1][0] < 360000 else None
            self.oi_observations.append((now, current))
            self.oi_observations = self.oi_observations[-200:]
            funding = float(m["lastFundingRate"])
            funding_z = None
            archive = self.root / "aux/fundingRate/SOLUSDT/history.parquet"
            if archive.exists():
                history = pd.read_parquet(archive)
                history = history[history.available_at <= now].tail(100)
                if len(history) > 30 and history.last_funding_rate.std() > 1e-10:
                    funding_z = float(
                        (funding - history.last_funding_rate.mean()) / history.last_funding_rate.std()
                    )
            self.derivatives = {
                "available_at": now,
                "mark_price": float(m["markPrice"]),
                "index_price": float(m["indexPrice"]),
                "funding_rate": funding,
                "funding_z": funding_z,
                "open_interest": current,
                "oi_notional": current * float(m["markPrice"]),
                "oi_change": current / prior - 1 if prior else None,
                "oi_change_basis": "backward as-of five-minute OI change; unavailable during warmup",
                "basis": float(m["markPrice"]) / float(m["indexPrice"]) - 1,
                "basis_definition": "perpetual mark versus exchange spot index",
                "long_short_ratios": None,
                "liquidation_events": None,
                "source": "public live futures REST; observed availability",
            }
        except (httpx.HTTPError, ValueError, KeyError):
            if self.derivatives and now - self.derivatives.get("available_at", 0) > 120000:
                self.derivatives = {"status": "stale derivatives; inputs excluded"}
        self.last_derivative = time.time()

    def consume(self):
        events = [e for e in getattr(self.market, "aggregate_events", []) if e["a"] > self.last_id]
        if events:
            raw = pd.DataFrame([[e["a"], e["p"], e["q"], e["f"], e["l"], e["T"], e["m"]] for e in events])
            ticks, _ = normalize_ticks(raw)
            self.last_id = int(ticks.aggregate_id.max())
        else:
            ticks = pd.DataFrame(
                columns=[
                    "aggregate_id",
                    "timestamp",
                    "price",
                    "quantity",
                    "quote_volume",
                    "buy_volume",
                    "sell_volume",
                    "trade_count",
                ]
            )
        completed = self.builder.update(ticks, int(time.time() * 1000))
        for tf, bars in completed.items():
            if bars.empty:
                continue
            # A connection started inside a bar cannot claim complete tick coverage.
            first_complete = (self.capture_start // INTERVALS[tf] + 1) * INTERVALS[tf]
            bars = bars[bars.open_time >= first_complete].copy()
            bars["source_basis"] = "prospective live aggregate trades"
            self.frames[tf] = (
                pd.concat([self.frames[tf], bars], ignore_index=True)
                .sort_values("open_time")
                .drop_duplicates("open_time", keep="last")
                .tail(700)
                .reset_index(drop=True)
            )

    def coverage(self):
        result = {}
        for name in ["coverage", "auxiliary-coverage"]:
            file = self.root / (name + ".json")
            if file.exists():
                d = json.loads(file.read_text())
                archives = d.get("archives", [])
                result[name] = {
                    "status": d["status"],
                    "processed": len(archives),
                    "expected": d.get("expected_archives"),
                    "unavailable_periods": sum(a["status"] == "unavailable" for a in archives),
                    "aggregate_trade_rows": sum(a.get("rows", 0) for a in archives),
                    "error": d.get("error"),
                }
        return result

    def compute(self):
        started = time.perf_counter()
        ta = {}
        for tf, bars in self.frames.items():
            if not bars.empty:
                key = (int(bars.open_time.iloc[-1]), float(bars.close.iloc[-1]), float(bars.volume.iloc[-1]))
                if tf not in self.ta_cache or self.ta_cache[tf][0] != key:
                    self.ta_cache[tf] = (key, report(features(bars, tf), tf))
                ta[tf] = copy.deepcopy(self.ta_cache[tf][1])
                ta[tf]["source"] = str(bars.source_basis.iloc[-1])
                age = time.time() * 1000 - int(bars.available_at.iloc[-1])
                ta[tf]["age_ms"] = age
                if age > INTERVALS[tf] * 2:
                    ta[tf]["status"] = "stale"
        ta_output = {"timeframes": ta, "confluence": confluence(ta)}
        hour = self.frames["1h"]
        if not hour.empty:
            key = (int(hour.open_time.iloc[-1]), float(hour.close.iloc[-1]), float(hour.volume.iloc[-1]))
            if key != self.quant_key:
                self.quant_cache = diagnostics(hour)
                self.quant_key = key
            quant = copy.deepcopy(self.quant_cache or {})
            quant["derivatives"] = self.derivatives.copy()
            volume_z = (hour.volume.iloc[-1] - hour.volume.tail(100).mean()) / max(
                hour.volume.tail(100).std(), 1e-12
            )
            atr = ta.get("1h", {}).get("features", {}).get("atr", 0)
            quant["liquidation"] = liquidation_proxy(
                float(hour.close.iloc[-1]),
                atr,
                self.derivatives.get("oi_change"),
                float(volume_z),
                self.derivatives.get("funding_z"),
                self.derivatives.get("oi_notional"),
                float(hour.close.iloc[-1] / hour.close.iloc[-2] - 1),
            )
        else:
            quant = {"status": "warming_up"}
        book = self.market.book
        micro = (
            book_features(book.bids, book.asks)
            if book.valid and book.bridged
            else {"status": "unavailable; sequenced book not valid"}
        )
        if micro.get("bid") is not None:
            if self.previous_book:
                micro["ofi"] = ofi(self.previous_book, micro)
            self.previous_book = micro.copy()
        quant["book"] = micro
        model = {
            "status": "not trained",
            "approved": False,
            "uncertainty": None,
            "attention": {},
            "attributions": {},
        }
        file = self.settings.artifact_dir / "phase2/latest.json"
        if file.exists():
            model.update(json.loads(file.read_text()))
        elapsed = (time.perf_counter() - started) * 1000
        self.snapshot = safe_json(
            {
                "status": "research_only",
                "generated_at": int(time.time() * 1000),
                "ta": ta_output,
                "quant": quant,
                "model": model,
                "explanation": explanation(ta_output, quant, model),
                "coverage": self.coverage(),
                "feeds": {
                    "trade_capture": self.market.connected,
                    "late_events": self.builder.late_events,
                    "feed_age_seconds": self.market.status().get("age_seconds"),
                    "feature_compute_ms": elapsed,
                    "latency_target_ms": 1000,
                    "target_met_for_feature_compute": elapsed < 1000,
                    "latency_scope": "feature computation only; not measured end-to-end market delivery",
                },
                "error": self.error,
                "live_execution": "locked",
                "liquidation_label": "estimated proxy; not observed liquidation history",
            }
        )
        self.last_features = time.time()

    async def run(self):
        while True:
            try:
                if time.time() - self.last_refresh > 300:
                    await self.refresh_context()
                if time.time() - self.last_derivative > 30:
                    await self.derivative_context()
                self.consume()
                if time.time() - self.last_features > 30:
                    await asyncio.to_thread(self.compute)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {str(exc)[:120]}"
            await asyncio.sleep(1)

    def candles(self, interval, limit=500):
        d = self.frames[interval].tail(limit)
        return [
            {
                "time": int(r.open_time // 1000),
                "open": float(r.open),
                "high": float(r.high),
                "low": float(r.low),
                "close": float(r.close),
                "volume": float(r.volume),
            }
            for _, r in d.iterrows()
        ]
