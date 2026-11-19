import asyncio
import json
import logging
import time
from contextlib import suppress
import pandas as pd
from backend.config import Settings
from backend.store import Store, utcnow
from backend.data.history import BinanceREST, candle_path, save_candles, clean_candles, INTERVAL_MS
from backend.data.live import LiveMarket
from backend.data.news import NewsIngestor, rolling_sentiment
from backend.features.engine import build_features, FEATURE_COLUMNS
from backend.models.train import load_bundle
from backend.monitoring import drift_report, outcome_summary
from backend.outcomes import evaluate_matured
from backend.backtest.engine import Costs
from backend.paper import PaperPortfolio
from backend.signals import generate_signal

log = logging.getLogger(__name__)


class SignalService:
    def __init__(self, settings: Settings, online: bool = True):
        self.settings, self.online = settings, online
        self.store = Store(settings.db_url)
        self.rest = BinanceREST(settings.api_base)
        self.market = LiveMarket(settings.symbol, self.rest, settings.ws_base)
        self.paper = PaperPortfolio(self.store, settings)
        self.news = NewsIngestor(
            self.store, settings.rss_urls, settings.news_enabled, settings.news_model, settings.news_revision
        )
        self.tasks: list[asyncio.Task] = []
        self.candles = pd.DataFrame()
        self.last_fetch = 0.0
        self.error: str | None = None
        self.drift: dict = {}
        self.bundle: dict | None = None
        self.current_signal: dict | None = None
        from backend.phase2.runtime import Phase2Runtime

        self.phase2 = Phase2Runtime(settings, self.market, self.store)
        self.artifact_mtime = 0.0

    async def start(self):
        file = candle_path(self.settings.data_dir, self.settings.symbol, self.settings.timeframe)
        if file.exists():
            self.candles = pd.read_parquet(file).tail(1000).reset_index(drop=True)
        if self.online:
            self.tasks = [
                asyncio.create_task(self.market.run()),
                asyncio.create_task(self.loop()),
                asyncio.create_task(self.news_loop()),
                asyncio.create_task(self.retrain_loop()),
                asyncio.create_task(self.phase2.run()),
            ]
            if self.settings.derivatives_enabled:
                from backend.data.derivatives import DerivativesIngestor

                derivatives = DerivativesIngestor(self.settings.symbol, self.store)
                self.tasks.extend(
                    [
                        asyncio.create_task(derivatives.poll_open_interest()),
                        asyncio.create_task(derivatives.streams()),
                    ]
                )

    async def stop(self):
        for task in self.tasks:
            task.cancel()
        for task in self.tasks:
            with suppress(asyncio.CancelledError):
                await task

    def refresh_bundle(self):
        # Prefer an approved champion; expose latest research candidate if there is none.
        root = self.settings.artifact_dir
        pointer = root / ("champion.json" if (root / "champion.json").exists() else "latest.json")
        if pointer.exists() and pointer.stat().st_mtime != self.artifact_mtime:
            self.bundle = load_bundle(root, champion_only=pointer.name == "champion.json")
            self.artifact_mtime = pointer.stat().st_mtime

    async def tick(self):
        now = await self.rest.now_ms()
        step = INTERVAL_MS[self.settings.timeframe]
        start = int(self.candles.open_time.iloc[-1]) if len(self.candles) else None
        new = await self.rest.candles(self.settings.symbol, self.settings.timeframe, start_ms=start)
        final = clean_candles(new, self.settings.timeframe, now)
        if final.empty:
            return
        await asyncio.to_thread(
            save_candles, self.settings.data_dir, self.settings.symbol, self.settings.timeframe, final, now
        )
        merged = (
            pd.concat([self.candles, final])
            .drop_duplicates("open_time", keep="last")
            .sort_values("open_time")
        )
        self.candles = merged.tail(1000).reset_index(drop=True)
        self.last_fetch = time.time()
        self.error = None
        self.refresh_bundle()
        features = build_features(self.candles, step)
        if self.settings.mode == "paper":
            previous = self.store.list("signal", 200)
            for _, bar in final.iterrows():
                eligible = [s for s in previous if s["decision_time"] <= bar.open_time < s["expires_at"]]
                prior = max(eligible, key=lambda s: s["generated_at"]) if eligible else None
                self.paper.on_bar(bar.to_dict(), prior)
        news = rolling_sentiment(self.store.list("news", 200), int(features.available_at.iloc[-1]))
        latest_close = int(self.candles.close_time.iloc[-1]) + 1
        # Age beyond one expected interval is stale, even if polling itself succeeds.
        age = max(time.time() - self.last_fetch, max(0.0, (now - latest_close - step) / 1000))
        signal = generate_signal(
            features, self.bundle, self.settings, self.market.status(), age, self.paper.state(), news
        )
        self.store.append("signal", "signal:" + signal["id"], signal)
        # Repeated audit keys remain immutable, but current risk checks must refresh.
        self.current_signal = signal
        # Preserve order-book/trade observations for future enriched-model research.
        self.store.append(
            "microstructure",
            f"micro:{now}",
            {"available_at": now, "symbol": self.settings.symbol, **self.market.status()},
        )
        cost = 2 * Costs(self.settings.fee_bps, self.settings.slippage_bps, self.settings.spread_bps).one_way
        await asyncio.to_thread(
            evaluate_matured, self.store, features, self.settings.label_horizon, cost, step
        )
        if self.bundle:
            self.drift = drift_report(
                self.bundle["reference"],
                features[FEATURE_COLUMNS].tail(168),
                self.settings.drift_psi_threshold,
            )
        await self.maybe_rollback()

    async def maybe_rollback(self):
        outcomes = self.store.list("outcome", 500)
        if not self.bundle or not self.bundle.get("approved"):
            return
        current = [o for o in outcomes if o["version"] == self.bundle["version"]]
        if len(current) < 100:
            return
        live = outcome_summary(current)
        report_file = self.settings.artifact_dir / self.bundle["version"] / "report.json"
        reference = json.loads(report_file.read_text())["calibration"]["multiclass_brier"]
        if live["brier"] <= reference * 1.5:
            return
        previous = self.settings.artifact_dir / "previous_champion.json"
        champion = self.settings.artifact_dir / "champion.json"
        if previous.exists():
            prev = json.loads(previous.read_text())["version"]
            artifact = self.settings.artifact_dir / prev / "report.json"
            if artifact.exists() and json.loads(artifact.read_text()).get("approved"):
                champion.write_text(previous.read_text())
                self.store.append(
                    "incident",
                    "rollback:" + self.bundle["version"],
                    {
                        "at": utcnow(),
                        "reason": "rolling Brier exceeds reference by 50%",
                        "from": self.bundle["version"],
                        "to": prev,
                        "outcomes": len(current),
                    },
                )
                self.refresh_bundle()
                return
        # No validated fallback: disable actionable signals rather than pretend rollback succeeded.
        self.bundle["approved"] = False
        (self.settings.artifact_dir / ("suspended-" + self.bundle["version"] + ".json")).write_text(
            json.dumps({"reason": "live calibration degraded", "at": utcnow()})
        )
        self.store.append(
            "incident",
            "disable:" + self.bundle["version"],
            {"at": utcnow(), "reason": "calibration degraded; no compatible approved fallback"},
        )

    async def loop(self):
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.error = str(exc)[:250]
                log.exception("Signal update failed")
            await asyncio.sleep(self.settings.poll_seconds)

    async def news_loop(self):
        while True:
            try:
                await self.news.poll()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("News unavailable: %s", type(exc).__name__)
            await asyncio.sleep(self.settings.news_poll_seconds)

    async def retrain_loop(self):
        while True:
            interval = (self.settings.retrain_hours * 3600 if self.settings.retrain_hours is not None
                        else self.settings.retrain_days * 86400)
            await asyncio.sleep(interval)
            # Separate subprocess keeps training and CPU load outside the event loop.
            import sys

            config_path = self.settings.artifact_dir / "retrain-settings.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(self.settings.model_dump_json(), encoding="utf-8")
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "backend.cli",
                "--config",
                str(config_path.resolve()),
                "mlops",
                "run",
            )
            try:
                code = await process.wait()
            except asyncio.CancelledError:
                if process.returncode is None:
                    process.terminate()
                    await process.wait()
                raise
            self.store.append(
                "training_job", f"training:{time.time_ns()}", {"at": utcnow(), "exit_code": code}
            )

    def latest(self) -> dict:
        signal = dict(self.current_signal) if self.current_signal is not None else self.store.latest("signal")
        now = int(time.time() * 1000)
        if signal is None:
            return {
                "action": "HOLD",
                "status": "warming_up",
                "symbol": self.settings.symbol,
                "confidence": None,
                "reasons": ["waiting for market data"],
                "drivers": [],
                "risk_overrides": [],
                "price": float(self.candles.close.iloc[-1]) if len(self.candles) else None,
            }
        if now >= signal["expires_at"] or time.time() - self.last_fetch > self.settings.stale_seconds:
            signal.update(
                action="HOLD",
                status="stale",
                direction=None,
                direction_confidence=None,
                suggested_size_pct=0.0,
                stop_loss=None,
                take_profit=None,
                risk_reward=None,
            )
            signal["risk_overrides"] = list(dict.fromkeys(signal["risk_overrides"] + ["data expired"]))
        return signal

    def health(self) -> dict:
        return {
            "status": "ready" if self.last_fetch and not self.error else "degraded",
            "error": self.error,
            "market": self.market.status(),
            "candle_count": len(self.candles),
            "last_fetch": self.last_fetch or None,
            "mode": self.settings.mode,
            "live_execution": "locked",
            "model": self.bundle["version"] if self.bundle else None,
            "model_approved": bool(self.bundle and self.bundle["approved"]),
            "drift": self.drift,
            "learning": outcome_summary(self.store.list("outcome", 500)),
            "incidents": self.store.list("incident", 20),
        }
