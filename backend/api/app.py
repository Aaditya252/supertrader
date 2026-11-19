import asyncio
import json
import os
import secrets
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Header, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from backend.config import load_settings, Settings
from backend.service import SignalService
from backend.store import utcnow
from backend.strategies import StrategyFinder
from typing import Literal


class Feedback(BaseModel):
    signal_id: str = Field(min_length=1, max_length=100)
    note: str = Field(min_length=1, max_length=2000)


def create_app(settings: Settings | None = None, online: bool = True) -> FastAPI:
    config = settings or load_settings()
    service = SignalService(config, online)
    finder = StrategyFinder()

    @asynccontextmanager
    async def lifespan(app):
        await service.start()
        yield
        await service.stop()

    app = FastAPI(title="Supertrader", version="0.1.0", lifespan=lifespan)
    app.state.service = service
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Admin-Token"],
    )

    @app.get("/health")
    def health():
        return service.health()

    @app.get("/mlops/status")
    def mlops_status():
        from backend.mlops.pipeline import pipeline_status

        return pipeline_status(config)

    @app.get("/orderflow")
    def orderflow():
        return service.market.orderflow_snapshot()

    @app.get("/strategies/discover")
    async def strategies_discover(
        q: str = Query("SOL momentum", min_length=2, max_length=100, pattern=r"^[A-Za-z0-9 ._-]+$"),
        source: Literal["github", "arxiv"] = "github",
    ):
        if not q.strip() or not any(c.isalnum() for c in q):
            raise HTTPException(422, "Enter a strategy topic")
        return await finder.search(q.strip(), source)

    @app.get("/strategies/comparison")
    def strategies_comparison():
        pointer = config.artifact_dir / "latest.json"
        if not pointer.exists():
            return {"status": "not_run", "rows": []}
        import re
        version = json.loads(pointer.read_text()).get("version", "")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", version):
            raise HTTPException(422, "Invalid artifact version")
        file = config.artifact_dir / version / "report.json"
        if not file.is_file():
            return {"status": "not_run", "rows": []}
        report = json.loads(file.read_text())
        rows = [{"name": "Candidate", "metrics": report["candidate"]["metrics"]}]
        rows += [{"name": name, "metrics": value["metrics"]}
                 for name, value in report.get("baselines", {}).items()]
        return {"status": "ready", "version": version, "approved": report.get("approved", False),
                "rows": rows, "holdout_first": report.get("holdout_first"),
                "holdout_last": report.get("holdout_last"),
                "promotion_failures": report.get("promotion_failures", []),
                "scope": "Existing local experiment only; discovered web candidates are untested"}

    @app.get("/signal/latest")
    def latest():
        return service.latest()

    @app.get("/signals/history")
    def history(limit: int = Query(100, ge=1, le=1000)):
        return service.store.list("signal", limit)

    @app.get("/candles")
    def candles(limit: int = Query(500, ge=1, le=1000)):
        if service.candles.empty:
            return []
        rows = service.candles.tail(limit)
        return [
            {
                "time": int(r.open_time // 1000),
                "open": float(r.open),
                "high": float(r.high),
                "low": float(r.low),
                "close": float(r.close),
                "volume": float(r.volume),
            }
            for _, r in rows.iterrows()
        ]

    @app.get("/features/importance")
    def importance():
        signal = service.latest()
        return {"drivers": signal.get("drivers", []), "basis": signal.get("driver_basis", "unavailable")}

    @app.get("/phase2/overview")
    def phase2_overview():
        return service.phase2.snapshot

    @app.get("/phase2/candles")
    def phase2_candles(interval: str = "1h", limit: int = Query(500, ge=1, le=1000)):
        from backend.phase2.bars import INTERVALS

        if interval not in INTERVALS:
            raise HTTPException(422, "Unsupported Phase 2 timeframe")
        return service.phase2.candles(interval, limit)

    @app.get("/phase2/validation")
    def phase2_validation():
        file = config.artifact_dir / "phase2/validation.json"
        return json.loads(file.read_text()) if file.exists() else {"status": "not_run", "approved": False}

    @app.get('/phase2/training')
    def phase2_training():
        file=config.artifact_dir/'phase2/job.json'
        result=json.loads(file.read_text()) if file.exists() else {'status':'not_started'}
        coverage=config.data_dir/'phase2/coverage.json'
        if coverage.exists():
            data=json.loads(coverage.read_text())
            result['dataset']={'status':data['status'],'processed_archives':len(data.get('archives',[])),
                               'expected_archives':data.get('expected_archives')}
        pilot=config.artifact_dir/'phase2/pilot/validation.json'
        if pilot.exists():
            data=json.loads(pilot.read_text())
            result['pilot']={'version':data['version'],'scope':data['protocol']['scope'],
                             'metrics':{name:value['metrics'] for name,value in data['results'].items()}}
        return result

    @app.get("/backtest/results")
    def backtest():
        pointer = config.artifact_dir / "latest.json"
        if not pointer.exists():
            return {"status": "not_run", "message": "No backtest has been run"}
        version = json.loads(pointer.read_text())["version"]
        return json.loads((config.artifact_dir / version / "report.json").read_text())

    @app.get("/news/feed")
    def news(limit: int = Query(50, ge=1, le=200)):
        return service.store.list("news", limit)

    @app.get("/paper/portfolio")
    def portfolio():
        return {
            **service.paper.state(),
            "trades": service.store.list("paper_trade", 200),
            "curve": service.store.list("paper", 500, ascending=True),
        }

    @app.get("/outcomes")
    def outcomes(limit: int = Query(100, ge=1, le=1000)):
        return service.store.list("outcome", limit)

    @app.post("/feedback")
    def feedback(body: Feedback, x_admin_token: str | None = Header(default=None)):
        expected = os.getenv("SUPERTRADER_ADMIN_TOKEN", "")
        if not expected or not secrets.compare_digest(x_admin_token or "", expected):
            raise HTTPException(401, "Valid admin token required")
        key = "feedback:" + secrets.token_hex(12)
        service.store.append(
            "feedback",
            key,
            {**body.model_dump(), "at": utcnow(), "status": "review note; not a market training label"},
        )
        return {"saved": True}

    @app.websocket("/ws")
    async def stream(ws: WebSocket):
        origin = ws.headers.get("origin")
        if origin and origin not in config.cors_origins:
            await ws.close(code=1008)
            return
        await ws.accept()
        sequence = 0
        try:
            while True:
                sequence += 1
                await ws.send_json(
                    {"sequence": sequence, "signal": service.latest(), "health": service.health()}
                )
                await asyncio.sleep(3)
        except (WebSocketDisconnect, RuntimeError):
            pass

    return app
