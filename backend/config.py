from pathlib import Path
from typing import Literal
import os
import yaml
from pydantic import BaseModel, Field, model_validator


class Settings(BaseModel):
    symbol: str = "SOLUSDT"
    data_source: Literal["binance"] = "binance"
    timeframe: str = "1h"
    market: Literal["spot", "futures"] = "spot"
    mode: Literal["signals", "paper", "live"] = "signals"
    history_start: str = "2020-01-01T00:00:00Z"
    data_dir: Path = Path("data")
    artifact_dir: Path = Path("artifacts")
    db_url: str = "sqlite:///data/supertrader.sqlite"
    api_base: str = "https://data-api.binance.vision"
    ws_base: str = "wss://stream.binance.com:9443/ws"
    poll_seconds: float = Field(30, ge=1)
    news_poll_seconds: float = Field(300, ge=30)
    retrain_days: int = Field(7, ge=1)
    retrain_hours: float | None = Field(None, ge=1)
    seed: int = 42
    label_horizon: int = Field(4, ge=1)
    fee_bps: float = Field(10, ge=0)
    slippage_bps: float = Field(5, ge=0)
    spread_bps: float = Field(2, ge=0)
    min_confidence: float = Field(.58, ge=.5, le=1)
    max_risk_per_trade: float = Field(.005, gt=0, le=.02)
    max_exposure: float = Field(.2, gt=0, le=1)
    target_annual_vol: float = Field(.15, gt=0)
    max_daily_loss: float = Field(.02, gt=0, le=.2)
    max_drawdown: float = Field(.1, gt=0, le=.5)
    max_spread_bps: float = Field(15, gt=0)
    max_bar_atr_fraction: float = Field(.08, gt=0)
    stale_seconds: float = Field(120, gt=0)
    paper_equity: float = Field(10000, gt=0)
    min_promotion_trades: int = Field(30, ge=1)
    drift_psi_threshold: float = Field(.25, gt=0)
    news_enabled: bool = False
    news_model: str = "ProsusAI/finbert"
    news_revision: str = "main"
    derivatives_enabled: bool = False
    optional_macro: bool = False
    optional_onchain: bool = False
    rss_urls: list[str] = []
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:3001"]

    @model_validator(mode="after")
    def safe_defaults(self):
        if self.symbol != "SOLUSDT":
            raise ValueError("Symbol must be SOLUSDT")
        if self.timeframe not in {"1m", "1h"}:
            raise ValueError("Serving/training supports 1m or 1h")
        if self.market != "spot":
            raise ValueError("Futures execution is not validated; spot only in this release")
        if self.mode == "live":
            raise ValueError("Live execution is intentionally locked in this release")
        if self.news_enabled and self.news_revision == "main":
            raise ValueError("Pin news_revision to a model commit before enabling sentiment")
        return self


def load_settings(path: str | None = None) -> Settings:
    file = Path(path or os.environ.get("SUPERTRADER_CONFIG") or "configs/default.yaml")
    raw = yaml.safe_load(file.read_text()) if file.exists() else {}
    for name in ["data_dir", "artifact_dir", "db_url"]:
        value = os.getenv(f"SUPERTRADER_{name.upper()}")
        if value:
            raw[name] = value
    if os.getenv("SUPERTRADER_LIVE_ENABLED", "false").lower() == "true":
        raise ValueError("Real orders require a separately reviewed execution implementation")
    return Settings(**raw)

