import math
from backend.config import Settings
from backend.data.history import INTERVAL_MS


def position_size(price: float, atr: float, hourly_vol: float, settings: Settings,
                  edge: float | None = None, variance: float | None = None) -> float:
    if min(price, atr, hourly_vol) <= 0 or not all(math.isfinite(x) for x in [price, atr, hourly_vol]):
        return 0.
    risk_cap = settings.max_risk_per_trade/(1.5*atr/price)
    annual_bars = 365 * 86400000 / INTERVAL_MS[settings.timeframe]
    vol_cap = settings.target_annual_vol/(hourly_vol*math.sqrt(annual_bars))
    caps = [settings.max_exposure, risk_cap, vol_cap]
    if edge is not None and variance is not None:
        # Optional quarter-Kelly; caller must supply validated net-return moments.
        caps.append(max(0., .25*edge/max(variance, 1e-9)))
    return max(0., min(caps))


def risk_reasons(settings: Settings, market: dict, data_age: float | None,
                 atr_fraction: float, daily_return: float = 0, drawdown: float = 0,
                 gap: bool = False) -> list[str]:
    reasons = []
    if data_age is None or data_age > settings.stale_seconds:
        reasons.append("stale candle ingestion")
    if gap:
        reasons.append("recent candle gap or incomplete warmup")
    if not market.get("book_valid") or not market.get("connected"):
        reasons.append("live order book unavailable")
    age = market.get("age_seconds")
    if age is None or age > settings.stale_seconds:
        reasons.append("stale market stream")
    if market.get("spread_bps", float("inf")) > settings.max_spread_bps:
        reasons.append("spread exceeds limit")
    if atr_fraction > settings.max_bar_atr_fraction:
        reasons.append("extreme volatility")
    if daily_return <= -settings.max_daily_loss:
        reasons.append("daily loss kill switch")
    if drawdown <= -settings.max_drawdown:
        reasons.append("drawdown kill switch")
    return reasons

