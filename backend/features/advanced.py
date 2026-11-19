"""Optional expensive features; fits must be repeated strictly within historical cutoffs."""
import numpy as np
import pandas as pd
from backend.features.engine import causal_join


def garch_forecast(past_returns: pd.Series, horizon: int = 4) -> float:
    from arch import arch_model
    if len(past_returns.dropna()) < 250:
        return float("nan")
    fit = arch_model(past_returns.dropna().tail(2000)*100, vol="GARCH", p=1, q=1,
                     mean="Zero", rescale=False).fit(disp="off", show_warning=False)
    variance = fit.forecast(horizon=horizon, reindex=False).variance.iloc[-1].sum()
    return float(np.sqrt(variance)/100)


def relative_momentum(sol: pd.DataFrame, btc: pd.DataFrame, eth: pd.DataFrame) -> pd.DataFrame:
    d = causal_join(sol, btc, ["return_24"], "btc_")
    d = causal_join(d, eth, ["return_24"], "eth_")
    d["relative_btc"] = d.return_24-d.btc_return_24
    d["relative_eth"] = d.return_24-d.eth_return_24
    return d


def microstructure_series(observations: list[dict]) -> pd.DataFrame:
    """Aggregate only observed valid records; absent book observations remain missing."""
    d = pd.DataFrame(observations)
    if d.empty:
        return d
    required = ["available_at", "book_valid", "imbalance", "spread_bps", "depth_usdt"]
    for column in required:
        if column not in d:
            d[column] = np.nan
    d = d[d.book_valid == True].copy()  # noqa: E712
    d["depth_change"] = d.depth_usdt.pct_change(fill_method=None)
    return d[required+["depth_change"]].sort_values("available_at")

