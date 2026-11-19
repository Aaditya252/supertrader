import numpy as np
import pandas as pd
import pytest
from backend.phase2.ta import features, report, confluence
from backend.phase2.quant import (
    volatility_features,
    kalman_trend,
    dynamic_beta,
    evt_tail,
    ofi,
    book_features,
    structure,
    uniqueness_weights,
    liquidation_proxy,
    har_forecast,
)


def bars(n=400):
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame(
        {
            "open_time": np.arange(n) * 3600000,
            "available_at": (np.arange(n) + 1) * 3600000,
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.uniform(10, 100, n),
        }
    )


def test_ta_prefix_and_future_perturbation_invariance():
    d = bars()
    a = features(d, "1h")
    changed = d.copy()
    changed.loc[250:, ["open", "high", "low", "close"]] *= 4
    b = features(changed, "1h")
    pd.testing.assert_frame_equal(a.iloc[:250], b.iloc[:250])
    pd.testing.assert_frame_equal(a.iloc[:250], features(d.iloc[:250], "1h"))
    assert "profile_poc" in a and "bearish_rsi_divergence" in a
    assert report(a, "1h")["status"] == "ready"
    assert confluence({"1h": report(a, "1h")})["score"] == pytest.approx(a.ta_score.iloc[-1])


def test_gap_restarts_ta_warmup():
    d = bars(300)
    d.loc[200:, "open_time"] += 3600000
    d.loc[200:, "available_at"] += 3600000
    a = features(d, "1h")
    assert np.isnan(a.ema_50.iloc[220])
    assert not np.isnan(a.ema_50.iloc[270])


def test_quant_causal_state_and_constant_series():
    d = bars()
    a = volatility_features(d.close)
    pd.testing.assert_frame_equal(a.iloc[:200], volatility_features(d.close.iloc[:200]))
    np.testing.assert_allclose(kalman_trend(d.close)[:200], kalman_trend(d.close.iloc[:200]))
    r = np.log(d.close).diff().fillna(0).to_numpy()
    np.testing.assert_allclose(dynamic_beta(r, r)[:200], dynamic_beta(r[:200], r[:200]))
    assert structure(np.ones(300))["hurst"] is None
    assert evt_tail(np.zeros(500))["var"] is None
    assert har_forecast(pd.Series(np.arange(80, dtype=float)))["forecast"] >= 0


def test_book_signs_and_missing_liquidation_inputs():
    p = {"bid": 100, "ask": 101, "bid_size": 5, "ask_size": 8}
    q = {**p, "bid_size": 7, "ask_size": 6}
    assert ofi(p, q) == 4
    assert book_features({100: 5}, {101: 8})["microprice"] == pytest.approx((101 * 5 + 100 * 8) / 13)
    assert liquidation_proxy(100, 2)["pressure"] is None
    proxy = liquidation_proxy(100, 2, -0.05, 3, 2, 1000000)
    assert proxy["estimated"] and len(proxy["clusters"]) == 8
    assert all(c["estimated_notional"] > 0 for c in proxy["clusters"])


def test_uniqueness_weights_exact_overlap():
    np.testing.assert_allclose(uniqueness_weights([0, 1], [1, 2]), [0.75, 0.75])
    with pytest.raises(ValueError):
        uniqueness_weights([2], [1])
