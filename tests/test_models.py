import numpy as np
import pytest
from backend.features.engine import build_features
from backend.signals import generate_signal
from backend.config import Settings


class ConstantModel:
    classes_ = np.array([-1, 0, 1])
    def predict_proba(self, x):
        return np.tile([.05, .05, .9], (len(x), 1))


def test_unapproved_candidate_cannot_buy(candles):
    d = build_features(candles)
    bundle = {"version": "test", "approved": False, "model": ConstantModel(), "base": ConstantModel()}
    market = {"connected": True, "book_valid": True, "age_seconds": 0, "spread_bps": 1}
    signal = generate_signal(d, bundle, Settings(), market, 0,
                              {"equity": 10000, "units": 0}, {"score": None})
    assert signal["action"] == "HOLD"
    assert signal["suggested_size_pct"] == 0
    assert signal["status"] == "research_only"


def test_minute_prediction_has_minute_expiry_and_preserves_gates(candles):
    minute = candles.copy()
    for column in ["open_time", "close_time", "available_at"]:
        minute[column] = ((minute[column] + (1 if column == "close_time" else 0)) // 60
                          - (1 if column == "close_time" else 0))
    d = build_features(minute, 60000)
    settings = Settings(timeframe="1m", label_horizon=1)
    bundle = {"version": "minute", "approved": False,
              "model": ConstantModel(), "base": ConstantModel()}
    market = {"connected": True, "book_valid": True, "age_seconds": 0, "spread_bps": 1}
    signal = generate_signal(d, bundle, settings, market, 0,
                             {"equity": 10000, "units": 0}, {})
    assert signal["expires_at"] - signal["decision_time"] == 60000
    assert signal["direction"] == "up"
    assert signal["direction_confidence"] == .9
    assert signal["horizon_minutes"] == 1
    assert signal["action"] == "HOLD"


def test_approved_model_requires_risk_clearance(candles):
    d = build_features(candles)
    bundle = {"version": "test", "approved": True, "model": ConstantModel(), "base": ConstantModel()}
    market = {"connected": True, "book_valid": True, "age_seconds": 0, "spread_bps": 1}
    good = generate_signal(d, bundle, Settings(), market, 0,
                           {"equity": 10000, "units": 0}, {"score": None})
    assert good["action"] == "BUY"
    assert good["stop_loss"] < good["price"] < good["take_profit"]
    assert good["suggested_size_pct"] <= 20
    blocked = generate_signal(d, bundle, Settings(), market, 1000,
                              {"equity": 10000, "units": 0}, {"score": None})
    assert blocked["action"] == "HOLD" and blocked["stop_loss"] is None
    bundle["settings"] = {"fee_bps": 1}
    changed_costs = generate_signal(d, bundle, Settings(), market, 0,
                                   {"equity": 10000, "units": 0}, {"score": None})
    assert changed_costs["action"] == "HOLD"
    assert "configuration differs from validated model" in changed_costs["reasons"]


def test_hmm_filter_is_future_invariant():
    pytest.importorskip("hmmlearn")
    from backend.models.regime import CausalHMM
    rng = np.random.default_rng(4)
    train = rng.normal(size=(500, 2))
    x = rng.normal(size=(100, 2))
    model = CausalHMM().fit(train)
    a = model.filter(x)
    x[60:] += 100
    b = model.filter(x)
    np.testing.assert_allclose(a[:60], b[:60])
    np.testing.assert_allclose(a.sum(axis=1), 1)


def test_outcomes_wait_for_maturity_and_are_idempotent(candles):
    from backend.store import Store
    from backend.outcomes import evaluate_matured
    d = build_features(candles)
    store = Store("sqlite:///:memory:")
    i = 300
    store.append("signal", "s", {"id": "s", "decision_time": int(d.available_at.iloc[i]),
                                  "model_version": "test", "regime": "range", "action": "HOLD",
                                  "probabilities": {"down": .2, "flat": .6, "up": .2}})
    assert evaluate_matured(store, d.iloc[:i+2], 4, .0032, 3600000) == 0
    assert evaluate_matured(store, d.iloc[:i+5], 4, .0032, 3600000) == 1
    assert evaluate_matured(store, d, 4, .0032, 3600000) == 0
    assert store.latest("outcome")["not_trade_pnl"]

