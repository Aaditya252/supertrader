import time
import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend.api.app import create_app
from backend.config import Settings
from backend.store import Store
from backend.risk import position_size, risk_reasons
from backend.models.ensemble import vote
from backend.paper import PaperPortfolio
from backend.monitoring import population_stability


def test_live_mode_is_locked():
    with pytest.raises(ValueError, match="locked"):
        Settings(mode="live")


def test_position_caps():
    settings = Settings()
    size = position_size(100, 10, .01, settings)
    assert size <= settings.max_exposure
    assert size*1.5*10/100 <= settings.max_risk_per_trade+1e-10
    assert position_size(0, 1, .01, settings) == 0


def test_risk_blocks_invalid_book_and_kill_switch():
    reasons = risk_reasons(Settings(), {"book_valid": False, "connected": False}, 1000, .01,
                           daily_return=-.03, drawdown=-.15)
    assert "daily loss kill switch" in reasons
    assert "drawdown kill switch" in reasons
    assert "live order book unavailable" in reasons


def test_ensemble_abstains_on_disagreement():
    result = vote([{"up": .8, "down": .1, "flat": .1}, {"up": .1, "down": .8, "flat": .1}])
    assert result["action"] == "HOLD"
    assert result["probability"] is None


def test_store_idempotency_and_json():
    store = Store("sqlite:///:memory:")
    assert store.append("signal", "a", {"action": "HOLD"})
    assert not store.append("signal", "a", {"action": "BUY"})
    assert store.latest("signal")["action"] == "HOLD"
    with pytest.raises(ValueError):
        store.append("signal", "b", {"value": float("nan")})


def test_api_offline_reports_untrained_and_protects_feedback(tmp_path):
    app = create_app(Settings(data_dir=tmp_path, artifact_dir=tmp_path/"models", db_url="sqlite:///:memory:"), online=False)
    with TestClient(app) as client:
        assert client.get("/health").json()["live_execution"] == "locked"
        assert client.get("/signal/latest").json()["action"] == "HOLD"
        assert client.get("/backtest/results").json()["status"] == "not_run"
        assert client.get("/candles").json() == []
        assert client.get("/signals/history?limit=10001").status_code == 422
        assert client.post("/feedback", json={"signal_id": "x", "note": "bad prediction"}).status_code == 401


def test_expired_signal_cannot_stay_buy(tmp_path):
    app = create_app(Settings(data_dir=tmp_path, db_url="sqlite:///:memory:"), online=False)
    service = app.state.service
    service.last_fetch = time.time()
    service.store.append("signal", "x", {"id": "x", "action": "BUY", "status": "ready",
                                         "expires_at": int(time.time()*1000)-1, "risk_overrides": []})
    assert service.latest()["action"] == "HOLD"
    assert service.latest()["status"] == "stale"
    # View-time staleness does not rewrite the audit event.
    assert service.store.latest("signal")["action"] == "BUY"


def test_current_risk_view_recovers_without_rewriting_audit(tmp_path):
    app = create_app(Settings(data_dir=tmp_path, db_url="sqlite:///:memory:"), online=False)
    service = app.state.service
    service.last_fetch = time.time()
    blocked = {"id": "x", "action": "HOLD", "status": "blocked",
               "expires_at": int(time.time()*1000)+3600000,
               "risk_overrides": ["live order book unavailable"]}
    service.store.append("signal", "x", blocked)
    service.current_signal = {**blocked, "status": "research_only", "risk_overrides": []}
    assert service.latest()["status"] == "research_only"
    assert service.latest()["risk_overrides"] == []
    assert service.store.latest("signal")["status"] == "blocked"
    service.current_signal["expires_at"] = 0
    assert service.latest()["status"] == "stale"
    assert service.current_signal["status"] == "research_only"


def test_paper_cannot_backdate_signal():
    store = Store("sqlite:///:memory:")
    paper = PaperPortfolio(store, Settings(mode="paper"))
    row = {"open_time": 3600000, "close_time": 7199999, "open": 100., "high": 101., "low": 99., "close": 100.}
    signal = {"id": "future", "decision_time": 7200000, "expires_at": 10800000, "action": "BUY",
              "target_exposure": .2, "stop_loss": 95., "take_profit": 110.}
    result = paper.on_bar(row, signal)
    assert result["units"] == 0
    assert not store.list("paper_trade")


def test_drift_detects_shift():
    rng = np.random.default_rng(1)
    reference = rng.normal(size=1000)
    assert population_stability(reference, reference) == pytest.approx(0)
    assert population_stability(reference, reference+5) > .25

