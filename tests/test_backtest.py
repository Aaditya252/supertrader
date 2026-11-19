import numpy as np
import pandas as pd
import pytest
from backend.backtest.engine import simulate, Costs, metrics


def frame(opens, closes, low=None, high=None):
    n = len(opens)
    return pd.DataFrame({"open_time": np.arange(n)*3600000, "close_time": np.arange(n)*3600000+3599999,
                         "open": opens, "close": closes, "low": low or closes, "high": high or closes,
                         "atr": np.ones(n)*2, "regime": ["range"]*n})


def test_next_open_fill_does_not_capture_previous_jump():
    d = frame([100, 200, 200], [100, 200, 200])
    result = simulate(d, np.ones(3), Costs(0, 0, 0), stop_atr=None)
    assert result["equity"][-1] == pytest.approx(1.)


def test_roundtrip_costs_are_charged():
    d = frame([100, 100, 100], [100, 100, 100])
    result = simulate(d, np.ones(3), Costs(10, 0, 0), stop_atr=None)
    assert result["equity"][-1] == pytest.approx((1-.001)/(1+.001))


def test_stop_wins_when_both_barriers_touch():
    d = frame([100, 100, 100], [100, 100, 100], [100, 95, 100], [100, 110, 100])
    result = simulate(d, np.array([1, 0, 0]), Costs(0, 0, 0), stop_atr=1.5, take_atr=3)
    assert result["equity"][-1] == pytest.approx(.97)
    assert result["trades"][0]["reason"] == "stop"


def test_gap_through_stop_fills_at_open():
    d = frame([100, 100, 90], [100, 100, 90], [100, 99, 89], [100, 101, 91])
    result = simulate(d, np.ones(3), Costs(0, 0, 0))
    assert result["equity"][-1] == pytest.approx(.9)


def test_flat_position_is_flat():
    d = frame([100, 100, 300], [100, 100, 300])
    result = simulate(d, np.zeros(3))
    assert result["equity"] == [1., 1.]


def test_spot_cannot_short():
    with pytest.raises(ValueError, match="Spot"):
        simulate(frame([100, 100, 100], [100, 100, 100]), np.array([-1, 0, 0]))


def test_drawdown_includes_starting_equity():
    result = metrics(np.array([-.1, 0.]))
    assert result["max_drawdown"] == pytest.approx(-.1)

