import numpy as np
import pandas as pd
import pytest
from backend.phase2.dataset import aligned_windows, fit_scale, cpcv, chronological_split, MODEL_FEATURES
from backend.phase2.bars import normalize_ticks
from backend.phase2.simulator import PerpetualEnv


def test_backward_completed_bar_join_and_gap_rejection():
    times = np.arange(40) * 3600000
    frame = pd.DataFrame({k: np.arange(40, dtype=float) for k in MODEL_FEATURES})
    frame["open_time"] = times
    frame["available_at"] = times + 3600000
    windows, valid = aligned_windows({"1h": frame}, np.array([times[25] + 100]), window=24)
    assert valid[0]
    assert windows["1h"][0, -1, 0] == 24
    frame.loc[20, "open_time"] += 1000
    assert not aligned_windows({"1h": frame}, np.array([times[25] + 100]), window=24)[1][0]


def test_train_scaler_does_not_observe_future_and_cpcv_purges_overlap():
    x = np.arange(100 * 24 * 2, dtype=np.float32).reshape(100, 24, 2)
    _, scalers = fit_scale({"1h": x}, np.arange(50))
    altered = x.copy()
    altered[50:] *= 1000
    assert fit_scale({"1h": altered}, np.arange(50))[1] == scalers
    times = np.arange(120) * 3600000
    ends = times + 14400000
    folds = list(cpcv(times, ends))
    assert len(folds) == 15
    for train, test in folds:
        assert not set(train) & set(test)
        for i in test:
            assert not ((times[train] <= ends[i]) & (ends[train] >= times[i])).any()
    daily = np.arange(365 * 6) * 86400000
    train, calibration, holdout = chronological_split(daily)
    assert daily[train[-1]] + 14400000 < daily[calibration[0]]
    assert daily[calibration[-1]] + 14400000 < daily[holdout[0]]


def test_archive_interleaving_preserves_source_time_and_strict_live_contract():
    raw = pd.DataFrame(
        [
            [1, 100.0, 2.0, 1, 1, 1000, False],
            [3, 102.0, 2.0, 3, 3, 3000, True],
            [2, 101.0, 2.0, 2, 2, 2000, False],
        ]
    )
    with pytest.raises(ValueError):
        normalize_ticks(raw)
    frame, _ = normalize_ticks(raw, enforce_order=False)
    assert frame.timestamp.tolist() == [1000, 3000, 2000]
    repaired = frame.sort_values(["timestamp", "aggregate_id"])
    assert normalize_ticks(repaired)[0].aggregate_id.tolist() == [1, 2, 3]


def test_funding_at_entry_charges_only_preexisting_units():
    times = np.arange(40) * 3600000
    bars = pd.DataFrame(
        {"open_time": times, "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "quote_volume": 1e9}
    )
    funding = pd.DataFrame({"available_at": [times[24]], "last_funding_rate": [0.01]})
    env = PerpetualEnv(np.zeros((40, 1), dtype=np.float32), bars, funding=funding, randomize=False)
    env.reset()
    info = env.step(np.array([0.5]))[-1]
    assert info["funding_cost"] == 0
    env.reset()
    env.units = 0.01
    env.position = 1
    info = env.step(np.array([0.0]))[-1]
    assert info["funding_cost"] == pytest.approx(0.01)
