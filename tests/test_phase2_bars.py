import pandas as pd
import numpy as np
from backend.phase2.bars import (
    normalize_ticks,
    tick_bars,
    merge_fragments,
    IncrementalBars,
    aggregate_bars,
)


def ticks():
    raw = pd.DataFrame(
        [
            [1, 100, 2, 1, 2, 1000, False],
            [2, 110, 1, 3, 3, 15000, True],
            [3, 105, 3, 4, 6, 31000, False],
            [4, 108, 1, 7, 7, 45000, True],
        ]
    )
    return normalize_ticks(raw)[0]


def test_tick_flow_and_chunk_invariance():
    d = ticks()
    expected = tick_bars(d, "30s")
    actual = merge_fragments([tick_bars(d.iloc[:1], "30s"), tick_bars(d.iloc[1:], "30s")], "30s")
    pd.testing.assert_frame_equal(expected, actual)
    first = expected.iloc[0]
    assert first.open == 100 and first.close == 110 and first.high == 110
    assert first.buy_volume == 2 and first.sell_volume == 1 and first.trade_count == 3
    assert first.vwap == 310 / 3
    assert first.available_at == 30000
    pd.testing.assert_frame_equal(aggregate_bars(expected, "1m"), tick_bars(d, "1m"))


def test_live_and_historical_bar_definitions_match():
    d = ticks()
    live = IncrementalBars(["30s"], lag_ms=0)
    first = live.update(d.iloc[:2], 30000)["30s"]
    second = live.update(d.iloc[2:], 60000)["30s"]
    actual = pd.concat([first, second], ignore_index=True)
    pd.testing.assert_frame_equal(actual, tick_bars(d, "30s"))
    live.update(d.iloc[:1], 90000)
    assert all(frame.empty for frame in live.pending.values())


def test_microsecond_and_invalid_tick_handling():
    raw = pd.DataFrame(
        [
            [1, 100, 2, 1, 1, 1735689600000000, "true"],
            [2, -1, 2, 2, 2, 1735689600001000, "false"],
        ]
    )
    d, q = normalize_ticks(raw)
    assert d.timestamp.iloc[0] == 1735689600000 and q["invalid_rows"] == 1
    assert np.isfinite(d.quote_volume).all()
