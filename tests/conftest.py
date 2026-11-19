import numpy as np
import pandas as pd
import pytest
from backend.data.history import clean_candles


@pytest.fixture
def candles():
    rng = np.random.default_rng(7)
    n = 450
    close = 100*np.exp(np.cumsum(rng.normal(.0002, .005, n)))
    times = np.arange(n)*3600000
    d = pd.DataFrame({"open_time": times, "open": close*.999, "high": close*1.01,
                      "low": close*.99, "close": close, "volume": rng.uniform(1000, 3000, n),
                      "close_time": times+3599999})
    return clean_candles(d, "1h", int(times[-1]+7200000))

