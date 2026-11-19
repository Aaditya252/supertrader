import numpy as np
import pandas as pd
import pytest
import torch
from stable_baselines3.common.env_checker import check_env
from backend.phase2.simulator import PerpetualEnv
from backend.phase2.network import DeepQuantAgent, CausalConv


def prices(n=40):
    return pd.DataFrame(
        {
            "open_time": np.arange(n) * 3600000,
            "open": np.ones(n) * 100,
            "high": np.ones(n) * 101,
            "low": np.ones(n) * 99,
            "close": np.ones(n) * 100,
        }
    )


def test_funding_sign_short_long_and_flat():
    b = prices()
    x = np.zeros((len(b), 2))
    funding = pd.DataFrame({"available_at": [24 * 3600000], "last_funding_rate": [0.01]})
    kwargs = dict(
        randomize=False,
        leverage_cap=1,
        fee_bps=0,
        spread_bps=0,
        slippage_bps=0,
        funding=funding,
    )
    long = PerpetualEnv(x, b, **kwargs)
    short = PerpetualEnv(x, b, **kwargs)
    long.reset()
    short.reset()
    long.units = 0.01
    long.position = 1.0
    short.units = -0.01
    short.position = -1.0
    _, _, _, _, long_result = long.step([1])
    _, _, _, _, s = short.step([-1])
    assert long_result["funding_cost"] == pytest.approx(0.01) and s["funding_cost"] == pytest.approx(-0.01)
    assert s["equity"] > long_result["equity"]
    check_env(PerpetualEnv(x, b, randomize=False), warn=True)


def test_agent_margin_liquidation_and_gap_rejection():
    b = prices()
    b.loc[24, "low"] = 40
    env = PerpetualEnv(np.zeros((40, 2)), b, leverage_cap=2, randomize=False)
    env.reset()
    _, _, done, _, info = env.step([1])
    assert done and info["liquidated"] and info["equity"] >= 0
    b.loc[30:, "open_time"] += 3600000
    with pytest.raises(ValueError, match="contiguous"):
        PerpetualEnv(np.zeros((40, 2)), b)


def test_causal_encoder_and_ordered_quantiles():
    torch.manual_seed(42)
    conv = CausalConv(2, 4)
    x = torch.randn(1, 2, 30)
    changed = x.clone()
    changed[:, :, 20:] += 100
    torch.testing.assert_close(conv(x)[:, :, :20], conv(changed)[:, :, :20])
    model = DeepQuantAgent(["30s", "1h"], 2)
    result = model({"30s": x.transpose(1, 2), "1h": x.transpose(1, 2)}, torch.zeros(1, 8))
    assert torch.all(result["quantiles"][:, 1:] >= result["quantiles"][:, :-1])
    assert result["attention"].sum().item() == pytest.approx(1)
