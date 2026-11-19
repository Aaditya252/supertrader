import numpy as np
import pytest
gym = pytest.importorskip("gymnasium")
from backend.rl.environment import TradingEnv  # noqa: E402


def test_env_uses_future_open_not_prior_price_jump():
    env = TradingEnv(np.zeros((10, 3)), np.array([100, 100, 200, 200, 200, 200, 200, 200, 200, 200]),
                     window=2, cost=0, exposure_cap=1, randomize_costs=False)
    observation, _ = env.reset(seed=42)
    _, _, _, _, info = env.step(np.array([1.], dtype=np.float32))
    assert info["net_return"] == 0
    assert env.observation_space.contains(observation)


def test_env_seed_and_costs():
    env = TradingEnv(np.zeros((40, 3)), np.ones(40)*100, window=4, randomize_costs=False)
    a, _ = env.reset(seed=7)
    b, _ = env.reset(seed=7)
    np.testing.assert_array_equal(a, b)
    _, _, _, _, info = env.step(np.array([1.], dtype=np.float32))
    assert info["net_return"] < 0


def test_env_passes_stable_baselines_contract():
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.env_checker import check_env
    env = TradingEnv(np.zeros((40, 3)), np.ones(40)*100, window=4, randomize_costs=False)
    check_env(env)

