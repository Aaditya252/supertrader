import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TradingEnv(gym.Env):
    """Spot long/flat research environment. Negative actions map to flat, not synthetic shorts.

    Feature row t is known at decision; reward is next-open to following-open,
    eliminating the impossible close-price fill used by many toy environments.
    """
    metadata = {"render_modes": []}

    def __init__(self, features: np.ndarray, opens: np.ndarray, window: int = 24,
                 cost: float = .0016, exposure_cap: float = .2, randomize_costs: bool = True):
        super().__init__()
        self.features = np.asarray(features, dtype=np.float32)
        self.opens = np.asarray(opens, dtype=np.float64)
        self.window, self.base_cost, self.cap = window, cost, exposure_cap
        self.randomize = randomize_costs
        if len(features) != len(opens) or len(features) < window+3 or not np.isfinite(features).all():
            raise ValueError("Invalid feature/price coverage")
        self.action_space = spaces.Box(-1., 1., shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(window*features.shape[1]+4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.i = self.window-1
        self.position, self.equity, self.peak, self.time_in_trade = 0., 1., 1., 0
        self.entry_equity = 1.
        self.cost = self.base_cost*(self.np_random.uniform(.75, 2.) if self.randomize else 1.)
        self.mean, self.second = 0., 1e-6
        return self._observation(), {}

    def _observation(self):
        x = self.features[self.i-self.window+1:self.i+1].flatten().copy()
        if self.randomize:
            x += self.np_random.normal(0, .002, x.shape).astype(np.float32)
        state = [self.position, self.equity/self.entry_equity-1 if self.position else 0,
                 self.time_in_trade/100, self.equity/self.peak-1]
        return np.r_[x, state].astype(np.float32)

    def step(self, action):
        a = float(np.clip(np.asarray(action).reshape(-1)[0], -1, 1))
        position = max(a, 0)*self.cap
        turnover = abs(position-self.position)
        entry, exit_price = self.opens[self.i+1], self.opens[self.i+2]
        net = position*(exit_price/entry-1)-turnover*self.cost
        if self.i+1 >= len(self.features)-2:
            net -= position*self.cost
            turnover += position
        self.equity *= 1+net
        self.peak = max(self.peak, self.equity)
        drawdown = self.equity/self.peak-1
        # Differential Sharpe update uses only past reward moments.
        eta = .01
        delta_mean, delta_second = net-self.mean, net*net-self.second
        variance = max(self.second-self.mean*self.mean, 1e-6)
        differential = (self.second*delta_mean-.5*self.mean*delta_second)/variance**1.5
        reward = float(np.clip(differential, -10, 10)-.1*turnover-.5*abs(drawdown))
        self.mean += eta*delta_mean
        self.second += eta*delta_second
        if position and not self.position:
            self.entry_equity = self.equity
        self.time_in_trade = self.time_in_trade+1 if position else 0
        self.position = position
        self.i += 1
        terminated = bool(self.i >= len(self.features)-2 or drawdown <= -.1)
        return self._observation(), reward, terminated, False, {"equity": self.equity,
                 "net_return": net, "turnover": turnover, "drawdown": drawdown, "exposure": position}

