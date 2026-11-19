"""Perpetual research simulator; never an exchange execution adapter."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from backend.phase2.quant import impact_cost


class PerpetualEnv(gym.Env):
    """Next-open target exposure, observed funding, adverse intrabar margin liquidation.

    Equity multiplier=1+exposure*(Pnext/Pfill-1)-costs-exposure*funding.
    Minute/bar data cannot prove millisecond latency fills; latency is conservative
    extra slippage. Liquidation is maintenance-triggered with a declared fee.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features,
        bars,
        window=24,
        leverage_cap=2.0,
        fee_bps=4.0,
        spread_bps=2.0,
        slippage_bps=5.0,
        maintenance=0.005,
        liquidation_fee=0.005,
        funding=None,
        randomize=True,
        drawdown_stop=0.3,
        latency_ms=250,
        episode_bars=None,
    ):
        self.x = np.asarray(features, dtype=np.float32)
        self.bars = bars.reset_index(drop=True)
        self.window, self.cap = window, leverage_cap
        self.fee, self.spread, self.slippage = (
            fee_bps / 10000,
            spread_bps,
            slippage_bps / 10000,
        )
        self.maintenance, self.liquidation_fee = maintenance, liquidation_fee
        self.funding = funding
        self.randomize = randomize
        self.drawdown_stop = drawdown_stop
        self.latency_ms = latency_ms
        self.episode_bars = episode_bars
        if len(self.x) != len(bars) or len(bars) < window + 3 or not np.isfinite(self.x).all():
            raise ValueError("Finite aligned features/price bars required")
        if not 0 < leverage_cap <= 5 or not 0 < maintenance < 0.1:
            raise ValueError("Invalid research leverage/margin cap")
        if (self.bars[["open", "high", "low", "close"]] <= 0).any().any():
            raise ValueError("Invalid price")
        gaps = self.bars.open_time.diff().dropna()
        if not gaps.empty and gaps.nunique() != 1:
            raise ValueError("Episodes must be contiguous; split at missing bars")
        self.action_space = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (window * self.x.shape[1] + 5,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.i = self.window - 1
        if self.episode_bars is not None:
            max_start = max(
                self.window - 1, len(self.x) - min(self.episode_bars, len(self.x) - self.window - 2) - 2
            )
            self.i = int(self.np_random.integers(self.window - 1, max_start + 1))
        self.episode_end = (
            min(len(self.x) - 2, self.i + self.episode_bars)
            if self.episode_bars is not None
            else len(self.x) - 2
        )
        self.equity = self.peak = 1.0
        self.position = 0.0
        self.units = 0.0
        self.age = 0
        self.mean = 0.0
        self.second = 1e-6
        self.returns = []
        self.turnover = 0.0
        self.cost_multiplier = float(self.np_random.uniform(0.75, 2.0)) if self.randomize else 1.0
        return self._observation(), {}

    def _observation(self):
        x = self.x[self.i - self.window + 1 : self.i + 1].ravel().copy()
        if self.randomize:
            x += self.np_random.normal(0, 0.002, x.shape).astype(np.float32)
        return np.r_[
            x,
            [
                self.position / self.cap,
                self.equity - 1,
                self.age / 100,
                self.equity / self.peak - 1,
                self.mean,
            ],
        ].astype(np.float32)

    def step(self, action):
        target = float(np.clip(np.asarray(action).ravel()[0], -1, 1)) * self.cap
        entry = self.bars.iloc[self.i + 1]
        following = self.bars.iloc[self.i + 2]
        fill = float(entry.open)
        exit_ = float(following.open)
        old_equity = self.equity
        funding_old = funding_new = 0.0
        if self.funding is not None and len(self.funding):
            rates = self.funding[
                (self.funding.available_at >= entry.open_time)
                & (self.funding.available_at < following.open_time)
            ]
            at_open = rates[rates.available_at == entry.open_time]
            later = rates[rates.available_at > entry.open_time]
            # Funding at the boundary belongs to the position held before the fill.
            funding_old = self.units * fill * float(at_open.last_funding_rate.sum())
            funded_equity = max(0.0, old_equity - funding_old)
            units = target * funded_equity / fill
            funding_new = units * fill * float(later.last_funding_rate.sum())
        else:
            units = target * old_equity / fill
        exposure = units * fill / old_equity
        turnover = abs(units - self.units) * fill / old_equity
        participation = min(
            0.1,
            abs(target) * 10000 / max(float(self.bars.iloc[self.i].get("quote_volume", 1e6)), 1.0),
        )
        latency_cost = self.slippage * max(0, self.latency_ms) / 1000
        one_way = (
            self.fee + self.slippage + impact_cost(participation, target, self.spread) + latency_cost
        ) * self.cost_multiplier
        cost = turnover * one_way
        funding_cost = (funding_old + funding_new) / old_equity
        adverse = float(entry.low) / fill - 1 if target > 0 else float(entry.high) / fill - 1
        adverse_equity = 1 - cost - funding_cost + exposure * adverse
        maintenance_requirement = self.maintenance * abs(exposure) * (1 + adverse)
        liquidated = bool(target != 0 and adverse_equity <= maintenance_requirement)
        if liquidated:
            # Residual collateral at maintenance minus liquidation fee, capped at zero.
            net = (
                max(
                    0.0,
                    maintenance_requirement - self.liquidation_fee * abs(exposure) * (1 + adverse),
                )
                - 1
            )
            turnover += abs(exposure) * (1 + adverse)
            target = 0.0
            units = 0.0
        else:
            net = exposure * (exit_ / fill - 1) - cost - funding_cost
        terminal = bool(self.i + 1 >= self.episode_end)
        kill = (old_equity * max(0.0, 1 + net) / max(self.peak, old_equity) - 1) <= -self.drawdown_stop
        if (terminal or kill) and units:
            closing_notional = abs(units) * exit_ / old_equity
            net -= closing_notional * one_way
            turnover += closing_notional
            units = 0.0
        self.equity *= max(0.0, 1 + net)
        self.peak = max(self.peak, self.equity)
        dd = self.equity / self.peak - 1
        self.returns.append(net)
        tail = np.asarray(self.returns[-100:])
        threshold = np.quantile(tail, 0.05)
        cvar = -float(tail[tail <= threshold].mean())
        dm, ds = net - self.mean, net * net - self.second
        variance = max(self.second - self.mean * self.mean, 1e-6)
        differential = (self.second * dm - 0.5 * self.mean * ds) / variance**1.5
        reward = float(np.clip(differential, -10, 10) - 0.05 * turnover - 0.5 * abs(dd) - max(0, cvar))
        self.mean += 0.01 * dm
        self.second += 0.01 * ds
        self.age = self.age + 1 if target else 0
        self.units = units
        self.position = units * exit_ / self.equity if self.equity > 0 else 0.0
        self.turnover += turnover
        self.i += 1
        terminated = bool(terminal or liquidated or dd <= -self.drawdown_stop or self.equity <= 0)
        return (
            self._observation(),
            reward,
            terminated,
            False,
            {
                "equity": self.equity,
                "net_return": net,
                "drawdown": dd,
                "turnover": turnover,
                "exposure": exposure,
                "funding_cost": funding_cost,
                "funding_basis": "position before boundary fill; entry mark approximation for intrabar funding",
                "liquidated": liquidated,
                "execution_basis": "next open plus assumed latency/impact slippage",
            },
        )
