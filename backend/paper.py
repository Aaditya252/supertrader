"""Closed-bar paper simulator; signal is applied at the next bar open, never retroactively."""
from backend.store import Store
from backend.config import Settings
from backend.backtest.engine import Costs


class PaperPortfolio:
    def __init__(self, store: Store, settings: Settings):
        self.store, self.settings = store, settings

    def state(self) -> dict:
        return self.store.latest("paper") or {"cash": self.settings.paper_equity, "units": 0.,
                "equity": self.settings.paper_equity, "peak": self.settings.paper_equity,
                "daily_open_equity": self.settings.paper_equity, "day": "", "daily_return": 0.,
                "drawdown": 0., "last_bar": 0, "stop": None, "take": None,
                "mode": self.settings.mode, "simulation": "next-bar-open with conservative OHLC exits"}

    def on_bar(self, row: dict, signal: dict | None) -> dict:
        state = self.state()
        if self.settings.mode != "paper" or row["open_time"] <= state["last_bar"]:
            return state
        import pandas as pd
        day = pd.to_datetime(row["open_time"], unit="ms", utc=True).strftime("%Y-%m-%d")
        cash, units = state["cash"], state["units"]
        opening = cash + units*row["open"]
        if state["day"] != day:
            state["daily_open_equity"] = opening
        cost_rate = Costs(self.settings.fee_bps, self.settings.slippage_bps, self.settings.spread_bps).one_way
        executed = []
        blocked = state["daily_return"] <= -self.settings.max_daily_loss or state["drawdown"] <= -self.settings.max_drawdown
        # New-day risk uses the new equity baseline.
        if state["day"] != day:
            blocked = state["drawdown"] <= -self.settings.max_drawdown
        if signal and signal["decision_time"] <= row["open_time"] < signal["expires_at"] and not blocked:
            action = signal["action"]
            if action in ("BUY", "SELL"):
                target = min(signal["target_exposure"], self.settings.max_exposure) if action == "BUY" else 0.
                desired = target*opening/row["open"]
                delta = desired-units
                if delta > 0:
                    delta = min(delta, cash/(row["open"]*(1+cost_rate)))
                notional = delta*row["open"]
                cash -= notional + abs(notional)*cost_rate
                units += delta
                if abs(delta) > 1e-10:
                    executed.append({"time": row["open_time"], "side": "BUY" if delta > 0 else "SELL",
                                     "price": row["open"], "units": abs(delta), "fee_and_slippage": abs(notional)*cost_rate,
                                     "signal_id": signal["id"], "reason": "signal"})
                if action == "BUY" and units > 0:
                    state["stop"], state["take"] = signal["stop_loss"], signal["take_profit"]
                else:
                    state["stop"] = state["take"] = None
        exit_price, reason = None, None
        if units > 0:
            if state["stop"] is not None and row["low"] <= state["stop"]:
                exit_price, reason = min(row["open"], state["stop"]), "stop"
            elif state["take"] is not None and row["high"] >= state["take"]:
                exit_price, reason = max(row["open"], state["take"]), "target"
            elif blocked:
                exit_price, reason = row["open"], "kill switch"
        if exit_price:
            proceeds = units*exit_price
            cash += proceeds*(1-cost_rate)
            executed.append({"time": row["close_time"], "side": "SELL", "price": exit_price,
                             "units": units, "fee_and_slippage": proceeds*cost_rate, "reason": reason})
            units = 0.
            state["stop"] = state["take"] = None
        equity = cash+units*row["close"]
        peak = max(state["peak"], equity)
        state.update(cash=cash, units=units, equity=equity, peak=peak, day=day,
                     daily_return=equity/state["daily_open_equity"]-1, drawdown=equity/peak-1,
                     last_bar=row["open_time"], mode="paper")
        for i, trade in enumerate(executed):
            self.store.append("paper_trade", f'paper-trade:{row["open_time"]}:{i}', trade)
        self.store.append("paper", f'paper:{row["open_time"]}', state)
        return state

