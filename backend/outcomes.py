import pandas as pd
from backend.store import Store
from backend.features.labels import triple_barrier


def evaluate_matured(store: Store, features: pd.DataFrame, horizon: int, costs: float,
                     interval_ms: int) -> int:
    labels = triple_barrier(features, horizon, cost_fraction=costs, interval_ms=interval_ms)
    lookup = {int(t): i for i, t in enumerate(features.available_at)}
    evaluated = {x["signal_id"] for x in store.list("outcome", 100000)}
    count = 0
    for signal in store.list("signal", 10000):
        if signal["id"] in evaluated or not signal.get("probabilities"):
            continue
        i = lookup.get(signal["decision_time"])
        if i is None or pd.isna(labels.label.iloc[i]):
            continue
        outcome = {"signal_id": signal["id"], "version": signal["model_version"],
                   "decision_time": signal["decision_time"], "resolved_at": int(labels.label_end.iloc[i]),
                   "label": int(labels.label.iloc[i]), "net_barrier_return": float(labels.forward_return.iloc[i]),
                   "probabilities": signal["probabilities"], "regime": signal["regime"],
                   "action": signal["action"], "reason": "matured prediction; includes abstentions",
                   "not_trade_pnl": True}
        count += store.append("outcome", "outcome:"+signal["id"], outcome)
    return count

