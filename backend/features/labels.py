import numpy as np
import pandas as pd


def triple_barrier(d: pd.DataFrame, horizon: int = 4, atr_multiple: float = 1.5,
                   cost_fraction: float = .0032, interval_ms: int = 3600000) -> pd.DataFrame:
    """Decision after bar i, entry at i+1 open. Both touched => conservative lower barrier."""
    result = pd.DataFrame(index=d.index, columns=["label", "forward_return", "label_end", "entry_time"],
                          dtype=float)
    for i in range(len(d)-horizon):
        if not np.isfinite(d.atr.iloc[i]):
            continue
        future = d.iloc[i+1:i+horizon+1]
        if not (np.diff(np.r_[d.open_time.iloc[i], future.open_time.to_numpy()]) == interval_ms).all():
            continue
        entry = float(future.open.iloc[0])
        distance = max(float(d.atr.iloc[i]) * atr_multiple, entry*cost_fraction*1.25)
        upper, lower = entry + distance, entry - distance
        label, exit_price, end = 0, float(future.close.iloc[-1]), int(future.close_time.iloc[-1])+1
        for _, bar in future.iterrows():
            if bar.low <= lower:
                label, exit_price, end = -1, min(float(bar.open), lower), int(bar.close_time)+1
                break
            if bar.high >= upper:
                label, exit_price, end = 1, max(float(bar.open), upper), int(bar.close_time)+1
                break
        net = exit_price/entry - 1 - cost_fraction
        if label == 0:
            label = 1 if net > distance/entry else (-1 if net < -distance/entry else 0)
        result.loc[d.index[i]] = [label, net, end, int(future.open_time.iloc[0])]
    return result


def walk_forward_splits(d: pd.DataFrame, min_train: int, test_size: int, embargo: int = 4,
                        step: int | None = None):
    """Purge labels that extend into the test; embargo additionally removes near-boundary rows."""
    stride = step or test_size
    for start in range(min_train, len(d)-test_size+1, stride):
        cutoff = int(d.available_at.iloc[start])
        eligible = np.arange(max(0, start-embargo))
        train = eligible[d.label_end.iloc[eligible].to_numpy() < cutoff]
        test = np.arange(start, start+test_size)
        if len(train):
            yield train, test

