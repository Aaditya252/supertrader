import numpy as np
import pandas as pd


def population_stability(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    ref, cur = np.asarray(reference), np.asarray(current)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if len(ref) < 50 or len(cur) < 20:
        return 0.
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins+1)))
    if len(edges) < 3:
        return float(abs(np.mean(cur)-np.mean(ref)) > 1e-6)
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(ref, edges)[0]/len(ref)
    b = np.histogram(cur, edges)[0]/len(cur)
    a, b = np.maximum(a, 1e-5), np.maximum(b, 1e-5)
    return float(np.sum((b-a)*np.log(b/a)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, threshold: float) -> dict:
    psi = {c: population_stability(reference[c].to_numpy(), current[c].to_numpy())
           for c in reference.columns if c in current}
    return {"feature_psi": psi, "alerts": [c for c, value in psi.items() if value > threshold],
            "threshold": threshold, "reference_rows": len(reference), "current_rows": len(current),
            "interpretation": "Distribution shift warrants review; it does not prove loss of edge"}


def outcome_summary(outcomes: list[dict]) -> dict:
    if not outcomes:
        return {"count": 0, "accuracy": None, "brier": None, "failure_patterns": {},
                "status": "waiting for completed prediction horizons"}
    errors: dict[str, int] = {}
    brier = []
    correct = []
    for row in outcomes:
        predicted = int(np.argmax([row["probabilities"][k] for k in ["down", "flat", "up"]]))-1
        truth = int(row["label"])
        correct.append(predicted == truth)
        brier.append(sum((row["probabilities"][k] - int(truth == label))**2
                         for k, label in [("down", -1), ("flat", 0), ("up", 1)]))
        if predicted != truth:
            key = f'directional error / {row.get("regime", "unknown")}'
            errors[key] = errors.get(key, 0)+1
    return {"count": len(outcomes), "accuracy": float(np.mean(correct)),
            "brier": float(np.mean(brier)), "failure_patterns": errors,
            "status": "observed outcomes; improvement requires forward candidate comparison"}

