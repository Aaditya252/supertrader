import numpy as np


def vote(probabilities: list[dict], threshold: float = .58) -> dict:
    """Combine calibrated probabilities only; RL exposure remains a separate veto."""
    if not probabilities:
        return {"action": "HOLD", "probability": None, "reason": "no eligible models"}
    directions = ["BUY" if p.get("up", 0) >= threshold else
                  "SELL" if p.get("down", 0) >= threshold else "HOLD" for p in probabilities]
    # HOLD from any model is uncertainty, not agreement with a directional forecast.
    if len(set(directions)) > 1:
        return {"action": "HOLD", "probability": None, "reason": "model disagreement"}
    avg = {k: float(np.mean([p.get(k, 0) for p in probabilities])) for k in ["up", "flat", "down"]}
    action = directions[0]
    probability = avg["up"] if action == "BUY" else avg["down"] if action == "SELL" else avg["flat"]
    return {"action": action, "probability": probability, "probabilities": avg,
            "reason": "calibrated model agreement" if action != "HOLD" else "insufficient direction confidence"}


def purged_stacking_predictions(d, folds, fit_predict):
    """Meta-estimators must consume these forward out-of-fold predictions, never fitted predictions."""
    output = np.full((len(d), 3), np.nan)
    for train, test in folds:
        output[test] = fit_predict(d.iloc[train], d.iloc[test])
    return output

