"""Deterministic narratives over observed fields; attributions are not causal proof."""

import numpy as np


def explanation(ta, quant, model=None):
    confluence = ta.get("confluence", {})
    liquidation = quant.get("liquidation", {})
    regime = quant.get("regime", {})
    model = model or {}
    uncertainty = model.get("uncertainty")
    confidence = model.get("confidence")
    clusters = liquidation.get("clusters", [])
    status = "HOLD: research model has not passed promotion gates."
    sentences = [status]
    if regime.get("name"):
        sentences.append(f"Trailing regime rules indicate {regime['name']}.")
    if confluence.get("score") is not None:
        sentences.append(f"Multi-timeframe agreement score is {confluence['score']:.2f} on a -1 to +1 scale.")
    if liquidation.get("pressure") is not None:
        sentences.append(
            f"Estimated liquidation-pressure proxy is {liquidation['pressure']:.2f}; this is not observed liquidation flow."
        )
    else:
        sentences.append("Liquidation pressure is unavailable because required inputs are missing.")
    if confidence is not None:
        sentences.append(f"Research direction confidence is {confidence:.1%}.")
    return {
        "action": "HOLD",
        "regime": regime,
        "confluence": confluence,
        "confidence": confidence,
        "uncertainty": uncertainty,
        "clusters": clusters,
        "cascade_risk": liquidation.get("pressure"),
        "derivatives": quant.get("derivatives", {}),
        "attention": model.get("attention", {}),
        "attributions": model.get("attributions", {}),
        "invalidation": model.get("invalidation"),
        "narrative": " ".join(sentences),
        "counterfactual": {
            "status": "unavailable until a calibrated promoted decision model exists",
            "meaning": "Local feature sensitivity is not a feasible market forecast",
        },
        "attribution_disclaimer": "Post-hoc attribution and attention weights describe model influence, not causal proof.",
    }


def ensemble_prediction(probabilities, actions, attention=None, threshold=0.58, max_disagreement=0.15):
    p = np.asarray(probabilities, dtype=float)
    a = np.asarray(actions, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError("Normalized finite direction probabilities required")
    mean = p.mean(axis=0)
    uncertainty = float(p.std(axis=0).max())
    confidence = float(mean.max())
    disagreement = float(a.std())
    entropy = float(-np.sum(mean * np.log(np.maximum(mean, 1e-12))) / np.log(3))
    abstain = confidence < threshold or disagreement > max_disagreement
    return {
        "probabilities": dict(zip(["down", "flat", "up"], mean.tolist())),
        "confidence": confidence,
        "uncertainty": uncertainty,
        "normalized_entropy": entropy,
        "action_disagreement": disagreement,
        "action": "HOLD" if abstain else "LONG" if a.mean() > 0 else "SHORT",
        "target_exposure": 0.0 if abstain else float(a.mean()),
        "attention": attention or {},
        "status": "research_only; independent promotion gate still required",
    }
