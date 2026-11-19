import hashlib
import json
from typing import Any
import numpy as np
import pandas as pd
from backend.config import Settings
from backend.data.history import INTERVAL_MS
from backend.features.engine import FEATURE_COLUMNS, FEATURE_VERSION
from backend.models.train import probabilities
from backend.models.ensemble import vote
from backend.risk import position_size, risk_reasons


def generate_signal(d: pd.DataFrame, bundle: dict | None, settings: Settings, market: dict,
                    data_age: float | None, paper: dict, news: dict) -> dict:
    clean = d.dropna(subset=FEATURE_COLUMNS)
    latest = d.iloc[-1] if len(d) else None
    now = int(pd.Timestamp.now(tz="UTC").timestamp()*1000)
    decision = int(latest.available_at) if latest is not None else now
    step = INTERVAL_MS[settings.timeframe]
    signal: dict[str, Any] = {"symbol": settings.symbol, "market": settings.market, "action": "HOLD",
              "confidence": None, "probabilities": None, "horizon_bars": settings.label_horizon,
              "probability_definition": f"net-cost triple-barrier outcome over next {settings.label_horizon} {settings.timeframe} bars",
              "direction": None, "direction_confidence": None, "timeframe": settings.timeframe,
              "horizon_minutes": settings.label_horizon * step / 60000,
              "target_exposure": float(paper.get("units", 0)*float(latest.close)/paper["equity"]) if latest is not None else 0,
              "suggested_size_pct": 0., "stop_loss": None, "take_profit": None,
              "risk_reward": None, "price": float(latest.close) if latest is not None else None,
              "decision_time": decision, "generated_at": now, "expires_at": decision+step,
              "regime": str(latest.regime) if latest is not None else "unknown",
              "model_version": bundle["version"] if bundle else None,
              "feature_version": FEATURE_VERSION, "drivers": [], "news": news,
              "data_health": market, "risk_overrides": [], "reasons": [],
              "status": "warming_up", "sizing_basis": "illustrative fraction of paper equity"}
    if latest is None or clean.empty or int(clean.available_at.iloc[-1]) != decision:
        signal["reasons"] = ["insufficient contiguous feature warmup"]
    elif bundle is None:
        signal["reasons"] = ["no trained model; run the research pipeline"]
    elif bundle.get("feature_version", FEATURE_VERSION) != FEATURE_VERSION:
        signal["reasons"] = ["model feature version incompatible; retraining required"]
    else:
        row = clean.iloc[[-1]]
        p = probabilities(bundle["model"], row[FEATURE_COLUMNS])[0]
        probability = {"down": float(p[0]), "flat": float(p[1]), "up": float(p[2])}
        direction = max(probability, key=lambda name: probability[name])
        signal.update(direction=direction, direction_confidence=probability[direction])
        selected = vote([probability], settings.min_confidence)
        validated_settings = bundle.get("settings", {})
        compatible = all(validated_settings.get(k, getattr(settings, k)) == getattr(settings, k)
                         for k in ["symbol", "data_source", "market", "timeframe", "label_horizon", "fee_bps",
                                   "slippage_bps", "spread_bps", "min_confidence", "max_exposure"])
        approved = bool(bundle["approved"] and compatible)
        signal.update(probabilities=probability, confidence=selected["probability"],
                      status="research_only" if not approved else "ready")
        signal["model_approved"] = approved
        signal["data_source"] = settings.data_source
        signal["inputs"] = {c: float(row[c].iloc[0]) for c in FEATURE_COLUMNS}
        signal["reasons"].append(selected["reason"])
        if not approved:
            signal["reasons"].append("candidate failed promotion gates; actionable signals disabled")
            if not compatible:
                signal["reasons"].append("configuration differs from validated model")
        else:
            signal["action"] = selected["action"]
        if signal["action"] == "BUY":
            size = position_size(float(latest.close), float(latest.atr), float(latest.realized_vol), settings)
            signal.update(target_exposure=size, suggested_size_pct=size*100,
                          stop_loss=float(latest.close-1.5*latest.atr),
                          take_profit=float(latest.close+3*latest.atr), risk_reward=2.)
        elif signal["action"] == "SELL":
            signal["target_exposure"] = 0.
        if bundle.get("quantiles"):
            q = sorted(float(model.predict(row[FEATURE_COLUMNS])[0]) for _, model in sorted(bundle["quantiles"].items()))
            signal["return_quantiles"] = {"p10": q[0], "p50": q[1], "p90": q[2]}
            signal["return_distribution_definition"] = "net long barrier-exit returns; not fixed terminal-close returns"
        if hasattr(bundle["base"], "booster_"):
            contributions = np.asarray(bundle["base"].booster_.predict(row[FEATURE_COLUMNS], pred_contrib=True))
            if contributions.shape[1] == (len(FEATURE_COLUMNS)+1)*3:
                class_index = list(bundle["base"].classes_).index(1) if 1 in bundle["base"].classes_ else 0
                values = contributions.reshape(1, 3, len(FEATURE_COLUMNS)+1)[0, class_index, :-1]
                ranked = sorted(zip(FEATURE_COLUMNS, values), key=lambda pair: abs(pair[1]), reverse=True)[:6]
                signal["drivers"] = [{"feature": name, "contribution": float(value)} for name, value in ranked]
                signal["driver_basis"] = "LightGBM TreeSHAP raw up-class margin; not calibrated probability or causation"
    if latest is not None:
        blocked = risk_reasons(settings, market, data_age, float(latest.atr_fraction)
                               if np.isfinite(latest.atr_fraction) else float("inf"),
                               paper.get("daily_return", 0), paper.get("drawdown", 0),
                               gap=clean.empty or int(clean.available_at.iloc[-1]) != decision)
        signal["risk_overrides"] = blocked
        if blocked:
            signal.update(action="HOLD", suggested_size_pct=0., stop_loss=None, take_profit=None,
                          risk_reward=None, status="blocked")
            signal["target_exposure"] = float(paper.get("units", 0)*float(latest.close)/paper["equity"])
    key = json.dumps([decision, signal["model_version"], signal["status"], signal["risk_overrides"]])
    signal["id"] = hashlib.sha256(key.encode()).hexdigest()[:24]
    return signal

