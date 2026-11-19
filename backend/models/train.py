import hashlib
import json
import uuid
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, mean_pinball_loss
from backend.config import Settings
from backend.features.engine import build_features, FEATURE_COLUMNS, FEATURE_VERSION
from backend.features.labels import triple_barrier, walk_forward_splits
from backend.data.history import candle_path, INTERVAL_MS
from backend.backtest.engine import Costs, simulate, baseline_targets, deflated_sharpe, block_bootstrap


def estimator(seed: int):
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=160, max_depth=4, num_leaves=12, learning_rate=.035,
                              min_child_samples=100, reg_lambda=2, n_jobs=2, random_state=seed, verbosity=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=12,
                                               l2_regularization=2, random_state=seed)


def probabilities(model, x: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(x)
    out = np.zeros((len(x), 3))
    for col, label in enumerate(model.classes_):
        out[:, int(label)+1] = raw[:, col]
    return out


def fit_calibrated(d: pd.DataFrame, seed: int, embargo: int = 4):
    boundary = int(len(d)*.8)
    cutoff = int(d.available_at.iloc[boundary])
    train = d.iloc[:max(0, boundary-embargo)]
    train = train[train.label_end < cutoff]
    calibration = d.iloc[boundary:]
    if len(train) < 300 or len(calibration) < 100 or train.label.nunique() < 2:
        raise ValueError("Insufficient training/calibration coverage")
    base = estimator(seed).fit(train[FEATURE_COLUMNS], train.label.astype(int))
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    except ImportError:
        calibrated = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
    calibrated.fit(calibration[FEATURE_COLUMNS], calibration.label.astype(int))
    return base, calibrated


def sized_target(p: np.ndarray, d: pd.DataFrame, settings: Settings) -> np.ndarray:
    annual_bars = 365 * 86400000 / INTERVAL_MS[settings.timeframe]
    annual = np.maximum(d.realized_vol.to_numpy()*np.sqrt(annual_bars), .05)
    size = np.minimum(settings.max_exposure, settings.target_annual_vol/annual)
    return np.where((p[:, 2] >= settings.min_confidence) & (d.regime_crash.to_numpy() == 0), size, 0)


def probability_metrics(p: np.ndarray, label: np.ndarray) -> dict:
    y = label.astype(int)+1
    onehot = np.eye(3)[y]
    bins = []
    for lo in np.arange(0, 1, .1):
        mask = (p[:, 2] >= lo) & (p[:, 2] < lo+.1)
        if mask.sum():
            bins.append({"bin": float(lo), "count": int(mask.sum()),
                         "predicted": float(p[mask, 2].mean()), "observed": float((y[mask] == 2).mean())})
    return {"log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
            "multiclass_brier": float(np.mean(np.sum((p-onehot)**2, axis=1))), "reliability": bins}


def compact(result: dict) -> dict:
    indexes = np.unique(np.linspace(0, len(result["equity"])-1, min(500, len(result["equity"]))).astype(int))
    return {"metrics": result["metrics"], "regimes": result["regimes"],
            "curve": [{"time": result["time"][i], "equity": result["equity"][i],
                       "drawdown": result["drawdown"][i]} for i in indexes]}


def train(settings: Settings, record_mlflow: bool = False) -> dict:
    file = candle_path(settings.data_dir, settings.symbol, settings.timeframe)
    candles = pd.read_parquet(file)
    step = INTERVAL_MS[settings.timeframe]
    features = build_features(candles, step)
    labels = triple_barrier(features, settings.label_horizon,
                            cost_fraction=2*Costs(settings.fee_bps, settings.slippage_bps,
                                                  settings.spread_bps).one_way, interval_ms=step)
    d = pd.concat([features, labels], axis=1).dropna(subset=FEATURE_COLUMNS+["label", "label_end"])
    d = d.reset_index(drop=True)
    if len(d) < 1500:
        raise ValueError("At least 1500 complete feature/label rows required")
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    dataset_hash = hashlib.sha256(pd.util.hash_pandas_object(candles, index=True).values.tobytes()).hexdigest()
    holdout_start = int(len(d)*.8)
    cutoff = int(d.available_at.iloc[holdout_start])
    development = d.iloc[:holdout_start-settings.label_horizon]
    development = development[development.label_end < cutoff].reset_index(drop=True)
    holdout = d.iloc[holdout_start:].reset_index(drop=True)
    costs = Costs(settings.fee_bps, settings.slippage_bps, settings.spread_bps)
    bars_per_year = 365*86400000/step
    fold_reports = []
    predictions = []
    test_size = max(200, int(len(development)*.15))
    for fold, (tr, te) in enumerate(walk_forward_splits(development, int(len(development)*.5),
                                                     test_size, settings.label_horizon)):
        _, model = fit_calibrated(development.iloc[tr], settings.seed+fold, settings.label_horizon)
        test = development.iloc[te].reset_index(drop=True)
        p = probabilities(model, test[FEATURE_COLUMNS])
        predictions.append(p)
        result = simulate(test, sized_target(p, test, settings), costs, bars_per_year=bars_per_year)
        baselines = {name: simulate(test, target, costs, stop_atr=None, bars_per_year=bars_per_year)["metrics"]
                     for name, target in baseline_targets(test, settings.max_exposure).items()}
        fold_reports.append({"fold": fold, "train_rows": len(tr), "test_rows": len(te),
                             "first_test": int(test.open_time.iloc[0]), "metrics": result["metrics"],
                             "baselines": baselines, "calibration": probability_metrics(p, test.label.to_numpy())})
    base, model = fit_calibrated(development, settings.seed, settings.label_horizon)
    p = probabilities(model, holdout[FEATURE_COLUMNS])
    result = simulate(holdout, sized_target(p, holdout, settings), costs, bars_per_year=bars_per_year)
    baseline_results = {name: simulate(holdout, target, costs, stop_atr=None, bars_per_year=bars_per_year)
                        for name, target in baseline_targets(holdout, settings.max_exposure).items()}
    quantile_models: dict[str, Any] = {}
    quantile_metrics: dict[str, Any] = {}
    try:
        from lightgbm import LGBMRegressor
        for q in [.1, .5, .9]:
            qm = LGBMRegressor(objective="quantile", alpha=q, n_estimators=100,
                               num_leaves=12, min_child_samples=100, n_jobs=2, verbosity=-1,
                               random_state=settings.seed)
            qm.fit(development[FEATURE_COLUMNS], development.forward_return)
            qp = qm.predict(holdout[FEATURE_COLUMNS])
            quantile_models[str(q)] = qm
            quantile_metrics[str(q)] = {"pinball_loss": float(mean_pinball_loss(holdout.forward_return, qp, alpha=q)),
                                       "empirical_below": float((holdout.forward_return <= qp).mean())}
    except ImportError:
        quantile_metrics["status"] = "LightGBM optional dependency missing"
    matched = ["risk_matched_hold", "ma_crossover", "momentum"]
    benchmark = max(baseline_results[name]["metrics"]["sharpe"] for name in matched)
    failures = []
    if result["metrics"]["trade_count"] < settings.min_promotion_trades:
        failures.append("insufficient holdout trades")
    if result["metrics"]["total_return"] <= 0:
        failures.append("nonpositive holdout return after costs")
    if result["metrics"]["sharpe"] <= benchmark:
        failures.append("does not beat risk-matched baseline Sharpe")
    if result["metrics"]["max_drawdown"] < -settings.max_drawdown:
        failures.append("drawdown exceeds configured limit")
    if not fold_reports or sum(f["metrics"]["total_return"] > 0 for f in fold_reports) < len(fold_reports)/2:
        failures.append("insufficient consistent walk-forward performance")
    version = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:8]
    directory = settings.artifact_dir/version
    directory.mkdir()
    bundle = {"model": model, "base": base, "quantiles": quantile_models,
              "features": FEATURE_COLUMNS, "feature_version": FEATURE_VERSION,
              "version": version, "dataset_hash": dataset_hash, "approved": not failures,
              "reference": development[FEATURE_COLUMNS].tail(5000), "settings": settings.model_dump(mode="json")}
    joblib.dump(bundle, directory/"model.joblib")
    sensitivity = []
    for multiplier in [1., 2., 3.]:
        stressed = simulate(holdout, sized_target(p, holdout, settings),
                            Costs(costs.fee_bps*multiplier, costs.slippage_bps*multiplier,
                                  costs.spread_bps*multiplier), bars_per_year=bars_per_year)
        sensitivity.append({"cost_multiplier": multiplier, **stressed["metrics"]})
    stress = {}
    for name, start, end in [("May 2021", "2021-05-01", "2021-06-01"),
                             ("FTX Nov 2022", "2022-11-01", "2022-12-01")]:
        mask = (d.open_time >= pd.Timestamp(start, tz="UTC").timestamp()*1000) & (d.open_time < pd.Timestamp(end, tz="UTC").timestamp()*1000)
        historical = d[mask].reset_index(drop=True)
        # Baselines only: final fitted model may have already seen historical crash windows.
        stress[name] = {"status": "historical baseline stress; not model out-of-sample",
                        "momentum": simulate(historical, baseline_targets(historical)["momentum"], costs,
                                             bars_per_year=bars_per_year)["metrics"]} if len(historical) > 2 else {"status": "no data coverage"}
    report = {"version": version, "dataset_hash": dataset_hash, "feature_version": FEATURE_VERSION,
              "created_at": pd.Timestamp.now(tz="UTC").isoformat(), "rows": len(d),
              "history_first": int(candles.open_time.iloc[0]), "history_last": int(candles.close_time.iloc[-1]),
              "development_last": int(development.close_time.iloc[-1]),
              "holdout_first": int(holdout.open_time.iloc[0]), "holdout_last": int(holdout.close_time.iloc[-1]),
              "approved": not failures, "promotion_failures": failures, "folds": fold_reports,
              "candidate": compact(result), "baselines": {k: compact(v) for k, v in baseline_results.items()},
              "calibration": probability_metrics(p, holdout.label.to_numpy()), "quantiles": quantile_metrics,
              "cost_sensitivity": sensitivity, "historical_stress": stress,
              "deflated_sharpe": deflated_sharpe(np.array(result["returns"]), trials=4+len(fold_reports),
                                                 bars_per_year=bars_per_year),
              "block_bootstrap": block_bootstrap(np.array(result["returns"]), settings.seed),
              "pbo": {"status": "not estimated: insufficient independent strategy configurations"},
              "scope": "OHLCV LightGBM; news/L2/RL are not claimed as trained inputs",
              "holdout_policy": "Retesting this holdout after tuning is exploratory, not untouched validation"}
    (directory/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    (settings.artifact_dir/"latest.json").write_text(json.dumps({"version": version}))
    if not failures:
        champion = settings.artifact_dir/"champion.json"
        if champion.exists():
            (settings.artifact_dir/"previous_champion.json").write_text(champion.read_text())
        champion.write_text(json.dumps({"version": version}))
    if record_mlflow:
        import mlflow
        mlflow.set_experiment("sol-causal-research")
        with mlflow.start_run(run_name=version) as tracking_run:
            report["mlflow_run_id"] = tracking_run.info.run_id
            mlflow.log_params({"seed": settings.seed, "dataset_hash": dataset_hash,
                               "feature_version": FEATURE_VERSION, "horizon": settings.label_horizon})
            mlflow.log_metrics({k: v for k, v in result["metrics"].items() if isinstance(v, (int, float))})
            mlflow.log_artifact(str(directory/"report.json"))
            mlflow.log_artifact(str(directory/"model.joblib"), artifact_path="candidate-model")
        (directory/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def load_bundle(root: Path, champion_only: bool = False) -> dict | None:
    pointer = root/("champion.json" if champion_only else "latest.json")
    if not pointer.exists():
        return None
    version = json.loads(pointer.read_text())["version"]
    # Artifact load is local trusted code only; API never accepts arbitrary model paths.
    bundle = joblib.load(root/version/"model.joblib")
    if (root/("suspended-"+version+".json")).exists():
        bundle["approved"] = False
    return bundle

