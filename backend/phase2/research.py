"""Registered Phase 2 experiment. Never promotes historical research to trading."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, accuracy_score
from backend.phase2.dataset import (
    TIMEFRAMES,
    compact_features,
    aligned_windows,
    read_bars,
    chronological_split,
    fit_scale,
    cpcv,
)
from backend.phase2.bars import INTERVALS
from backend.phase2.learning import pretrain, iql_warm_start, tensors
from backend.phase2.regime import RegimeFilter
from backend.phase2.quant import volatility_features, uniqueness_weights
from backend.phase2.rl import MultiTimeframeEnv, train_online
from backend.phase2.validation import evaluate, longest_segment, compact, diagnostics
from backend.phase2.runtime import safe_json
from backend.features.labels import triple_barrier
from backend.phase2.gpu import gpu_details


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.json")
    temp.write_text(json.dumps(safe_json(value), indent=2), encoding="utf-8")
    temp.replace(path)


def encoder_data(root, hour, window=24):
    """Monthly bounded reads with 700-bar causal context and a shared feature function.

    Earlier context primes recursive EMA filters; monthly truncation error is
    documented separately from unavailable receipt-time latency.
    """
    times = hour.available_at.to_numpy(dtype="int64")
    windows = {}
    valid = np.ones(len(hour), dtype=bool)
    import duckdb

    for tf in TIMEFRAMES:
        result = np.zeros((len(hour), window, 8), dtype=np.float32)
        seen = np.zeros(len(hour), dtype=bool)
        context = pd.DataFrame()
        files = sorted((root / "bars/spot/SOLUSDT" / tf).glob("*.parquet"))
        files = [f for f in files if (root / "ticks/spot/SOLUSDT" / (f.stem + ".receipt.json")).exists()]
        with duckdb.connect() as con:
            for file in files:
                partition = pd.Timestamp(file.stem, tz="UTC")
                end = partition + (pd.offsets.MonthBegin(1) if len(file.stem) == 7 else pd.Timedelta(days=1))
                if (
                    end.timestamp() * 1000 < times[0] - 800 * INTERVALS[tf]
                    or partition.timestamp() * 1000 > times[-1]
                ):
                    continue
                raw = con.execute("SELECT * FROM read_parquet(?) ORDER BY open_time", [str(file)]).df()
                if raw.empty:
                    continue
                frame = (
                    pd.concat([context, raw], ignore_index=True)
                    .drop_duplicates("open_time", keep="last")
                    .sort_values("open_time")
                    .reset_index(drop=True)
                )
                # A partition owns decisions strictly before the next UTC partition.
                lo = int(raw.open_time.min())
                hi = int(raw.available_at.max())
                selected = np.flatnonzero((times >= lo) & (times <= hi))
                if len(selected):
                    values, good = aligned_windows({tf: compact_features(frame, tf)}, times[selected], window)
                    result[selected] = values[tf]
                    seen[selected] = good
                context = frame.tail(700).copy()
                del frame, raw
        windows[tf] = result
        valid &= seen
        print(f"ENCODER {tf}: {seen.sum():,} valid hourly windows", flush=True)
    return windows, valid


def asof_aux(root, kind, times):
    file = root / "aux" / kind / "SOLUSDT/history.parquet"
    if not file.exists():
        return pd.DataFrame({"available_at": times})
    d = pd.read_parquet(file).sort_values("available_at").drop_duplicates("available_at", keep="last")
    return pd.merge_asof(
        pd.DataFrame({"available_at": times}),
        d,
        on="available_at",
        direction="backward",
        tolerance=28800000 if kind == "fundingRate" else 600000,
    )


def temperature(logits, labels):
    fit = minimize_scalar(
        lambda t: log_loss(labels, softmax(logits / t, axis=1), labels=[0, 1, 2]),
        bounds=(0.2, 10),
        method="bounded",
    )
    return float(fit.x)


def policy_probabilities(model, windows, state, index, temp=1):
    result = []
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for batch in np.array_split(index, max(1, len(index) // 512)):
            w, s = tensors(windows, state, batch, device)
            result.append(softmax(model(w, s)["direction_logits"].cpu().numpy() / temp, axis=1))
    return np.concatenate(result)


def run(
    root=Path("data/phase2"),
    artifacts=Path("artifacts/phase2"),
    epochs=3,
    rl_steps=10000,
    seeds=(42, 43, 44),
    pilot=False,
):
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=2)
    torch.set_num_threads(2)
    root, artifacts = Path(root), Path(artifacts)
    coverage = json.loads((root / "coverage.json").read_text())
    if not pilot and coverage["status"] not in ["archives_complete", "complete_to_frozen_cutoff"]:
        raise ValueError(
            "Research requires completed raw archives; partial history is not five-year evidence"
        )
    if not pilot and (
        len(coverage["archives"]) < coverage["expected_archives"]
        or any(x["status"] != "complete" for x in coverage["archives"] if "bars" in x)
    ):
        raise ValueError("Incomplete archive inventory")
    identity = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = artifacts / identity
    out.mkdir(parents=True)
    protocol = {
        "version": identity,
        "seeds": list(seeds),
        "epochs": epochs,
        "rl_steps_per_algorithm_seed": rl_steps,
        "holdout_days": 30 if pilot else 365,
        "horizon_hours": 4,
        "purge_hours": 4,
        "cpcv_blocks": 6,
        "cpcv_test_blocks": 2,
        "algorithms": ["pretrained_IQL", "RecurrentPPO", "SAC", "QRDQN"],
        "registered_trials": len(seeds) * 4 + 4 + 3,
        "holdout_status": "Previously observed market period; historical holdout is not pristine prospective evidence",
        "promotion_rule": "Never auto-promote; independent prospective validation is still required",
        "availability_basis": "exchange timestamp, historical receipt latency unverified",
        "input_contract": "8 shared compact features per timeframe; static RV/BV, flow, funding, five-minute OI change and availability masks; news unavailable",
        "missing_inputs": ["historical sequenced L2", "complete liquidation tape", "historical news"],
        "offline_behavior": "generated simulator transitions; no actual trader logs",
        "hardware": gpu_details(),
        "scope": "integration pilot on a completed subset; not five-year validation"
        if pilot
        else "full acquired history",
    }
    save(out / "protocol.json", protocol)
    save(out / "data-snapshot.json", coverage)
    save(artifacts / "latest.json", {"status": "building_dataset", "version": identity, "approved": False})
    hour = read_bars(root, "perpetual", "1h")
    if pilot:
        hour = hour.tail(5000).reset_index(drop=True)
    print(f"DATASET {len(hour):,} perpetual hourly bars; pilot={pilot}", flush=True)
    windows, valid = encoder_data(root, hour)
    core = compact_features(hour, "1h")
    spot = read_bars(root, "spot", "1h").set_index("open_time").reindex(hour.open_time).reset_index(drop=True)
    vol = volatility_features(spot.close)
    funding = asof_aux(root, "fundingRate", hour.available_at.to_numpy())
    # OI change is computed on native five-minute records before the hourly as-of join.
    metric_file = root / "aux/metrics/SOLUSDT/history.parquet"
    oi_change = np.full(len(hour), np.nan)
    if metric_file.exists():
        raw = pd.read_parquet(metric_file).sort_values("available_at").drop_duplicates("available_at")
        if "sum_open_interest" in raw:
            positive = raw.sum_open_interest.where(raw.sum_open_interest > 0)
            raw["oi_change"] = (positive / positive.shift() - 1).where(raw.available_at.diff() == 300000)
            oi_change = pd.merge_asof(
                hour[["available_at"]],
                raw[["available_at", "oi_change"]],
                on="available_at",
                direction="backward",
                tolerance=600000,
            ).oi_change.to_numpy()
    f = funding.get("last_funding_rate", pd.Series(np.nan, index=hour.index)).to_numpy()
    flow = spot.signed_volume_delta.to_numpy() / spot.volume.replace(0, np.nan).to_numpy()
    state = np.column_stack(
        [
            vol.rv,
            vol.bipower_variation,
            flow,
            f,
            oi_change,
            np.zeros(len(hour)),
            np.isfinite(flow),
            np.isfinite(f),
            np.isfinite(oi_change),
            np.zeros(len(hour)),
        ]
    ).astype(np.float32)
    hour["atr"] = core.atr_fraction * hour.close
    labels = triple_barrier(hour)
    returns = hour.open.shift(-5) / hour.open.shift(-1) - 1
    future = np.column_stack([np.log(hour.open.shift(-i) / hour.open.shift(-i + 1)) for i in range(2, 6)])
    finite_future = np.where(np.isfinite(future), future, 0.0)
    volatility = np.std(finite_future, axis=1)
    risk = np.column_stack(
        [
            np.maximum(0, -np.min(np.cumsum(finite_future, axis=1), axis=1)),
            np.maximum(0, -np.min(finite_future, axis=1)),
        ]
    )
    contiguous = np.ones(len(hour), dtype=bool)
    for i in range(1, 6):
        contiguous &= (hour.open_time.shift(-i) - hour.open_time == i * 3600000).fillna(False).to_numpy()
    valid &= (
        np.isfinite(core.to_numpy()).all(axis=1)
        & np.isfinite(state[:, :3]).all(axis=1)
        & labels.label.notna().to_numpy()
        & contiguous
    )
    index = np.flatnonzero(valid)
    if len(index) < (1200 if pilot else 8760 * 4):
        raise ValueError(f"Insufficient valid multi-timeframe decisions: {len(index)}")
    windows = {k: v[index] for k, v in windows.items()}
    state = np.nan_to_num(state[index], nan=0.0, posinf=0.0, neginf=0.0)
    hour = hour.iloc[index].reset_index(drop=True)
    labels = labels.iloc[index].reset_index(drop=True)
    returns = returns.iloc[index].to_numpy(dtype=np.float32)
    volatility = volatility[index].astype(np.float32)
    risk = risk[index].astype(np.float32)
    times = hour.available_at.to_numpy()
    train, calibration, holdout = chronological_split(times, holdout_days=30 if pilot else 365)
    print(f"SPLITS train={len(train)}, calibration={len(calibration)}, holdout={len(holdout)}", flush=True)
    # Tree fold validation uses unscaled values; it must not inherit a scaler
    # fitted on observations assigned to the fold's test partition.
    tree_features = np.concatenate([v[:, -1] for v in windows.values()] + [state], axis=1)
    windows, scalers = fit_scale(windows, train)
    mean = state[train, :6].astype(np.float64).mean(axis=0)
    std = np.maximum(state[train, :6].astype(np.float64).std(axis=0), 1e-6)
    state[:, :6] = np.clip((state[:, :6] - mean) / std, -10, 10)
    model_state = np.column_stack([state, np.zeros((len(state), 5), dtype=np.float32)]).astype(np.float32)
    regime_input = np.column_stack(
        [returns * 0 + core.trend.to_numpy()[index], vol.rv.to_numpy()[index], flow[index]]
    )
    hmm = RegimeFilter().fit(regime_input[train], seeds[0])
    posterior = hmm.filter(regime_input)
    regimes = posterior.argmax(axis=1)
    joblib.dump(hmm, out / "regime.joblib")
    print("HMM fitted on training only; forward-filtered regimes ready", flush=True)
    save(
        out / "scalers.json",
        {
            "timeframes": scalers,
            "state_mean": mean.tolist(),
            "state_std": std.tolist(),
            "counts": {
                "train": len(train),
                "calibration": len(calibration),
                "historical_holdout": len(holdout),
            },
            "recursive_context": "700 preceding bars at partition boundaries; same indicator equations as live",
        },
    )
    funding_file = root / "aux/fundingRate/SOLUSDT/history.parquet"
    funding_events = pd.read_parquet(funding_file) if funding_file.exists() else None
    # Baselines and ablations fit only development data. CPCV refits each fold.
    x = tree_features
    y = labels.label.to_numpy(dtype=int) + 1
    family_columns = {
        "TA-only": np.array([i * 8 + j for i in range(7) for j in [1, 2, 3, 7]]),
        "quant-only": np.array([i * 8 + j for i in range(7) for j in [0, 4]] + [56, 57]),
        "flow-only": np.array([i * 8 + j for i in range(7) for j in [5, 6]] + [58, 59, 60, 62, 63, 64]),
        "full": np.arange(x.shape[1]),
    }
    import optuna

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        storage=f"sqlite:///{(out / 'optuna.sqlite').as_posix()}",
        study_name="development-only",
    )

    def objective(trial):
        candidate = HistGradientBoostingClassifier(
            max_iter=trial.suggest_categorical("max_iter", [60, 100, 140]),
            max_leaf_nodes=trial.suggest_categorical("max_leaf_nodes", [7, 15, 31]),
            l2_regularization=1,
            random_state=42,
        )
        candidate.fit(x[train], y[train])
        return log_loss(y[calibration], candidate.predict_proba(x[calibration]), labels=[0, 1, 2])

    study.optimize(objective, n_trials=3)
    print("Development-only Optuna search completed", flush=True)
    save(
        out / "optimization.json",
        {
            "scope": "GBM calibration period only; no historical-holdout selection",
            "best_params": study.best_params,
            "best_loss": study.best_value,
        },
    )
    cv = []
    development = np.r_[train, calibration]
    for fold, (tr, te) in enumerate(cpcv(times[development], labels.label_end.to_numpy()[development])):
        estimator = HistGradientBoostingClassifier(
            max_iter=60, max_leaf_nodes=15, l2_regularization=1, random_state=42
        )
        estimator.fit(x[development[tr]], y[development[tr]])
        predicted = estimator.predict_proba(x[development[te]])
        cv.append(
            {
                "fold": fold,
                "train_count": len(tr),
                "test_count": len(te),
                "log_loss": log_loss(y[development[te]], predicted, labels=[0, 1, 2]),
                "accuracy": accuracy_score(y[development[te]], predicted.argmax(axis=1)),
            }
        )
    walk = []
    for cut in [0.55, 0.70, 0.85]:
        boundary = int(len(development) * cut)
        te = development[boundary : min(len(development), boundary + 365 * 24)]
        tr = development[:boundary]
        tr = tr[labels.label_end.to_numpy()[tr] < times[te[0]] - 14400000]
        estimator = HistGradientBoostingClassifier(
            max_iter=60, max_leaf_nodes=15, l2_regularization=1, random_state=42
        ).fit(x[tr], y[tr])
        walk.append(
            {
                "train_count": len(tr),
                "test_count": len(te),
                "log_loss": log_loss(y[te], estimator.predict_proba(x[te]), labels=[0, 1, 2]),
            }
        )
    segment = longest_segment(hour, holdout)
    price = hour.iloc[segment].reset_index(drop=True)
    wtest = {k: v[segment] for k, v in windows.items()}
    stest = state[segment]
    rtest = regimes[segment]
    results = {}
    ablations = {}
    gbms = {}
    # Continuous size 0.5 corresponds to one-times exposure under the shared cap.
    results["buy_hold"] = evaluate(wtest, stest, price, lambda i, o: 0.5, funding_events, regimes=rtest)
    momentum = np.tanh(windows["1h"][segment, -1, 1]) * 0.5
    results["momentum"] = evaluate(
        wtest, stest, price, lambda i, o: float(momentum[i]), funding_events, regimes=rtest
    )
    weights = uniqueness_weights(times[train], labels.label_end.to_numpy()[train])
    for family, columns in family_columns.items():
        gbm = HistGradientBoostingClassifier(**study.best_params, l2_regularization=1, random_state=42)
        gbm.fit(x[train][:, columns], y[train], sample_weight=weights)
        pcal = gbm.predict_proba(x[calibration][:, columns])
        temp = temperature(np.log(np.maximum(pcal, 1e-9)), y[calibration])
        p = softmax(np.log(np.maximum(gbm.predict_proba(x[segment][:, columns]), 1e-9)) / temp, axis=1)
        target = np.where(p.max(axis=1) >= 0.58, 0.5 * (p[:, 2] - p[:, 0]), 0)
        result = evaluate(wtest, stest, price, lambda i, o: float(target[i]), funding_events, regimes=rtest)
        ablations[family] = {
            "comparison_scope": "GBM feature-family ablation; not a retrained DRL ablation",
            **compact(result),
        }
        gbms[family] = (gbm, columns, temp)
        if family == "full":
            results["GBM"] = result
    joblib.dump(gbms, out / "baselines.joblib")
    print("GBM folds, baselines and feature-family ablations completed", flush=True)
    save(
        artifacts / "latest.json",
        {"status": "training_encoders_and_agents", "version": identity, "approved": False},
    )
    histories = []
    models = []
    training_segment = longest_segment(hour, train)
    rng = np.random.default_rng(42)
    env = MultiTimeframeEnv(
        {k: v[training_segment] for k, v in windows.items()},
        state[training_segment],
        hour.iloc[training_segment],
        funding=funding_events,
        randomize=True,
        episode_bars=512,
    )
    transitions = []
    for episode in range(3 if pilot else 12):
        observation, _ = env.reset(seed=42 + episode)
        done = False
        while not done:
            local = env.i
            global_index = int(training_segment[local])
            action = float(np.clip(0.4 * windows["1h"][global_index, -1, 1] + rng.normal(0, 0.2), -1, 1))
            previous = observation[-15:].copy()
            observation, reward, done, _, info = env.step(np.array([action], dtype=np.float32))
            transitions.append(
                {
                    "index": global_index,
                    "next_index": int(training_segment[env.i]),
                    "action": action,
                    "reward": reward,
                    "terminal": done,
                    "state": previous,
                    "next_state": observation[-15:].copy(),
                }
            )
    for seed in seeds:
        print(f"PRETRAIN seed={seed}, epochs={epochs}", flush=True)
        model, history = pretrain(
            windows,
            model_state,
            returns,
            volatility,
            regimes,
            train,
            calibration,
            seed,
            epochs,
            risk,
            checkpoint_dir=out / "checkpoints",
        )
        history.pop("validation_logits")
        history["iql"] = iql_warm_start(
            model, windows, model_state, transitions, steps=100 if pilot else 1000, seed=seed
        )
        pcal = policy_probabilities(model, windows, model_state, calibration)
        direction = np.where(returns > 0.003, 2, np.where(returns < -0.003, 0, 1))
        temp = temperature(np.log(np.maximum(pcal, 1e-9)), direction[calibration])
        history["temperature"] = temp
        path = out / f"pretrained-IQL-{seed}.pt"
        torch.save(model.state_dict(), path)
        histories.append(history)
        models.append(model)

        def learned_policy(i, observation):
            device = next(model.parameters()).device
            w = {k: torch.as_tensor(v[i : i + 1], device=device) for k, v in wtest.items()}
            s = torch.as_tensor(observation[-15:][None, :], device=device)
            with torch.no_grad():
                prediction = model(w, s)
            confidence = float(torch.softmax(prediction["direction_logits"] / temp, dim=1).max())
            return float(prediction["action"].item()) if confidence >= 0.58 else 0.0

        results[f"pretrained_IQL-{seed}"] = evaluate(
            wtest, stest, price, learned_policy, funding_events, regimes=rtest
        )
        print(f"PRETRAIN/IQL checkpoint and cost-aware assessment saved for seed {seed}", flush=True)
    save(out / "pretraining.json", histories)
    online = train_online(
        models[0],
        {k: v[train] for k, v in windows.items()},
        state[train],
        hour.iloc[train].reset_index(drop=True),
        out,
        timesteps=rl_steps,
        seeds=seeds,
        funding=funding_events,
        regimes=regimes[train],
    )
    for algorithm, seed, agent in online:
        results[f"{algorithm}-{seed}"] = evaluate(
            wtest, stest, price, agent, funding_events, algorithm, seed, rtest
        )
    baseline = max(results[name]["metrics"]["sharpe"] for name in ["buy_hold", "momentum", "GBM"])
    comparison = {
        name: {
            **compact(value),
            "research_diagnostics": diagnostics(value, protocol["registered_trials"]),
            "beats_best_baseline_sharpe": value["metrics"]["sharpe"] > baseline,
        }
        for name, value in results.items()
    }
    report = {
        "status": "completed_historical_research",
        "version": identity,
        "approved": False,
        "protocol": protocol,
        "results": comparison,
        "ablations": ablations,
        "cpcv": cv,
        "walk_forward": walk,
        "holdout_coverage": {
            "eligible_bars": len(holdout),
            "evaluated_contiguous_bars": len(segment),
            "omitted_bars": len(holdout) - len(segment),
        },
        "remaining_validation": [
            "prospective holdout",
            "neural CPCV/walk-forward retraining",
            "retrained neural family ablations",
            "historical crash policy stress tests",
            "nested comparable search for PBO",
        ],
        "promotion_reason": "Historical comparisons and GBM fold validation do not satisfy full DRL promotion criteria",
    }
    save(out / "validation.json", report)
    save(artifacts / "validation.json", report)
    print("Historical evaluation completed; model remains research-only", flush=True)
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{(out / 'mlflow.sqlite').as_posix()}")
    mlflow.set_experiment("SOLUSDT Phase 2 research")
    with mlflow.start_run(run_name=identity):
        mlflow.log_params({"epochs": epochs, "rl_steps": rl_steps, "seeds": str(seeds), "approved": False})
        for name, value in results.items():
            mlflow.log_metric(name + "_sharpe", value["metrics"]["sharpe"])
        mlflow.log_artifact(str(out / "validation.json"))
    save(
        artifacts / "latest.json",
        {
            "status": "trained_research_only",
            "approved": False,
            "version": identity,
            "directory": str(out.resolve()),
            "seeds": list(seeds),
            "temperatures": [h["temperature"] for h in histories],
        },
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rl-steps", type=int, default=10000)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    run(
        epochs=args.epochs,
        rl_steps=args.rl_steps,
        pilot=args.pilot,
        seeds=(42,) if args.pilot else (42, 43, 44),
        artifacts=Path("artifacts/phase2/pilot") if args.pilot else Path("artifacts/phase2"),
    )
