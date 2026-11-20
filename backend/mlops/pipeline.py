"""Reproducible, promotion-gated MLOps pipeline for the SOL/USDT baseline."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend.backtest.engine import Costs, simulate
from backend.config import Settings
from backend.data.history import INTERVAL_MS, candle_path
from backend.features.engine import FEATURE_COLUMNS, FEATURE_VERSION, build_features
from backend.features.labels import triple_barrier
from backend.models.train import probabilities, probability_metrics, sized_target, train

PIPELINE_VERSION = "supertrader-mlops-v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False, default=_json_default), encoding="utf-8")
    os.replace(temp, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _RunLock:
    def __init__(self, path: Path, run_id: str):
        self.path = path
        self.run_id = run_id
        self.fd: int | None = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            details = self.path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(f"An MLOps run already holds {self.path.name}: {details[:300]}") from exc
        os.write(self.fd, json.dumps({"run_id": self.run_id, "pid": os.getpid(), "started_at": _utc_now()}).encode())
        os.close(self.fd)
        self.fd = None
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.path.unlink(missing_ok=True)


def _snapshot_and_validate(settings: Settings, run_dir: Path) -> tuple[Path, dict]:
    source = candle_path(settings.data_dir, settings.symbol, settings.timeframe)
    if not source.is_file():
        raise FileNotFoundError(f"Canonical candle dataset is missing: {source}")

    source_hash_before = _sha256(source)
    snapshot_root = run_dir / "data_snapshot"
    snapshot = candle_path(snapshot_root, settings.symbol, settings.timeframe)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, snapshot)
    snapshot_hash = _sha256(snapshot)
    source_hash_after = _sha256(source)
    if source_hash_before != source_hash_after or snapshot_hash != source_hash_after:
        raise RuntimeError("Candle data changed while the immutable training snapshot was being made")

    candles = pd.read_parquet(snapshot)
    required = {"open_time", "close_time", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(candles.columns))
    if missing:
        raise ValueError(f"Candle schema is missing required columns: {', '.join(missing)}")
    if len(candles) < 2_000:
        raise ValueError(f"Only {len(candles)} candle rows are present; at least 2000 are required for this pipeline")
    if candles[list(required)].isna().any().any():
        raise ValueError("Candle data contains null values in required market columns")
    if not candles.open_time.is_monotonic_increasing or candles.open_time.duplicated().any():
        raise ValueError("Candle open times must be sorted and unique before training")

    step = INTERVAL_MS[settings.timeframe]
    if (candles.open_time.astype("int64") % step != 0).any():
        raise ValueError("Candle timestamps are not aligned to their UTC interval")
    expected_close = candles.open_time.astype("int64") + step - 1
    if not np.array_equal(candles.close_time.astype("int64").to_numpy(), expected_close.to_numpy()):
        raise ValueError("The dataset contains incomplete or misaligned candles")
    if (candles[["open", "high", "low", "close"]] <= 0).any().any() or (candles.volume < 0).any():
        raise ValueError("Candle data contains invalid prices or negative volume")
    bounds = candles[["open", "close", "low"]].max(axis=1)
    upper_bounds = candles[["open", "close", "high"]].min(axis=1)
    if (candles.high < bounds).any() or (candles.low > upper_bounds).any():
        raise ValueError("Candle OHLC bounds are inconsistent")

    now_ms = int(time.time() * 1000)
    latest_close = int(candles.close_time.iloc[-1])
    age_hours = max(0.0, (now_ms - latest_close - 1) / 3_600_000)
    if age_hours > (2 * step / 3_600_000):
        raise ValueError(f"Latest completed candle is {age_hours:.2f} hours old; refresh data before training")

    gaps = int(((candles.open_time.diff().dropna().astype("int64") // step) - 1).clip(lower=0).sum())
    features = build_features(candles, step)
    labels = triple_barrier(
        features,
        settings.label_horizon,
        cost_fraction=2
        * Costs(settings.fee_bps, settings.slippage_bps, settings.spread_bps).one_way,
        interval_ms=step,
    )
    eligible = pd.concat([features, labels], axis=1).dropna(subset=FEATURE_COLUMNS + ["label", "label_end"])
    if len(eligible) < 1_500:
        raise ValueError(f"Only {len(eligible)} complete feature/label rows remain; at least 1500 are required")

    data_hash = hashlib.sha256(pd.util.hash_pandas_object(candles, index=True).values.tobytes()).hexdigest()
    summary = {
        "status": "passed",
        "source_path": str(source.resolve()),
        "snapshot_path": str(snapshot.resolve()),
        "file_sha256": snapshot_hash,
        "dataset_hash": data_hash,
        "rows": int(len(candles)),
        "eligible_feature_label_rows": int(len(eligible)),
        "label_counts": {str(int(key)): int(value) for key, value in eligible.label.value_counts().sort_index().items()},
        "timeframe": settings.timeframe,
        "first_open_ms": int(candles.open_time.iloc[0]),
        "latest_close_ms": latest_close,
        "latest_age_hours": round(age_hours, 4),
        "missing_bar_count": gaps,
        "feature_version": FEATURE_VERSION,
        "feature_columns": len(FEATURE_COLUMNS),
        "availability_basis": str(candles.get("availability_basis", pd.Series(["exchange_bar_close_assumed"])).iloc[-1]),
    }
    return snapshot_root, summary


def _holdout_frame(settings: Settings, snapshot_root: Path) -> pd.DataFrame:
    candles = pd.read_parquet(candle_path(snapshot_root, settings.symbol, settings.timeframe))
    step = INTERVAL_MS[settings.timeframe]
    features = build_features(candles, step)
    labels = triple_barrier(
        features,
        settings.label_horizon,
        cost_fraction=2 * Costs(settings.fee_bps, settings.slippage_bps, settings.spread_bps).one_way,
        interval_ms=step,
    )
    data = pd.concat([features, labels], axis=1).dropna(subset=FEATURE_COLUMNS + ["label", "label_end"])
    data = data.reset_index(drop=True)
    return data.iloc[int(len(data) * 0.8) :].reset_index(drop=True)


def _score_bundle(bundle: dict, holdout: pd.DataFrame, settings: Settings) -> dict:
    if bundle.get("feature_version") != FEATURE_VERSION or bundle.get("features") != FEATURE_COLUMNS:
        raise ValueError("Model feature schema is incompatible with the current evaluation pipeline")
    p = probabilities(bundle["model"], holdout[FEATURE_COLUMNS])
    target = sized_target(p, holdout, settings)
    result = simulate(
        holdout,
        target,
        Costs(settings.fee_bps, settings.slippage_bps, settings.spread_bps),
        bars_per_year=365 * 86_400_000 / INTERVAL_MS[settings.timeframe],
    )
    return {"metrics": result["metrics"], "calibration": probability_metrics(p, holdout.label.to_numpy())}


def _challenger_gate(
    settings: Settings,
    report: dict,
    candidate_bundle: dict,
    snapshot_root: Path,
) -> tuple[list[str], dict]:
    failures: list[str] = []
    champion_file = settings.artifact_dir / "champion.json"
    if not champion_file.exists():
        return failures, {"status": "no approved champion exists; baseline and walk-forward gates apply"}

    champion_version = json.loads(champion_file.read_text(encoding="utf-8"))["version"]
    champion_path = settings.artifact_dir / champion_version
    champion_report_path = champion_path / "report.json"
    if not champion_report_path.is_file():
        return ["existing champion report is missing; automatic promotion is blocked"], {
            "status": "blocked",
            "champion_version": champion_version,
        }
    champion_report = json.loads(champion_report_path.read_text(encoding="utf-8"))
    champion_train_end = int(champion_report.get("development_last", champion_report["holdout_first"] - 1))
    candidate_holdout_start = int(report["holdout_first"])
    comparison = {
        "status": "same candidate holdout",
        "champion_version": champion_version,
        "candidate_holdout_first": candidate_holdout_start,
        "champion_training_through": champion_train_end,
    }
    if candidate_holdout_start <= champion_train_end:
        failures.append("candidate holdout is not later than the champion training window")
        comparison["status"] = "blocked: holdout is not prospective to the champion"
        return failures, comparison

    try:
        champion_bundle = joblib.load(champion_path / "model.joblib")
        candidate_holdout = _holdout_frame(settings, snapshot_root)
        champion_score = _score_bundle(champion_bundle, candidate_holdout, settings)
        candidate_score = _score_bundle(candidate_bundle, candidate_holdout, settings)
    except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
        failures.append(f"candidate-versus-champion comparison failed: {type(exc).__name__}")
        comparison["status"] = "blocked: evaluation could not be completed"
        return failures, comparison

    candidate_metrics = candidate_score["metrics"]
    champion_metrics = champion_score["metrics"]
    candidate_brier = candidate_score["calibration"]["multiclass_brier"]
    champion_brier = champion_score["calibration"]["multiclass_brier"]
    comparison.update(
        {
            "candidate_metrics": candidate_metrics,
            "champion_metrics": champion_metrics,
            "candidate_brier": candidate_brier,
            "champion_brier": champion_brier,
        }
    )
    if candidate_metrics["sharpe"] <= champion_metrics["sharpe"]:
        failures.append("candidate does not beat the champion's Sharpe on the common holdout")
    if candidate_metrics["total_return"] <= champion_metrics["total_return"]:
        failures.append("candidate does not beat the champion's net return on the common holdout")
    if candidate_metrics["trade_count"] < settings.min_promotion_trades:
        failures.append("candidate has too few common-holdout trades for promotion")
    if candidate_brier > champion_brier + 0.02:
        failures.append("candidate calibration is materially worse than the champion on the common holdout")
    return failures, comparison


def _register(settings: Settings, manifest: dict, candidate_dir: Path, candidate_version: str) -> None:
    if manifest["promotion"]["approved"]:
        registered_dir = settings.artifact_dir / candidate_version
        if registered_dir.exists():
            raise FileExistsError(f"Artifact version already exists: {registered_dir}")
        shutil.copytree(candidate_dir, registered_dir)

        champion = settings.artifact_dir / "champion.json"
        previous = settings.artifact_dir / "previous_champion.json"
        if champion.exists():
            _write_json(previous, json.loads(champion.read_text(encoding="utf-8")))
        pointer = {"version": candidate_version, "registered_at": _utc_now(), "pipeline_run": manifest["run_id"]}
        _write_json(champion, pointer)
        _write_json(settings.artifact_dir / "latest.json", pointer)
        manifest["registration"] = {"status": "promoted", "artifact_path": str(registered_dir.resolve())}
    else:
        manifest["registration"] = {
            "status": "candidate_not_promoted",
            "artifact_path": str(candidate_dir.resolve()),
            "active_artifacts_unchanged": True,
        }


def _record_run(settings: Settings, manifest: dict) -> None:
    registry_root = settings.artifact_dir / "mlops"
    registry_file = registry_root / "registry.json"
    registry: dict[str, Any] = json.loads(registry_file.read_text(encoding="utf-8")) if registry_file.exists() else {
        "schema_version": 1,
        "runs": [],
    }
    registry["runs"] = [row for row in registry["runs"] if row.get("run_id") != manifest["run_id"]]
    promotion = manifest.get("promotion", {})
    registration = manifest.get("registration", {})
    registry["runs"].append(
        {
            "run_id": manifest["run_id"],
            "candidate_version": manifest.get("candidate", {}).get("version"),
            "status": manifest["status"],
            "approved": bool(promotion.get("approved", False)),
            "promoted": registration.get("status") == "promoted",
            "run_manifest": str(manifest["manifest_path"]),
            "completed_at": manifest["completed_at"],
        }
    )
    registry["runs"] = registry["runs"][-500:]
    _write_json(registry_file, registry)
    _write_json(registry_root / "latest-run.json", {"run_id": manifest["run_id"]})


def run_pipeline(settings: Settings, record_mlflow: bool = True) -> dict:
    """Snapshot data, train/evaluate a candidate, and publish only after all gates pass."""
    root = settings.artifact_dir / "mlops"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "pipeline.json"
    lock_path = root / "pipeline.lock"
    manifest: dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "run_id": run_id,
        "status": "running",
        "started_at": _utc_now(),
        "completed_at": None,
        "manifest_path": str(manifest_path.resolve()),
        "config": settings.model_dump(mode="json"),
        "config_sha256": hashlib.sha256(
            json.dumps(settings.model_dump(mode="json"), sort_keys=True, default=_json_default).encode()
        ).hexdigest(),
        "stages": {},
    }

    def save() -> None:
        _write_json(manifest_path, manifest)

    with _RunLock(lock_path, run_id):
        save()
        try:
            manifest["stages"]["snapshot_and_preflight"] = {"status": "running", "started_at": _utc_now()}
            save()
            snapshot_root, dataset = _snapshot_and_validate(settings, run_dir)
            manifest["dataset"] = dataset
            manifest["stages"]["snapshot_and_preflight"].update(
                {"status": "passed", "completed_at": _utc_now()}
            )
            save()

            candidate_artifact_root = run_dir / "training_artifacts"
            candidate_settings = settings.model_copy(
                update={"data_dir": snapshot_root, "artifact_dir": candidate_artifact_root}
            )
            mlflow_status: dict = {"status": "disabled"}
            if record_mlflow:
                try:
                    import mlflow

                    tracking_db = root / "mlflow.sqlite"
                    tracking_db.parent.mkdir(parents=True, exist_ok=True)
                    mlflow.set_tracking_uri(f"sqlite:///{tracking_db.resolve().as_posix()}")
                    mlflow_status = {"status": "enabled", "tracking_uri": mlflow.get_tracking_uri()}
                except ImportError:
                    mlflow_status = {"status": "skipped", "reason": "optional mlflow dependency is not installed"}
                    record_mlflow = False
            manifest["mlflow"] = mlflow_status

            manifest["stages"]["candidate_training"] = {"status": "running", "started_at": _utc_now()}
            save()
            report = train(candidate_settings, record_mlflow=record_mlflow)
            version = str(report["version"])
            candidate_dir = candidate_artifact_root / version
            model_path = candidate_dir / "model.joblib"
            report_path = candidate_dir / "report.json"
            if not model_path.is_file() or not report_path.is_file():
                raise RuntimeError("Training completed without a complete model/report artifact pair")

            candidate_bundle = joblib.load(model_path)
            if candidate_bundle.get("version") != version:
                raise RuntimeError("Candidate model and training report versions do not match")
            if candidate_bundle.get("dataset_hash") != report.get("dataset_hash"):
                raise RuntimeError("Candidate model and report dataset hashes do not match")
            if candidate_bundle.get("feature_version") != FEATURE_VERSION:
                raise RuntimeError("Candidate was built with an unexpected feature schema")
            if not report.get("approved") and candidate_bundle.get("approved"):
                raise RuntimeError("Candidate model claims approval when its validation report does not")

            manifest["candidate"] = {
                "version": version,
                "approved_by_training_gates": bool(report.get("approved")),
                "training_report": str(report_path.resolve()),
                "model_sha256": _sha256(model_path),
                "promotion_failures": list(report.get("promotion_failures", [])),
            }
            manifest["stages"]["candidate_training"].update(
                {"status": "passed", "completed_at": _utc_now(), "candidate_version": version}
            )
            save()

            manifest["stages"]["quality_gate"] = {"status": "running", "started_at": _utc_now()}
            save()
            promotion_failures = list(report.get("promotion_failures", []))
            challenger_failures, challenger_comparison = _challenger_gate(
                settings, report, candidate_bundle, snapshot_root
            )
            promotion_failures.extend(challenger_failures)
            approved = bool(report.get("approved")) and not challenger_failures
            manifest["promotion"] = {
                "approved": approved,
                "failures": promotion_failures,
                "challenger_comparison": challenger_comparison,
            }
            manifest["stages"]["quality_gate"].update(
                {"status": "passed", "completed_at": _utc_now(), "approved": approved}
            )
            save()

            manifest["stages"]["registration"] = {"status": "running", "started_at": _utc_now()}
            save()
            _register(settings, manifest, candidate_dir, version)
            manifest["stages"]["registration"].update(
                {"status": manifest["registration"]["status"], "completed_at": _utc_now()}
            )
            manifest["monitoring"] = {
                "status": "continuous_runtime_monitoring",
                "health_endpoint": "/health",
                "signals_and_outcomes": "SignalService records matured outcomes and reports calibration/drift health",
                "candidate_serving": "enabled" if approved else "not enabled; candidate remains isolated",
                "note": "Offline holdout evidence is not a substitute for prospective paper validation.",
            }
            manifest["stages"]["monitoring"] = {"status": "configured", "completed_at": _utc_now()}
            manifest["status"] = "completed_and_promoted" if approved else "completed_candidate_not_promoted"
            manifest["completed_at"] = _utc_now()
            save()
            _record_run(settings, manifest)
            _write_json(root / "latest-run.json", {"run_id": run_id})
            return manifest
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["failed_at"] = _utc_now()
            manifest["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
            save()
            _record_run(settings, manifest)
            _write_json(root / "latest-run.json", {"run_id": run_id})
            raise


def pipeline_status(settings: Settings) -> dict:
    root = settings.artifact_dir / "mlops"
    pointer = root / "latest-run.json"
    if not pointer.is_file():
        return {"status": "not_run", "pipeline_version": PIPELINE_VERSION, "runs": 0}
    run_id = json.loads(pointer.read_text(encoding="utf-8"))["run_id"]
    manifest_path = root / "runs" / run_id / "pipeline.json"
    if not manifest_path.is_file():
        return {"status": "incomplete", "run_id": run_id, "manifest": str(manifest_path)}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "status": manifest.get("status", "unknown"),
        "pipeline_version": manifest.get("pipeline_version"),
        "run_id": run_id,
        "started_at": manifest.get("started_at"),
        "completed_at": manifest.get("completed_at"),
        "candidate": manifest.get("candidate"),
        "promotion": manifest.get("promotion"),
        "stages": manifest.get("stages", {}),
        "manifest": str(manifest_path.resolve()),
    }
