"""One local staged training job: wait for active acquisition, audit, train, evaluate."""

import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
from backend.phase2.research import run, save
from backend.phase2.current_ticks import complete_snapshot
from backend.phase2.acquire import acquire
from backend.phase2.gpu import gpu_details


def quality_report(root):
    coverage = json.loads((root / "coverage.json").read_text())
    aux = json.loads((root / "auxiliary-coverage.json").read_text())
    archives = coverage["archives"]
    report = {
        "status": "ingestion_receipts_audited",
        "snapshot_cutoff_utc": coverage["snapshot_cutoff_utc"],
        "archive_count": len(archives),
        "expected_archives": coverage["expected_archives"],
        "aggregate_trade_rows": sum(a.get("rows", 0) for a in archives),
        "parquet_bytes": sum(a.get("parquet_bytes", 0) for a in archives),
        "original_zip_bytes": sum(a.get("zip_bytes", 0) for a in archives),
        "invalid_rows": sum(a.get("quality", {}).get("invalid_rows", 0) for a in archives),
        "duplicate_rows_removed": sum(a.get("quality", {}).get("duplicate_rows", 0) for a in archives),
        "source_order_repairs": [
            {"market": a["market"], "period": a["period"], "quality": a["quality"]}
            for a in archives
            if a.get("quality", {}).get("source_order_regressions", 0)
        ],
        "outlier_flags": [
            {"market": a["market"], "period": a["period"], "count": a["quality"]["large_price_jumps"]}
            for a in archives
            if a.get("quality", {}).get("large_price_jumps", 0)
        ],
        "bar_gap_segments": {
            tf: sum(a.get("bars", {}).get(tf, {}).get("gap_segments", 0) for a in archives)
            for tf in ["30s", "1m", "5m", "15m", "1h", "2h", "4h", "6h", "12h", "1d"]
        },
        "auxiliary_missing_archives": sum(a["status"] == "unavailable" for a in aux["archives"]),
        "timezone": "UTC; epoch milliseconds; source microseconds normalized",
        "checksum_basis": "official ZIP checksum and normalized Parquet SHA256 verified at ingestion; receipts audited here",
        "gap_cause": "no-trade intervals and source outages cannot be distinguished from trade records alone",
        "survivorship": "single SOL instrument; results do not generalize to delisted instruments",
        "historical_liquidations": "complete event history unavailable; estimated proxies only",
        "historical_book": "percent-depth summaries; sequenced L2 unavailable",
        "current_day": coverage.get("current_day"),
    }
    save(root / "data-quality.json", report)
    return report


def pipeline(root=Path("data/phase2"), artifacts=Path("artifacts/phase2"), epochs=10, rl_steps=30000):
    status_file = artifacts / "job.json"
    started = pd.Timestamp.now(tz="UTC").isoformat()
    hardware = gpu_details()

    def status(stage, **details):
        value = {
            "status": stage,
            "pid": os.getpid(),
            "started_at_utc": started,
            "updated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "epochs": epochs,
            "rl_steps_per_algorithm_seed": rl_steps,
            "seeds": [42, 43, 44],
            "hardware": hardware,
            **details,
        }
        save(status_file, value)
        print(json.dumps(value), flush=True)

    try:
        coverage = json.loads((root / "coverage.json").read_text())
        if coverage["status"] not in ["archives_complete", "complete_to_frozen_cutoff"]:
            status(
                "resuming_acquisition",
                completed_archives=len(coverage.get("archives", [])),
                expected_archives=coverage.get("expected_archives"),
            )
            coverage = acquire(root, workers=3)
        # Rate-paced current-day backfill runs alongside model training, after
        # the completed-archive dataset has been frozen for the experiment.
        backfill_pool = ThreadPoolExecutor(max_workers=1)
        backfill = (
            backfill_pool.submit(complete_snapshot, root)
            if coverage["status"] == "archives_complete"
            else None
        )
        status("auditing_dataset")
        quality_report(root)
        status("training_and_evaluating")
        result = run(root, artifacts, epochs, rl_steps)
        if backfill is not None:
            status("finishing_current_day_backfill", version=result["version"])
            backfill.result()
            quality_report(root)
        backfill_pool.shutdown()
        status(
            "completed_historical_research",
            version=result["version"],
            approved=False,
            remaining_validation=result["remaining_validation"],
        )
    except Exception as exc:
        status("failed", error=str(exc), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    pipeline()
