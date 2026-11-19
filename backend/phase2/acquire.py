"""Resumable checksum-verified raw aggregate trades and tick-derived bars."""

import concurrent.futures
import hashlib
import json
import shutil
import time
import zipfile
from typing import Any
from pathlib import Path
import httpx
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from backend.phase2.bars import (
    INTERVALS,
    normalize_ticks,
    tick_bars,
    merge_fragments,
    aggregate_bars,
    TICK_COLUMNS,
)


def sha(file):
    digest = hashlib.sha256()
    with file.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            digest.update(b)
    return digest.hexdigest()


def download(client, url, file):
    for attempt in range(6):
        try:
            with client.stream("GET", url) as r:
                if r.status_code == 404:
                    return False
                r.raise_for_status()
                with file.open("wb") as f:
                    for b in r.iter_bytes(1048576):
                        f.write(b)
            return True
        except httpx.HTTPError:
            if attempt == 5:
                raise
            time.sleep(min(2**attempt, 30))


def trade_archive(root: Path, market: str, period: str, kind="monthly") -> dict:
    folder = root / "ticks" / market / "SOLUSDT"
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / f"{period}.parquet"
    receipt = file.with_suffix(".receipt.json")
    if file.exists() and receipt.exists():
        result = json.loads(receipt.read_text())
        if sha(file) == result["parquet_sha256"]:
            print(f"RESUME {market} {period}", flush=True)
            return result
    route = "spot" if market == "spot" else "futures/um"
    url = f"https://data.binance.vision/data/{route}/{kind}/aggTrades/SOLUSDT/SOLUSDT-aggTrades-{period}.zip"
    zfile = folder / f"{period}.zip"
    with httpx.Client(timeout=180, follow_redirects=True) as client:
        c = client.get(url + ".CHECKSUM")
        if c.status_code == 404:
            return {
                "market": market,
                "period": period,
                "status": "unavailable",
                "url": url,
            }
        c.raise_for_status()
        expected = c.text.split()[0]
        if not zfile.exists() or sha(zfile) != expected:
            if shutil.disk_usage(root).free < 2_000_000_000:
                raise RuntimeError("Insufficient disk space: paused before archive download")
            download(client, url, zfile)
        if sha(zfile) != expected:
            raise ValueError("Archive checksum mismatch")
    parts = []
    writer = None
    previous_id = previous_time = None
    quality: dict[str, Any] = {
        "invalid_rows": 0,
        "duplicate_rows": 0,
        "id_gap_segments": 0,
        "large_price_jumps": 0,
        "source_order_regressions": 0,
    }
    rows = 0
    first = last = last_price = None
    temp = file.with_suffix(".tmp.parquet")
    try:
        with zipfile.ZipFile(zfile) as z:
            name = next(n for n in z.namelist() if n.endswith(".csv"))
            for raw in pd.read_csv(z.open(name), header=None, usecols=range(7), chunksize=100000):
                ticks, q = normalize_ticks(raw, enforce_order=False)
                for key, value in q.items():
                    quality[key] += value
                if ticks.empty:
                    continue
                if previous_id is not None:
                    assert previous_time is not None
                    if (
                        int(ticks.aggregate_id.iloc[0]) <= previous_id
                        or int(ticks.timestamp.iloc[0]) < previous_time
                    ):
                        quality["source_order_regressions"] += 1
                    quality["id_gap_segments"] += int(ticks.aggregate_id.iloc[0] > previous_id + 1)
                quality["id_gap_segments"] += int((ticks.aggregate_id.diff() > 1).sum())
                quality["source_order_regressions"] += int(
                    ((ticks.timestamp.diff() < 0) | (ticks.aggregate_id.diff() <= 0)).sum()
                )
                returns = ticks.price.pct_change().abs()
                quality["large_price_jumps"] += int((returns > 0.2).sum())
                if last_price is not None:
                    quality["large_price_jumps"] += int(abs(ticks.price.iloc[0] / last_price - 1) > 0.2)
                previous_id, previous_time = (
                    int(ticks.aggregate_id.iloc[-1]),
                    int(ticks.timestamp.iloc[-1]),
                )
                last_price = float(ticks.price.iloc[-1])
                first = (
                    int(ticks.timestamp.min()) if first is None else min(first, int(ticks.timestamp.min()))
                )
                last = int(ticks.timestamp.max()) if last is None else max(last, int(ticks.timestamp.max()))
                rows += len(ticks)
                table = pa.Table.from_pandas(ticks[TICK_COLUMNS], preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temp, table.schema, compression="zstd")
                writer.write_table(table)
                parts.append(tick_bars(ticks, "30s"))
        if writer is not None:
            writer.close()
            writer = None
        if rows == 0:
            raise ValueError("Empty trade archive")
        if quality["source_order_regressions"]:
            # Some official files interleave large stretches of aggregate IDs.
            # Sort the entire partition, not individual CSV chunks, preserving
            # exchange timestamps and the checksum-verified original ZIP.
            import duckdb

            ordered = file.with_suffix(".ordered.parquet")
            with duckdb.connect() as con:
                con.execute("SET memory_limit='512MB'")
                con.execute("SET threads=2")
                con.execute("SET temp_directory=?", [str(folder / f"{period}.sort-spill")])
                duplicate_result = con.execute(
                    "SELECT count(*)-count(DISTINCT aggregate_id) FROM read_parquet(?)", [str(temp)]
                ).fetchone()
                assert duplicate_result is not None
                duplicates = duplicate_result[0]
                quality["duplicate_rows"] += int(duplicates)
                source = temp.as_posix().replace("'", "''")
                target = ordered.as_posix().replace("'", "''")
                con.execute(
                    f"COPY (SELECT * FROM read_parquet('{source}') QUALIFY row_number() OVER (PARTITION BY aggregate_id ORDER BY timestamp, first_id)=1 ORDER BY timestamp, aggregate_id) TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)"
                )
            ordered.replace(temp)
            parts = []
            rows = 0
            for batch in pq.ParquetFile(temp).iter_batches(batch_size=100000):
                ticks, _ = normalize_ticks(batch.to_pandas(), enforce_order=False)
                parts.append(tick_bars(ticks, "30s"))
                rows += len(ticks)
            quality["ordering_repair"] = (
                "whole-partition stable exchange-time/aggregate-ID sort; source timestamps preserved"
            )
            quality["id_gap_segments_basis"] = "source order; interleaving can inflate this count"
        temp.replace(file)
    finally:
        if writer is not None:
            writer.close()
    bars = {}
    period_start = int(pd.Timestamp(period, tz="UTC").timestamp() * 1000)
    period_end = int(
        (
            pd.Timestamp(period, tz="UTC")
            + (pd.offsets.MonthBegin(1) if kind == "monthly" else pd.Timedelta(days=1))
        ).timestamp()
        * 1000
    )
    smallest = merge_fragments(parts, "30s")
    for tf in INTERVALS:
        frame = smallest if tf == "30s" else aggregate_bars(smallest, tf)
        frame = frame[(frame.open_time >= period_start) & (frame.available_at <= period_end)]
        out = root / "bars" / market / "SOLUSDT" / tf / f"{period}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(out, index=False, compression="zstd")
        bars[tf] = {
            "rows": len(frame),
            "gap_segments": int((frame.open_time.diff() > INTERVALS[tf]).sum()),
            "bytes": out.stat().st_size,
        }
    result = {
        "market": market,
        "period": period,
        "status": "complete",
        "rows": rows,
        "first_ms": first,
        "last_ms": last,
        "source_url": url,
        "archive_sha256": expected,
        "parquet_sha256": sha(file),
        "parquet_bytes": file.stat().st_size,
        "zip_bytes": zfile.stat().st_size,
        "quality": quality,
        "bars": bars,
        "availability_basis": "historical exchange event time; unverified receipt latency",
    }
    receipt.write_text(json.dumps(result, indent=2))
    print(f"COMPLETE {market} {period}: {rows:,} aggregate trades", flush=True)
    return result


def catalog(root):
    import duckdb

    con = duckdb.connect(str(root / "catalog.duckdb"))
    for market in ["spot", "perpetual"]:
        for tf in INTERVALS:
            path = (root / "bars" / market / "SOLUSDT" / tf / "*.parquet").as_posix().replace("'", "''")
            if list((root / "bars" / market / "SOLUSDT" / tf).glob("*.parquet")):
                con.execute(f"CREATE OR REPLACE VIEW {market}_{tf} AS SELECT * FROM read_parquet('{path}')")
    con.close()


def acquire(root=Path("data/phase2"), start="2020-09", workers=2):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC")
    months = pd.period_range(start, now.strftime("%Y-%m"), freq="M")[:-1]
    days = pd.date_range(now.strftime("%Y-%m-01"), now.strftime("%Y-%m-%d"), freq="D")[:-1]
    tasks = [(market, str(month), "monthly") for month in months for market in ["spot", "perpetual"]]
    tasks += [(market, day.strftime("%Y-%m-%d"), "daily") for day in days for market in ["spot", "perpetual"]]
    report = {
        "status": "downloading",
        "snapshot_cutoff_utc": now.isoformat(),
        "archives": [],
        "expected_archives": len(tasks),
        "raw_trades_kind": "aggregated, not individual fills",
    }
    manifest = root / "coverage.json"

    def save():
        temp = manifest.with_suffix(".tmp.json")
        temp.write_text(json.dumps(report, indent=2))
        temp.replace(manifest)

    save()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(lambda task: trade_archive(root, *task), tasks):
                report["archives"].append(result)
                save()
        catalog(root)
        report["status"] = (
            "complete_with_gaps"
            if any(r["status"] != "complete" for r in report["archives"])
            else "archives_complete"
        )
        report["current_day"] = "not yet archived; live capture/REST backfill required"
        report["aggregate_trade_rows"] = sum(r.get("rows", 0) for r in report["archives"])
        report["finished_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
        save()
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        save()
        raise
    return report


if __name__ == "__main__":
    acquire()
