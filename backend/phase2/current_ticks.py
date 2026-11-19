"""Frozen current-day aggregate-trade REST backfill with market-specific rate pacing."""

import json
import time
from pathlib import Path
import httpx
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from backend.phase2.bars import (
    normalize_ticks,
    TICK_COLUMNS,
    tick_bars,
    merge_fragments,
    aggregate_bars,
    INTERVALS,
)
from backend.phase2.acquire import sha, catalog


def backfill_current(root: Path, market, cutoff_ms):
    stamp = pd.to_datetime(cutoff_ms, unit="ms", utc=True)
    period = stamp.strftime("%Y-%m-%d")
    midnight = int(stamp.normalize().timestamp() * 1000)
    folder = root / "ticks" / market / "SOLUSDT"
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / f"{period}.parquet"
    receipt = file.with_suffix(".receipt.json")
    if file.exists() and receipt.exists():
        r = json.loads(receipt.read_text())
        if r.get("snapshot_cutoff_ms") == cutoff_ms and sha(file) == r["parquet_sha256"]:
            return r
    url = (
        "https://data-api.binance.vision/api/v3/aggTrades"
        if market == "spot"
        else "https://fapi.binance.com/fapi/v1/aggTrades"
    )
    cursor = None
    rows = 0
    first = last = None
    parts = []
    writer = None
    temp = file.with_suffix(".tmp.parquet")
    try:
        with httpx.Client(timeout=60) as client:
            while True:
                params: dict[str, str | int] = {"symbol": "SOLUSDT", "limit": 1000}
                params.update({"startTime": midnight} if cursor is None else {"fromId": cursor})
                for attempt in range(6):
                    response = client.get(url, params=params)
                    if response.status_code in [418, 429]:
                        time.sleep(min(float(response.headers.get("Retry-After", 2**attempt)), 60))
                        continue
                    response.raise_for_status()
                    break
                else:
                    raise RuntimeError("REST backfill rate limit retries exhausted")
                values = response.json()
                if not values:
                    break
                raw = pd.DataFrame([[r["a"], r["p"], r["q"], r["f"], r["l"], r["T"], r["m"]] for r in values])
                d, q = normalize_ticks(raw)
                if q["invalid_rows"]:
                    raise ValueError("Invalid live REST aggregate trades")
                reached_cutoff = bool((d.timestamp >= cutoff_ms).any())
                d = d[(d.timestamp >= midnight) & (d.timestamp < cutoff_ms)]
                if d.empty:
                    break
                if last is not None and int(d.timestamp.iloc[0]) < last:
                    raise ValueError("REST backfill moved backwards")
                following = int(d.aggregate_id.iloc[-1]) + 1
                if cursor is not None and following <= cursor:
                    raise RuntimeError("REST pagination failed to advance")
                cursor = following
                first = int(d.timestamp.iloc[0]) if first is None else first
                last = int(d.timestamp.iloc[-1])
                rows += len(d)
                table = pa.Table.from_pandas(d[TICK_COLUMNS], preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temp, table.schema, compression="zstd")
                writer.write_table(table)
                parts.append(tick_bars(d, "30s"))
                if rows % 100000 == 0:
                    print(f"CURRENT {market}: {rows:,} aggregate trades", flush=True)
                if reached_cutoff:
                    break
                time.sleep(0.12 if market == "spot" else 0.65)
        if writer:
            writer.close()
            writer = None
            temp.replace(file)
    finally:
        if writer:
            writer.close()
    if not rows:
        return {"market": market, "period": period, "status": "no current-day records", "rows": 0}
    smallest = merge_fragments(parts, "30s")
    bars = {}
    for tf in INTERVALS:
        d = smallest if tf == "30s" else aggregate_bars(smallest, tf)
        d = d[(d.open_time >= midnight) & (d.available_at <= cutoff_ms)]
        if not d.empty:
            out = root / "bars" / market / "SOLUSDT" / tf / f"{period}.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".tmp.parquet")
            d.to_parquet(tmp, index=False, compression="zstd")
            tmp.replace(out)
        bars[tf] = {"rows": len(d)}
    result = {
        "market": market,
        "period": period,
        "status": "complete_to_frozen_cutoff",
        "rows": rows,
        "first_ms": first,
        "last_ms": last,
        "snapshot_cutoff_ms": cutoff_ms,
        "source_url": url,
        "parquet_sha256": sha(file),
        "parquet_bytes": file.stat().st_size,
        "bars": bars,
        "availability_basis": "downloaded REST exchange timestamps; historical receipt latency unverified",
    }
    receipt.write_text(json.dumps(result, indent=2))
    return result


def complete_snapshot(root=Path("data/phase2")):
    root = Path(root)
    file = root / "coverage.json"
    report = json.loads(file.read_text())
    cutoff = int(pd.Timestamp(report["snapshot_cutoff_utc"]).timestamp() * 1000)
    report["current_day"] = [backfill_current(root, market, cutoff) for market in ["spot", "perpetual"]]
    report["aggregate_trade_rows"] = sum(r.get("rows", 0) for r in report["archives"]) + sum(
        r.get("rows", 0) for r in report["current_day"]
    )
    report["status"] = (
        "complete_to_frozen_cutoff"
        if all(r["status"] == "complete" for r in report["archives"])
        else "complete_with_reported_archive_gaps"
    )
    report["finished_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    file.write_text(json.dumps(report, indent=2))
    catalog(root)
    return report


if __name__ == "__main__":
    complete_snapshot()
